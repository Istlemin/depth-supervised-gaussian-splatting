#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import glob
import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.general_utils import read_depth_exr_file
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        #center = avg_cam_center
        center = np.array([0.0,0.0,0.0]).reshape((3,1))
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, depth_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        
        depth_path = os.path.join(depth_folder, f"{image_name}.png")
        if Path(depth_path).exists():
            depth = Image.open(depth_path)
        else:
            depth = None

        cam_info = CameraInfo(uid=extr.id, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path, visible_in_cameras=None, camera_ids=None, min_visibility=0):
    
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    if camera_ids is not None:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
        mask = [min_visibility <= len(set(camera_ids) & set(visible_in_cameras[i])) for i in range(len(vertices))]
        return BasicPointCloud(points=positions[mask], colors=colors[mask], normals=normals[mask], visible_in_cameras=[x for x,m in zip(visible_in_cameras,mask) if m])
    else:
        normals = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        return BasicPointCloud(points=positions, colors=colors, normals=normals, visible_in_cameras=None)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, num_train_images=1, min_visibility=0):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    image_folder = os.path.join(path, "images" if images == None else images)
    depth_folder = os.path.join(path, "depth")
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=image_folder, depth_folder=depth_folder)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_idx = [int(round(x)) for x in np.linspace(0,len(cam_infos)-1,num_train_images)]

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in train_idx]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx not in train_idx]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path) or True:
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _, visible_in_cameras = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    # try:
    
    pcd = fetchPly(ply_path,visible_in_cameras, [x.uid for x in train_cam_infos], min_visibility)
    # except:
    #     pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def getTUMIntrinsics(path):
    lines = (path / "intr.txt").read_text().split("\n")
    width, height = lines[0].split(" ")
    fx, _,cx = lines[1].split(" ")
    _, fy,cy = lines[2].split(" ")

    return int(width),int(height),float(fx),float(fy),float(cx),float(cy)

def getTUMExtrinsics(path):
    lines = (path / "traj.txt").read_text().split("\n")
    extrinisics = []
    for i in range(len(lines)//5):
        extr = lines[i*5:i*5+5]
        extr = [[float(x) for x in line.split(" ")] for line in extr]
        
        assert extr[0][0]==i
        assert extr[0][1]==i
        assert extr[0][2]==i+1

        mat = np.array(extr[1:])
        zz = np.eye(4)
        zz[0, 0] = -1
        zz[2, 2] = -1

        mat = np.linalg.inv(np.dot(mat, zz))

        mat[:3,3:4] = -1 * mat[:3,3:4]

        extrinisics.append(mat)    

    return extrinisics

def readTUMSceneInfo(path,num_train_images=-1,max_total_images=200):
    path = Path(path)
    image_folder = path / "rgb"
    depth_folder = path / "depth"
    
    
    cam_infos_unsorted = []
    
    width,height,fx,fy,cx,cy = getTUMIntrinsics(path)
    FovY = focal2fov(fy,height)
    FovX = focal2fov(fx,width)
    extrinsics = list(enumerate(getTUMExtrinsics(path)))
    extrinsics = extrinsics[::len(extrinsics)//max_total_images]
    
    for camera_idx,extr in extrinsics:
        rgb_image_path = image_folder / f"{camera_idx:0>5}.jpg"
        depth_image_path = depth_folder / f"{camera_idx:0>5}.png"
        
        image = Image.open(rgb_image_path)
        depth = Image.open(depth_image_path)
        
        R = extr[:3, :3].T
        T = extr[:3, 3]
        
        camera_info = CameraInfo(uid=camera_idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth,
                              image_path=rgb_image_path, image_name=str(camera_idx), width=width, height=height)
        cam_infos_unsorted.append(camera_info)
        
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_idx = [int(round(x)) for x in np.linspace(0,len(cam_infos)-1,num_train_images)]

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in train_idx]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx not in train_idx]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=None,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=None)

    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=""):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"].replace("\\","/") + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            
            image_path = Path(cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            
            depth_image_path = Path(path) / "depth" / ("Image"+image_name + ".exr")
            depth_data = read_depth_exr_file(depth_image_path)
            depth_data = depth_data*(depth_data<10000)
            depth = Image.fromarray(depth_data*1000)


            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, num_train_images, eval, extension=""):
    print("Reading Training Transforms")
    cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    
    # print("Reading Test Transforms")
    # test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []
    cam_infos = sorted(cam_infos.copy(), key = lambda x : x.image_name)

    
    train_idx = [int(round(x)) for x in np.linspace(0,len(cam_infos)-1,num_train_images)]

    train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in train_idx]
    
    ood_test_angles = [
        "0004",
        "0008",
        "0016",
        "0018",
        "0026",
        "0028",
        "0034",
        "0040",
        "0046",
        "0061",
        "0069",
        "0097"
    ]
    test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx not in train_idx and c.image_name not in ood_test_angles]
    
    
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # ply_path = os.path.join(path, "points3d.ply")
    # if not os.path.exists(ply_path):
    #     # Since this data set has no colmap data, we start with random points
    #     num_pts = 100_000
    #     print(f"Generating random point cloud ({num_pts})...")
        
    #     # We create random points inside the bounds of the synthetic Blender scenes
    #     xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    #     shs = np.random.random((num_pts, 3)) / 255.0
    #     pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    #     storePly(ply_path, xyz, SH2RGB(shs) * 255)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    scene_info = SceneInfo(point_cloud=None,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=None)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}