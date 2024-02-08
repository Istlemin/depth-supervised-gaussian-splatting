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

import os
import random
import json
from re import split
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import readTUMSceneInfo, sceneLoadTypeCallbacks, fetchPly
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path,"traj.txt")):
            scene_info = readTUMSceneInfo(args.source_path,args.num_train_images)
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.num_train_images, args.min_visibility)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            if scene_info.ply_path is not None:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            json_cams = []
            idx = 0
            if scene_info.test_cameras:
                for cam in scene_info.test_cameras:
                    json_cams.append(camera_to_JSON(idx, cam, "test"))
                    idx += 1
                    
            if scene_info.train_cameras:
                for cam in scene_info.train_cameras:
                    json_cams.append(camera_to_JSON(idx, cam,"train"))
                    idx += 1
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            for cam in self.train_cameras[resolution_scale]:
                cam.split = "train"
                
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            for cam in self.test_cameras[resolution_scale]:
                cam.split = "test"
            

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            if args.initialisation == "random":
                import numpy as np
                positions = np.random.uniform(-1,1,(10000,3))*5
                colors = np.random.uniform(0,1,(10000,3))
                normals = np.random.uniform(-1,1,(10000,3))
                pcd = BasicPointCloud(points=positions, colors=colors, normals=normals, visible_in_cameras=None)
                self.gaussians.create_from_pcd(pcd, self.cameras_extent)
            elif args.initialisation == "colmap":
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            elif args.initialisation == "depth":
                depth_pcd = fetchPly("test.ply")
                self.gaussians.create_from_pcd(depth_pcd, self.cameras_extent,downsample_factor=10)
            else:
                raise ValueError("Unknown initialisation method "+args.initialisation)
        
        self.point_cloud = scene_info.point_cloud

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]