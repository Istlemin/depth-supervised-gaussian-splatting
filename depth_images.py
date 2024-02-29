from utils.graphics_utils import geom_transform_points, fov2focal
import torch
import numpy as np
from utils.graphics_utils import fov2focal


def calibrate_depth(scene):
    # all_colmap_depths = []
    # all_image_depths = []
    # for train_camera in scene.getTrainCameras():
    #     points = torch.tensor(scene.point_cloud.points,device=train_camera.camera_center.device)
    #     valid_points = torch.tensor([train_camera.colmap_id in s for s in scene.point_cloud.visible_in_cameras],device=points.device)
    #     points = points[valid_points]

    #     depths = torch.sum((points - train_camera.camera_center)**2,dim=1)**0.5
    #     trans = geom_transform_points(points,train_camera.full_proj_transform)
    #     trans[:]+=1
    #     trans[:,0] *= train_camera.image_width/2
    #     trans[:,1] *= train_camera.image_height/2

    #     sample_x = trans[:,0].round().long()
    #     sample_y = trans[:,1].round().long()

    #     val = (sample_x>=0) & (sample_x<train_camera.image_width) & (sample_y>=0) & (sample_y<train_camera.image_height)

    #     image_depths = torch.zeros_like(depths)
    #     image_depths[val] = train_camera.depth.cuda()[0,sample_y[val],sample_x[val]]

    #     all_colmap_depths.append(depths[image_depths>0])
    #     all_image_depths.append(image_depths[image_depths>0])
        
    # all_image_depths = torch.cat(all_image_depths).cpu()
    # all_colmap_depths = torch.cat(all_colmap_depths).cpu()
    # alpha,beta = np.polyfit(all_image_depths,all_colmap_depths, 1)
    # print(alpha,beta)
    
    beta = 0
    alpha = 1.0#2.5
    
    for train_camera in scene.getTrainCameras():
        train_camera.depth[train_camera.depth!=0] = train_camera.depth[train_camera.depth!=0]*alpha+beta
    
    for test_camera in scene.getTestCameras():
        test_camera.depth[test_camera.depth!=0] = test_camera.depth[test_camera.depth!=0]*alpha+beta

def depth_image_to_point_cloud(depth, camera):
    mat = camera.world_view_transform.T

    fx = fov2focal(camera.FoVx,camera.image_width)
    fy = fov2focal(camera.FoVy,camera.image_height)
    cx = camera.image_width/2
    cy = camera.image_height/2

    _,h,w = depth.shape

    device = depth.device
    pixel_coords = torch.zeros((h,w,2),device = device).float()

    pixel_coords[:,:,1], pixel_coords[:,:,0]  = torch.meshgrid(torch.arange(0,h,device=device),torch.arange(0,w,device=device),indexing="ij")
    
    flat_cam_coords = torch.zeros((h,w,3),device=device).float()
    flat_cam_coords[:,:,0] = (pixel_coords[:,:,0] - cx)/fx
    flat_cam_coords[:,:,1]  = (pixel_coords[:,:,1] - cy)/fy
    flat_cam_coords[:,:,2]  = 1

    #flat_origin_dist = flat_cam_coords.norm(dim=2).unsqueeze(0)
    point_cam_coords = flat_cam_coords.unsqueeze(0) * (depth / 1).unsqueeze(3)
    
    if mat is None:
        return point_cam_coords.reshape((-1,3))
    else:
        return geom_transform_points(point_cam_coords.reshape((-1,3)), torch.linalg.inv(mat).T)



def camera_to_pcd(camera):
    depth = camera.depth.cuda()
    points = depth_image_to_point_cloud(depth,camera)
    colors = camera.original_image.permute((1,2,0)).reshape((-1,3))

    return points, colors

def camera_frustrum_points(camera):
    mat = camera.world_view_transform.T

    fx = fov2focal(camera.FoVx,camera.image_width)
    fy = fov2focal(camera.FoVy,camera.image_height)
    cx = camera.image_width/2
    cy = camera.image_height/2

    _,h,w = camera.original_image.shape

    device = camera.original_image.device

    y, x  = torch.meshgrid(torch.tensor(list(range(0,h,100)) + [h-1],device=device),torch.tensor(list(range(0,w,100)) + [w-1],device=device),indexing="ij")
    
    flat_cam_coords = torch.zeros((y.shape[0],y.shape[1],3),device=device).float()
    flat_cam_coords[:,:,0] = (x - cx)/fx
    flat_cam_coords[:,:,1]  = (y - cy)/fy
    flat_cam_coords[:,:,2]  = 1

    point_cam_coords = torch.cat([
        flat_cam_coords*0.5,
        flat_cam_coords*1,
        flat_cam_coords*2,
    ]).reshape((-1,3))

    if mat is None:
        return point_cam_coords.reshape((-1,3))
    else:
        return geom_transform_points(point_cam_coords.reshape((-1,3)), torch.linalg.inv(mat).T)
    
def depth_smoothness_loss(depth_image, image,alpha=20):
    dx_depth = depth_image[:,1:,:]-depth_image[:,:-1,:]
    dy_depth = depth_image[:,1:,:]-depth_image[:,:-1,:]
    dx_image = image[:,1:,:]-image[:,:-1,:]
    dy_image = image[:,1:,:]-image[:,:-1,:]
    loss = (torch.abs(dx_depth)*torch.exp(-dx_image*alpha)).sum() / torch.exp(-dx_image*alpha).sum()
    loss = (torch.abs(dy_depth)*torch.exp(-dy_image*alpha)).sum() / torch.exp(-dy_image*alpha).sum()

    return loss*100