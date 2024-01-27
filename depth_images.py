from utils.graphics_utils import geom_transform_points
import torch
import numpy as np

def calibrate_depth(scene):
    all_colmap_depths = []
    all_image_depths = []
    for train_camera in scene.getTrainCameras():
        points = torch.tensor(scene.point_cloud.points,device=train_camera.camera_center.device)
        valid_points = torch.tensor([train_camera.colmap_id in s for s in scene.point_cloud.visible_in_cameras],device=points.device)
        points = points[valid_points]

        depths = torch.sum((points - train_camera.camera_center)**2,dim=1)**0.5
        trans = geom_transform_points(points,train_camera.full_proj_transform)
        trans[:]+=1
        trans[:,0] *= train_camera.image_width/2
        trans[:,1] *= train_camera.image_height/2

        sample_x = trans[:,0].round().long()
        sample_y = trans[:,1].round().long()

        val = (sample_x>=0) & (sample_x<train_camera.image_width) & (sample_y>=0) & (sample_y<train_camera.image_height)

        image_depths = torch.zeros_like(depths)
        image_depths[val] = train_camera.depth.cuda()[0,sample_y[val],sample_x[val]]

        all_colmap_depths.append(depths[image_depths>0])
        all_image_depths.append(image_depths[image_depths>0])
        
    all_image_depths = torch.cat(all_image_depths).cpu()
    all_colmap_depths = torch.cat(all_colmap_depths).cpu()
    alpha,beta = np.polyfit(all_image_depths,all_colmap_depths, 1)
    print(alpha,beta)
    
    for train_camera in scene.getTrainCameras():
        train_camera.depth[train_camera.depth!=0] = train_camera.depth[train_camera.depth!=0]*alpha+beta
    
    for test_camera in scene.getTestCameras():
        test_camera.depth[test_camera.depth!=0] = test_camera.depth[test_camera.depth!=0]*alpha+beta
    