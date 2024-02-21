
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import geom_transform_points
from gaussian_renderer import render
import torchvision
from depth_images import depth_image_to_point_cloud
from tqdm import tqdm

def textured_render(render_points,viewpoint_camera, texture_camera):

    #texture_coords = geom_transform_points(render_points, texture_camera.full_proj_transform)
    points_texture_camera = geom_transform_points(render_points, texture_camera.world_view_transform)
    texture_camera_depth = points_texture_camera[:,2].reshape((1,viewpoint_camera.image_height,viewpoint_camera.image_width))
    texture_coords = (texture_camera.proj_mat @ points_texture_camera.T).T
    texture_coords = (texture_coords/ texture_coords[:,2].reshape((-1,1)))[:,:2]
    
    texture_coords[:,0] = (texture_coords[:,0]-texture_camera.image_width/2)/(texture_camera.image_width/2)
    texture_coords[:,1] = (texture_coords[:,1]-texture_camera.image_height/2)/(texture_camera.image_height/2)

    _,tex_h, tex_w = texture_camera.original_image.shape

    texture_color = torch.nn.functional.grid_sample(texture_camera.original_image.reshape((1,3,tex_h,tex_w)), texture_coords.reshape((1,1,-1,2)),align_corners=False)
    texture_color = texture_color.reshape((3,viewpoint_camera.image_height,viewpoint_camera.image_width))

    texture_camera_target_depth = torch.nn.functional.grid_sample(texture_camera.rendered_depth.reshape((1,1,tex_h,tex_w)), texture_coords.reshape((1,1,-1,2)),align_corners=False)
    texture_camera_target_depth = texture_camera_target_depth.reshape((1,viewpoint_camera.image_height,viewpoint_camera.image_width))

    not_in_shadow = torch.abs(texture_camera_depth - texture_camera_target_depth)<0.1

    texture_coords = texture_coords.reshape((1,viewpoint_camera.image_height,viewpoint_camera.image_width,2))
    not_in_shadow = not_in_shadow & (texture_camera_depth>0.2) & (texture_coords[:,:,:,0]<=1) & (texture_coords[:,:,:,0]>=-1) & (texture_coords[:,:,:,1]>=-1) & (texture_coords[:,:,:,1]<=1)

    tex_vec = texture_camera.camera_center - render_points
    view_vec = viewpoint_camera.camera_center - render_points
    pixel_camera_score = ((tex_vec / tex_vec.norm(dim=1).unsqueeze(1)) * (view_vec / view_vec.norm(dim=1).unsqueeze(1))).sum(dim=1) / tex_vec.norm(dim=1)
    pixel_camera_score = pixel_camera_score.reshape((1,viewpoint_camera.image_height,viewpoint_camera.image_width))
    return texture_color*not_in_shadow, not_in_shadow, pixel_camera_score

def textured_render_multicam(viewpoint_camera, texture_cameras, pc : GaussianModel, pipe, bg_color : torch.Tensor,exclude_texture_idx=-1):
    render_pkg_view = render(viewpoint_camera, pc, pipe, bg_color)

    render_textured = torch.zeros_like(viewpoint_camera.original_image)
    render_textured_mask  = torch.zeros_like(viewpoint_camera.depth).cuda().bool()

    render_points = depth_image_to_point_cloud(render_pkg_view["render_depth"], viewpoint_camera)

    texture_colors = []
    texture_masks = []
    texture_scores = []

    for i,texture_camera in enumerate(texture_cameras):
        if i == exclude_texture_idx:
            continue
        
        curr_texture_colors, curr_texture_mask,pixel_camera_score = textured_render(render_points,viewpoint_camera, texture_camera)

        texture_colors.append(curr_texture_colors)
        texture_masks.append(curr_texture_mask)
        texture_scores.append(pixel_camera_score)

    texture_colors = torch.stack(texture_colors)
    texture_masks = torch.stack(texture_masks)
    texture_scores = torch.stack(texture_scores)    
    texture_scores = torch.exp(texture_scores*10-(~texture_masks)*10000)
    render_textured = (texture_colors *texture_scores).sum(dim=0) / texture_scores.sum(dim=0)
    render_textured_mask = torch.sum(texture_masks,dim=0)>0
    #     render_textured += curr_texture_colors * (~render_textured_mask)
    #     render_textured_mask = render_textured_mask | curr_texture_mask

    # render_textured = render_textured/torch.clip(render_textured_mask,0.1,10000)

    return {
        "render_textured": render_textured,
        "render_textured_mask": render_textured_mask,
        **render_pkg_view
    }

def prerender_depth(cameras, pc, pipe, bg_color):
    for camera in cameras:
        camera.rendered_depth = render(camera, pc, pipe, bg_color)["render_depth"]
        camera.proj_mat = camera.get_proj_mat().cuda()