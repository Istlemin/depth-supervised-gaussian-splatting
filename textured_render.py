
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import geom_transform_points
from gaussian_renderer import render
import torchvision
from depth_images import camera_frustrum_points, depth_image_to_point_cloud
from tqdm import tqdm

def textured_render(render_points,viewpoint_camera, texture_camera, texture_scale, shadowmap_tol=0.05):

    #texture_coords = geom_transform_points(render_points, texture_camera.full_proj_transform)
    points_texture_camera = geom_transform_points(render_points, texture_camera.world_view_transform)
    texture_camera_depth = points_texture_camera[:,2]#.reshape((1,viewpoint_camera.image_height,viewpoint_camera.image_width))
    texture_coords = (texture_camera.proj_mat @ points_texture_camera.T).T
    texture_coords = (texture_coords/ texture_coords[:,2].reshape((-1,1)))[:,:2]
    
    texture_coords[:,0] = (texture_coords[:,0]+0.5-texture_camera.image_width/2)/(texture_camera.image_width/2)
    texture_coords[:,1] = (texture_coords[:,1]+0.5-texture_camera.image_height/2)/(texture_camera.image_height/2)

    _,tex_h, tex_w = texture_camera.original_image.shape

    texture_camera_image = (texture_camera.image_scales[texture_scale])
    # texture_camera_image = torch.sigmoid(texture_camera.learnable_image)
    # texture_camera_image = torch.nn.functional.interpolate(texture_camera_image.unsqueeze(0), texture_camera.image_scales[texture_scale].shape[1:], mode='area').squeeze()
    texture_color = torch.nn.functional.grid_sample(texture_camera_image.unsqueeze(0), texture_coords.reshape((1,1,-1,2)),align_corners=False,padding_mode="border",mode='bicubic')
    #texture_color = texture_color.reshape((3,viewpoint_camera.image_height,viewpoint_camera.image_width))

    texture_camera_target_depth = torch.nn.functional.grid_sample(texture_camera.rendered_depth_scales[texture_scale].unsqueeze(0), texture_coords.reshape((1,1,-1,2)),align_corners=False,mode='bicubic')
    #texture_camera_target_depth = texture_camera_target_depth.reshape((1,viewpoint_camera.image_height,viewpoint_camera.image_width))

    # not_in_shadow = (abs(texture_camera_depth - texture_camera_target_depth)<0.05).float()
    
    shadowmap_diff = texture_camera_depth - texture_camera_target_depth
    not_in_shadow = torch.exp(-shadowmap_diff**2/shadowmap_tol**2)

    #texture_coords = texture_coords.reshape((1,viewpoint_camera.image_height,viewpoint_camera.image_width,2))
    #not_in_shadow = not_in_shadow & (texture_camera_depth>0.2) & (texture_coords[:,:,:,0]<1) & (texture_coords[:,:,:,0]>-1) & (texture_coords[:,:,:,1]>-1) & (texture_coords[:,:,:,1]<1)

    tex_vec = texture_camera.camera_center - render_points
    view_vec = viewpoint_camera.camera_center - render_points

    eps = 1e-4
    pixel_camera_score = ((tex_vec / (tex_vec.norm(dim=1).unsqueeze(1)+eps)) * (view_vec / (view_vec.norm(dim=1).unsqueeze(1)+eps))).sum(dim=1)**2 / (tex_vec.norm(dim=1)+eps)
    #pixel_camera_score = pixel_camera_score.reshape((1,viewpoint_camera.image_height,viewpoint_camera.image_width))
    # print((texture_color*not_in_shadow).sum().item())
    return texture_color, not_in_shadow, pixel_camera_score

import torch
import math

def gaussian_kernel_1d(sigma: float, num_sigmas: float = 3.) -> torch.Tensor:
    
    radius = math.ceil(num_sigmas * sigma)
    support = torch.arange(-radius, radius + 1, dtype=torch.float)
    kernel = torch.distributions.Normal(loc=0, scale=sigma).log_prob(support).exp_()
    # Ensure kernel weights sum to 1, so that image brightness is not altered
    return kernel.mul_(1 / kernel.sum())


kernel_radius = 3
channels = 3
vertical_kernel = torch.zeros((channels,channels,1,kernel_radius*2+1)).float()
for i in range(channels):
    vertical_kernel[i,i,0,:] = gaussian_kernel_1d(1,kernel_radius)
vertical_kernel = vertical_kernel.cuda()
horizontal_kernel = torch.zeros((channels,channels,kernel_radius*2+1,1)).float()
for i in range(channels):
    horizontal_kernel[i,i,:,0] = gaussian_kernel_1d(1,kernel_radius)
horizontal_kernel = horizontal_kernel.cuda()

def blur(img):
    channels = img.shape[-3]
    img2 = torch.nn.functional.conv2d(img.unsqueeze(0),vertical_kernel[:channels,:channels],padding='same')
    img3 = torch.nn.functional.conv2d(img2,horizontal_kernel[:channels,:channels],padding='same')
    return img3[0]

def get_top_texture_cameras(viewpoint_camera, render_args, texture_cameras, num_texture_views,in_training):
    visible_texture_cameras = texture_cameras[:]
    # visible_texture_cameras = [viewpoint_camera]
    
    visible_texture_cameras.sort(key=(lambda cam: torch.norm(cam.camera_center-viewpoint_camera.camera_center)))
    visible_texture_cameras = visible_texture_cameras[int(in_training):(num_texture_views)*2+int(in_training):2]
    
    for camera in (visible_texture_cameras):
        if not hasattr(camera,"rendered_depth"):
            camera.rendered_depth = render(camera, *render_args)["render_depth"]
        camera.rendered_depth_scales = [camera.rendered_depth]
        for _ in range(5):
            #print(camera.rendered_depth.shape)
            camera.rendered_depth_scales.append(torch.nn.functional.interpolate(camera.rendered_depth_scales[-1].unsqueeze(0),scale_factor=0.5,mode="area")[0])
            
        camera.proj_mat = camera.get_proj_mat().cuda()
    
    return visible_texture_cameras

def textured_render_multicam(viewpoint_camera, texture_cameras, pc : GaussianModel, pipe, bg_color : torch.Tensor,in_training=False, blur_inpaint=False, texture_scale=0, blend_mode=None,num_texture_views=32):
    render_pkg_view = render(viewpoint_camera, pc, pipe, bg_color)

    render_textured = torch.zeros_like(viewpoint_camera.original_image)
    render_textured_mask  = torch.zeros((render_textured.shape[0],render_textured.shape[1],1)).cuda().bool()

    render_points = depth_image_to_point_cloud(render_pkg_view["render_depth"], viewpoint_camera)
    # render_points = depth_image_to_point_cloud(viewpoint_camera.depth.cuda()-4e-2, viewpoint_camera)

    # viewpoint_frustrum_points = camera_frustrum_points(viewpoint_camera)

    # with torch.no_grad():
    #     visible_texture_cameras = []
    #     for cam in texture_cameras:
    #         cam_points = geom_transform_points(viewpoint_frustrum_points, cam.world_view_transform)
    #         cam_points = (cam.proj_mat @ cam_points.T).T

    #         if ((cam_points[:,0]>=0) & (cam_points[:,1]>=0) & (cam_points[:,0]<cam.image_height) & (cam_points[:,1]<cam.image_width)).sum()>10:
    #             visible_texture_cameras.append(cam)

            
    #     #print("Vis cams:",len(visible_texture_cameras))
    
    visible_texture_cameras = get_top_texture_cameras(viewpoint_camera,(pc,pipe,bg_color),texture_cameras,num_texture_views,in_training)
    
    texture_colors = []
    texture_masks = []
    texture_scores = []

    for i,texture_camera in enumerate(visible_texture_cameras):
        #print(texture_camera.colmap_id, exclude_texture_idx)
        curr_texture_colors, curr_texture_mask,pixel_camera_score = textured_render(render_points,viewpoint_camera, texture_camera, texture_scale=texture_scale)

        texture_colors.append(curr_texture_colors.reshape(3,viewpoint_camera.image_height,viewpoint_camera.image_width))
        texture_masks.append(curr_texture_mask.reshape(1,viewpoint_camera.image_height,viewpoint_camera.image_width))
        texture_scores.append(pixel_camera_score.reshape(1,viewpoint_camera.image_height,viewpoint_camera.image_width))

    texture_colors = torch.stack(texture_colors)
    texture_masks = torch.stack(texture_masks)
    texture_scores = torch.stack(texture_scores)

    if blend_mode == "alpha":
        w = torch.zeros_like(texture_masks)
        T = torch.ones_like(texture_masks[0])
        render_textured_mask = 1 - torch.prod(1-texture_masks,dim=0)
        
        for i in range(len(visible_texture_cameras)):
            w[i] = texture_masks[i] * T
            T = T*(1-texture_masks[i])
            
        render_textured = (texture_colors*w).sum(dim=0) + (1-render_textured_mask)*texture_colors[0]
        
    else:
        eps = 1e-10
        #texture_scores = torch.exp((texture_scores-0.5*(1-texture_masks)+0.5)*100)
        
        texture_scores = texture_scores-1*(1-texture_masks)
        if blend_mode=="scores_softmax":
            texture_scores = texture_scores-texture_scores.min()
            texture_scores = torch.exp(texture_scores*50)
        else:
            texture_scores = (texture_scores==(texture_scores.amax(dim=0).unsqueeze(0)))
            
        render_textured = (texture_colors * texture_scores).sum(dim=0) / (texture_scores.sum(dim=0)+eps)
        #texture_scores.sum(dim=0)<eps
        #render_textured = render_textured * (texture_scores.sum(dim=0)>eps) + (texture_scores.sum(dim=0)<eps)
        
        render_textured_mask = torch.minimum(torch.sum(texture_masks,dim=0),torch.ones_like(torch.sum(texture_masks,dim=0)))
        
        #     render_textured += curr_texture_colors * (~render_textured_mask)
        #     render_textured_mask = render_textured_mask | curr_texture_mask

        # render_textured = render_textured/torch.clip(render_textured_mask,0.1,10000)


    #torch.nn.functional.conv2d(render_textured, )
    
    opacity = render_pkg_view["render_opacity"]
    render_textured[:,(opacity[0]<0.15)] *= opacity[(opacity<0.15)]
    # render_pkg_view["render_opacity"]
    # render_textured *= (render_pkg_view["render_opacity"]>0.1)
    # render_textured_mask[render_pkg_view["render_opacity"]<=0.1] = 1

    if blur_inpaint:
        eps = 1e-4
        extra_mask = blur(render_textured_mask.float())
        extra = blur(render_textured) / torch.clip(extra_mask,eps,10000)
        
        render_textured =  render_textured + (~render_textured_mask) * extra * (extra_mask>eps)
        render_textured_mask = render_textured_mask | (extra_mask>eps)
        #print((render_textured).sum().item())
    
    return {
        "render_textured": render_textured,
        "render_textured_mask": render_textured_mask,
        "texture_colors": texture_colors,
        "texture_masks": texture_masks, 
        "texture_images": [cam.learnable_image for cam in visible_texture_cameras],
        **render_pkg_view
    }

def prerender_depth(cameras, pc, pipe, bg_color):
    with torch.no_grad():
        for camera in (cameras):
            if not hasattr(camera,"rendered_depth"):
                camera.rendered_depth = render(camera, pc, pipe, bg_color)["render_depth"]
            camera.rendered_depth_scales = [camera.rendered_depth]
            for _ in range(5):
                #print(camera.rendered_depth.shape)
                camera.rendered_depth_scales.append(torch.nn.functional.interpolate(camera.rendered_depth_scales[-1].unsqueeze(0),scale_factor=0.5,mode="area")[0])
                
            camera.proj_mat = camera.get_proj_mat().cuda()

def get_normal(scale, q):
    local_normal = (scale==torch.amin(scale,dim=1).reshape((-1,1))).float()
    
    r = q[:,0]
    x = q[:,1]
    y = q[:,2]
    z = q[:,3]

    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)
    ],dim=1).reshape((-1,3,3))

    normals = (torch.bmm(local_normal.reshape((-1,1,3)), R)).reshape((-1,3))
    return normals/normals.norm(dim=1).unsqueeze(1)

def get_3d_point(pc,view_camera):
    normals = get_normal(pc.get_scale,pc.get_rot)
    

def get_uv_function(pc, view_camera, texture_camera):
    normals = get_normal(pc.get_scale,pc.get_rot)
    
    

def textured_render_per_gaussian(viewpoint_camera, texture_cameras, pc : GaussianModel, pipe, bg_color : torch.Tensor,in_training=False, blur_inpaint=False, texture_scale=0, blend_mode=None,num_texture_views=32):
    visible_texture_cameras = get_top_texture_cameras(viewpoint_camera,(pc,pipe,bg_color),texture_cameras,num_texture_views,in_training)
    
    render_points = pc.get_xyz

    texture_colors = []
    texture_masks = []

    for i,texture_camera in enumerate(visible_texture_cameras):
        #print(texture_camera.colmap_id, exclude_texture_idx)
        curr_texture_colors, curr_texture_mask,_ = textured_render(render_points,viewpoint_camera, texture_camera, texture_scale=texture_scale)

        texture_colors.append(curr_texture_colors)
        texture_masks.append(curr_texture_mask)

    texture_colors = torch.stack(texture_colors)
    texture_masks = torch.stack(texture_masks)

    w = torch.zeros_like(texture_masks)
    T = torch.ones_like(texture_masks[0])
    render_textured_mask = 1 - torch.prod(1-texture_masks,dim=0)
    
    for i in range(len(visible_texture_cameras)):
        w[i] = texture_masks[i] * T
        T = T*(1-texture_masks[i])
        
    colors = (texture_colors*w).sum(dim=0) + (1-render_textured_mask)*texture_colors[0]
    
    return render(viewpoint_camera, pc, pipe, bg_color,render_depth=True,override_color=colors[0,:,0,:].T)