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

import torch
from depth_images import calibrate_depth
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from textured_render import prerender_depth, textured_render_multicam, textured_render_per_gaussian
import torchvision
from utils.general_utils import farthest_point_down_sample, safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import cv2
import numpy as np

from utils.image_utils import psnr
from utils.loss_utils import gaussian

def render_set(model_path, name, iteration, views,texture_views,gaussians, pipeline, background, blend_mode, render_type):
    approach = f"{render_type}_{blend_mode}_{iteration}_{len(texture_views)}"
    render_path = os.path.join(model_path, name, approach, "renders")
    gts_path = os.path.join(model_path, name, approach, "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    print(name)
    
    with open(os.path.join(model_path, name, approach, "num_gaussians"),"w") as f:
        f.write(str(len(gaussians.get_xyz)))

    psnrs = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        #import dill
        print(idx,view.image_name)
        # view = dill.load(open("tmp/viewpoint_cam","rb"))
        # gaussians2 = dill.load(open("tmp/gaussians","rb"))
        # pipeline = dill.load(open("tmp/pipe","rb"))
        # background = dill.load(open("tmp/bg","rb"))
        
        #rendering_pkg = render(view, gaussians, pipeline, background)
        #texture_views = [views[i] for i in [15,21,26,37,42,43,40,69,53,58,72,74,82,98]]
        #texture_views = views[1:]

        if render_type == "texture":
            rendering_pkg = textured_render_multicam(view, texture_views,gaussians, pipeline, background,in_training=(name=="train"),blend_mode=blend_mode)
            
            if args.inpaint:
                render_textured = cv2.inpaint(
                    (rendering_pkg["render_textured"].clamp(0,1).cpu().numpy().transpose((1,2,0))*255).astype(np.uint8),
                    ((1-rendering_pkg["render_textured_mask"]).cpu().numpy()).transpose((1,2,0)).astype(np.uint8),
                    10,
                    cv2.INPAINT_NS
                )
                render_textured = torch.tensor(render_textured).permute((2,0,1)).float().cuda()/255
            else:
                render_textured = rendering_pkg["render_textured"]
            
            render_textured_mask = rendering_pkg["render_textured_mask"]
            #psnrs.append(psnr(view.original_image*render_textured_mask, render_textured*render_textured_mask).mean().item())
            psnrs.append(psnr(view.original_image, render_textured).mean().item())
            #print(idx,psnrs[-1], render_textured_mask.mean())
            if render_textured_mask.mean()<0.995:
                print(render_textured_mask.mean(), view.image_name)
            #render_textured = rendering_pkg["render_textured"]
            torchvision.utils.save_image(render_textured, os.path.join(render_path, '{0:05d}'.format(idx) + "_texture.png"))
            torchvision.utils.save_image(render_textured_mask, os.path.join(render_path, '{0:05d}'.format(idx) + "_mask.png"))
            torchvision.utils.save_image(rendering_pkg["before_blend"], os.path.join(render_path, '{0:05d}'.format(idx) + "_before_blend.png"))
            torchvision.utils.save_image(rendering_pkg["render_textured_in_frame"], os.path.join(render_path, '{0:05d}'.format(idx) + "_in_frame.png"))
            torchvision.utils.save_image(rendering_pkg["render_opacity"], os.path.join(render_path, '{0:05d}'.format(idx) + "_opacity.png"))
        elif render_type == "texture_per_gaussian":
            rendering_pkg = textured_render_per_gaussian(view, texture_views,gaussians, pipeline, background,in_training=(name=="train"))
            torchvision.utils.save_image(rendering_pkg["render_textured"], os.path.join(render_path, '{0:05d}'.format(idx) + "_texture.png"))
            torchvision.utils.save_image(0.1*render(view, gaussians, pipeline, background,render_depth=True,depth_exp=1.0)["render_depth"], os.path.join(render_path, '{0:05d}'.format(idx) + "_depth1.0.png"))
            torchvision.utils.save_image(rendering_pkg["render_opacity"], os.path.join(render_path, '{0:05d}'.format(idx) + "_opacity.png"))
            psnrs.append(psnr(view.original_image, rendering_pkg["render_textured"]).mean().item())
        else:
            rendering_pkg = render(view, gaussians, pipeline, background,render_depth=True)
            #torchvision.utils.save_image(0.1*render(view, gaussians, pipeline, background,render_depth=True,depth_exp=0.25)["render_depth"]**4, os.path.join(render_path, '{0:05d}'.format(idx) + "_depth0.5.png"))
            torchvision.utils.save_image(0.1*render(view, gaussians, pipeline, background,render_depth=True,depth_exp=1.0)["render_depth"], os.path.join(render_path, '{0:05d}'.format(idx) + "_depth1.0.png"))
            #torchvision.utils.save_image(render(view, gaussians, pipeline, background,render_depth=True,depth_exp=4.0)["render_depth"]**0.25-rendering_pkg["render_depth"], os.path.join(render_path, '{0:05d}'.format(idx) + "_depth2.png"),normalize=True)
            

        rendering = rendering_pkg["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(render_path, '{0:05d}'.format(idx) + "_gt.png"))
        
        torchvision.utils.save_image(rendering_pkg["render_depth"]*0.1, os.path.join(render_path, '{0:05d}'.format(idx) + "_depth.png"))
        if view.depth is not None:
            torchvision.utils.save_image(view.depth*0.1, os.path.join(render_path, '{0:05d}'.format(idx) + "gtdepth.png"))
    print("Mean PSNR:",np.mean(psnrs))
    
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,blend_mode, render_type, train_images):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        calibrate_depth(scene)

        print("Num gaussians:", len(gaussians.get_xyz))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        prerender_depth(scene.getTrainCameras(), gaussians, pipeline, background)
        
        render_images = scene.getTrainCameras()
        
        render_images_subset = farthest_point_down_sample(torch.stack([c.camera_center.cpu() for c in render_images]), train_images)
        render_images = [render_images[i] for i in render_images_subset]
        
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(),render_images, gaussians, pipeline, background,blend_mode, render_type)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(),render_images, gaussians, pipeline, background, blend_mode,render_type)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--blend_mode", default="alpha", type=str)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default="normal",type=str)
    parser.add_argument("--inpaint", action="store_true")
    parser.add_argument("--train_images", default=1000, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    render_type = args.mode
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.blend_mode, render_type, args.train_images)