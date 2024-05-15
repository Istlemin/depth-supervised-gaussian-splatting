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

from pathlib import Path
import torch
from depth_images import calibrate_depth
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from textured_render import prerender_depth, textured_render_multicam, textured_render_per_gaussian
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import cv2
import numpy as np
from scene.cameras import MiniCam
import math
import glm
import copy

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, output:Path,blend_mode, render_type):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        print("Num gaussians:", len(gaussians.get_xyz))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        prerender_depth(scene.getTrainCameras(), gaussians, pipeline, background)

        view = copy.deepcopy(scene.getTrainCameras()[0])
        radius = np.linalg.norm(view.camera_center.cpu())
        for i in tqdm(range(200)):
            angle = i*0.01
            cam_pos = np.array([math.cos(angle),math.sin(angle),-0.5])
            up = np.array([0,0,-1]).astype(float)
            
            M = torch.tensor(np.array(glm.lookAt(cam_pos, np.zeros(3).astype(float), up)))
            #print(M)
            view.R = M[:3,:3].numpy().T
            view.T = np.array([0,0,radius])
            
            view.update_transforms()
            
            rendering_pkg = textured_render_multicam(
                view,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                blend_mode=blend_mode
            )
            
            #rendering_pkg = textured_render_per_gaussian(view, scene.getTrainCameras(),gaussians, pipeline, background,in_training=False)
            torchvision.utils.save_image(
                rendering_pkg["render_textured"],
                output/f"{i}.png"
            )
            

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--blend_mode", default="alpha", type=str)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--textured_render", action="store_true")
    parser.add_argument("--inpaint", action="store_true")
    parser.add_argument("--output", type=Path)
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    render_type = "normal"
    if args.textured_render:
        render_type = "texture"

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.output, args.blend_mode, render_type)