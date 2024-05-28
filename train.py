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
import time
import torch
from random import randint
from depth_images import calibrate_depth, depth_smoothness_loss
from textured_render import prerender_depth, textured_render_multicam
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import torchvision


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    init_gaussian_ply,
    loss_version,
):
    scales = 2
    # if loss_version == 0:
    #     loss_version = 6
    #     if loss_version 
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_ply=init_gaussian_ply)
    # calibrate_depth(scene)

    gaussians.training_setup(opt, [camera.learnable_image for camera in scene.getTrainCameras()])
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_loss_depth_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam, gaussians, pipe, background, scaling_modifer
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        
        # selected_cameras = []
        # for camera in scene.getTrainCameras():
        #     #print(camera.colmap_id)
        #     # if camera.colmap_id in [0,10]:
        #     #     selected_cameras.append(camera)
        #     if camera.colmap_id in [55]:
        #         viewpoint_cam = camera
        
        # viewpoint_cam = selected_cameras[iteration%len(selected_cameras)]
        selected_cameras = scene.getTrainCameras()
        # print(viewpoint_cam.colmap_id)
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if opt.textured_render:
            if iteration % 10 == 1:
                #print("Prerendering Depths for shadow mapping")
                start_time = time.time()
                prerender_depth(selected_cameras, gaussians, pipe, bg)
                #print("Took time ", time.time() - start_time)
            
            scale=(iteration+scales-1)%scales
            
            render_pkg = textured_render_multicam(
                viewpoint_cam,
                selected_cameras,
                gaussians,
                pipe,
                background,
                in_training=True,
                texture_scale=scale,#(3-(iteration//600)),
                blend_mode="scores_softmax2",#"scores_softmax"
                num_texture_views=8
            )
            
            if iteration==1:
                original_depth = render_pkg["render_depth"]

            viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            masks = render_pkg["texture_masks"]
            images = render_pkg["texture_colors"]
            
            if len(masks.shape)<3:
                masks = masks.unsqueeze(0)
                images = images.unsqueeze(0)
            
            #images.retain_grad()
            render_pkg["render_depth"].retain_grad()
            #render_pkg["render"].retain_grad()

            gt_image_rz = viewpoint_cam.image_scales[scale]
            #gt_image_rz = viewpoint_cam.image_scales[0]
            
            images = torch.nn.functional.interpolate(images, (gt_image_rz.shape[1],gt_image_rz.shape[2]), mode='area')
            masks = torch.nn.functional.interpolate(masks, (gt_image_rz.shape[1],gt_image_rz.shape[2]), mode='area')
            
            mask = render_pkg["render_textured_mask"]
            image = render_pkg["render_textured"]
            
            gt_image_rz = gt_image_rz.unsqueeze(0)
            gt_image = viewpoint_cam.image_scales[0]
            
            full_photo = l1_loss((image)[:,:,:], (gt_image)[:,:,:])
                
            if loss_version==0:
                full_photo_per_cam = l1_loss((images)[:,:,:], (gt_image_rz)[:,:,:])
                loss = full_photo_per_cam
                
            if loss_version==1:
                masked_photo_per_cam = l1_loss((masks.detach()*images)[:,:,:], (masks.detach()*gt_image_rz)[:,:,:])
                loss = masked_photo_per_cam
    
            if loss_version==2:
                mask_loss_per_cam = l2_loss(masks*torch.maximum(torch.zeros_like(images),(1-abs(images-gt_image_rz).detach()*10)),torch.ones_like(masks))
                loss = mask_loss_per_cam
                
            if loss_version==3:
                masked_photo = l1_loss((mask.detach()*image)[:,:,:], (mask.detach()*gt_image)[:,:,:])
                loss = masked_photo
            
            if loss_version==4:
                loss = full_photo
            
            if loss_version==5:
                mask_loss = l2_loss(mask*torch.maximum(torch.zeros_like(image),(1-abs(image-gt_image).detach()*10)),torch.ones_like(mask))
                loss = mask_loss
                
            if loss_version==6:
                mask_loss_per_cam = l2_loss(masks*torch.maximum(torch.zeros_like(images),(1-abs(images-gt_image_rz).detach()*10)),torch.ones_like(masks))
                masked_photo = l1_loss((mask.detach()*image)[:,:,:], (mask.detach()*gt_image)[:,:,:])
                loss = (mask_loss_per_cam + masked_photo)/2
                
            if loss_version==7:
                loss = full_photo*0
            
            loss *= 2
            Ll1 = full_photo
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
                (1.0 - ssim(image, gt_image))
            )

        if viewpoint_cam.depth is not None:
            gt_depth = viewpoint_cam.depth.cuda()
            Ll1_depth = l1_loss(render_pkg["render_depth"] * (gt_depth > 0), gt_depth)
            Ll1_depth += (render_pkg["render_opacity"] * (gt_depth==0)).mean()
            loss = loss * (1.0 - opt.lambda_depth) + opt.lambda_depth * Ll1_depth
            
        else:
            Ll1_depth = torch.tensor(0)
        # loss += depth_smoothness_loss(render_pkg["render_depth"], viewpoint_cam.original_image.cuda())
        # print(depth_smoothness_loss(render_pkg["render_depth"], viewpoint_cam.original_image.cuda()))
        # print(f"{Ll1.item():.5f} {Ll1_2.item():.5f} {gaussians.depth_scale.item():.5f}")
        loss.backward()

        if iteration % 50 == 1:
            if opt.textured_render:
                # torchvision.utils.save_image(render_pkg["texture_images"], "tmp/textured_imgs.png")
                # torchvision.utils.save_image(images, "tmp/imgs.png")
                # torchvision.utils.save_image(images*masks, "tmp/imgs_masked.png")
                torchvision.utils.save_image(render_pkg["render_textured"], "tmp/img.png")
                torchvision.utils.save_image(render_pkg["render_textured"]*render_pkg["render_textured_mask"], "tmp/img_masked.png")
                # torchvision.utils.save_image(masks.float(), "tmp/mask.png")
                # torchvision.utils.save_image((masks*images), "tmp/outside.png")
                # torchvision.utils.save_image(
                #     gt_image_rz.broadcast_to(images.shape)*masks, "tmp/gt_image_masked.png"
                # )
                # torchvision.utils.save_image(
                #     gt_image_rz.broadcast_to(images.shape), "tmp/gt_images.png"
                # )
                torchvision.utils.save_image(
                    gt_image_rz, "tmp/gt_image.png"
                )
                torchvision.utils.save_image(
                    render_pkg["render_depth"].grad*10000+0.5, "tmp/depth_grad.png"
                )
                print(l1_loss((render_pkg["render_textured"])[:,:,:], (gt_image)[:,:,:])*5)#,outside_loss)
            else:
                torchvision.utils.save_image(image, "tmp/img.png")
                torchvision.utils.save_image(
                    render_pkg["render_opacity"], "tmp/opacity.png"
                )
                torchvision.utils.save_image(
                    gt_image, "tmp/gt_image.png"
                )
                

            if 'gt_depth' in locals():
                torchvision.utils.save_image(
                    render_pkg["render_depth"]/gt_depth.max(), "tmp/depth.png"
                )
                torchvision.utils.save_image(
                    gt_depth/gt_depth.max(), "tmp/gt_depth.png"
                )
            else:
                torchvision.utils.save_image(
                    render_pkg["render_depth"]*0.1, "tmp/depth.png"
                )
                
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if ema_loss_for_log==0:
                ema_loss_for_log = Ll1.item()
                ema_loss_depth_for_log = Ll1_depth.item()
            
            ema_loss_for_log = 0.01 * Ll1.item() + 0.99 * ema_loss_for_log
            ema_loss_depth_for_log = (
                0.01 * Ll1_depth.item() + 0.99 * ema_loss_depth_for_log
            )
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        "Depth Loss": f"{ema_loss_depth_for_log:.{7}f}",
                        "Gaussians": f"{len(gaussians.get_xyz)}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
            )

            if (iteration in saving_iterations) or iteration == 1:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                # import dill

                # dill.dump(viewpoint_cam,open("tmp/viewpoint_cam","wb"))
                # dill.dump(gaussians,open("tmp/gaussians","wb"))
                # dill.dump(pipe,open("tmp/pipe","wb"))
                # dill.dump(bg,open("tmp/bg","wb"))
                # torchvision.utils.save_image(
                #     0.2 * render_pkg["render_depth"] * (gt_depth > 0),
                #     "render_depth.png",
                # )
                # torchvision.utils.save_image(0.2 * gt_depth, "gt_depth.png")

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                    and len(gaussians.get_xyz) < opt.max_gaussians
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if (
                    iteration % opt.opacity_reset_interval == 0
                    or (dataset.white_background and iteration == opt.densify_from_iter)
                ) and len(gaussians.get_xyz) > 10000:
                    print("Resetting Opacity!")
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            print(f"{config['name']}:")
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    curr_psnr = psnr(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    print(curr_psnr)

                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )
            tb_writer.add_scalar(
                "total_points", scene.gaussians.get_xyz.shape[0], iteration
            )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--init_gaussian_ply", type=str, default=None)
    parser.add_argument("--loss_version", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet,args.seed)

    test_iterations = list(range(0, 100000, 1000))
    
    if op.extract(args).textured_render:
        save_iterations = list(range(0, 100000, 100))
    else:
        save_iterations = list(range(0, 100000, 1000))
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        test_iterations,
        save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        init_gaussian_ply=args.init_gaussian_ply,
        loss_version=args.loss_version
    )

    # All done
    print("\nTraining complete.")
