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
import torch
import imageio
import random
import torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import colormap
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import pylab as plt
import numpy as np
import cv2
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from os import makedirs

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from lpipsPyTorch import lpips
import json

def custom_update_param1(param, base_lr, scale, eps=1e-8):
    if param.grad is None:
        return

    with torch.no_grad():
        scaled_lr = base_lr * (scale + eps)
        param.sub_(param.grad * scaled_lr)  # .unsqueeze(-1))


def custom_update_param2(param, base_lr, scale, eps=1e-8):
    if param.grad is None:
        return

    with torch.no_grad():
        scaled_lr = base_lr * (scale + eps)
        param.sub_(param.grad * scaled_lr.unsqueeze(-1))


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    random_seed = 3407
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    sorted_ws = []
    losses = []
    train_psnr = []
    test_psnr = []
    for iteration in range(first_iter, opt.iterations + 1):
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
        #                                                                                                        0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # print("proj matrix", viewpoint_cam.full_proj_transform)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        # image, depth_map, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]

        # median_depth = render_pkg["median_depth"][0]
        # median_depth_weight = render_pkg["median_depth"][1]
        # opacity = render_pkg["opacity"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # depth loss for test
        # depth_distortion = (median_depth_weight * torch.abs(depth_map - opacity.detach() * median_depth)).mean()
        # loss += 0.2 * depth_distortion

        loss.backward()

        losses.append(loss.data.cpu())

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                    print("number of points after ADC:")
                    print(gaussians.get_xyz.shape[0])

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            w = gaussians.get_w  # .unsqueeze(1)
            xyz = gaussians.get_xyz
            x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            dist = torch.sqrt(x ** 2 + y ** 2 + z ** 2)  # / w
            dist = dist / w

            # print("check dist", dist.shape)
            # print("check w", w)

            if iteration % 1000 == 0:
                print("iteration", iteration)
                print("check w", w)
                # sorted_w_descending, indices_w1 = torch.sort(w[~torch.isnan(w)], descending=True)[:500]
                # sorted_w_ascending, indices_w2 = torch.sort(w[~torch.isnan(w)])[:500]
                # print("sorted w descending", sorted_w_descending)
                # print("sorted w ascending", sorted_w_ascending)
                sorted_w_500_d, indices_d = torch.sort(dist[~torch.isnan(dist)], descending=True)[:500]
                sorted_w_500, indices = torch.sort(dist[~torch.isnan(dist)])[:500]
                print("mean farthest 500 w", torch.mean(sorted_w_500_d[:500]).detach().cpu().numpy())
                print("farthest 50 w", sorted_w_500_d[:50].detach().cpu().numpy())
                print("xyz value of farthest points", xyz[indices_d].detach().cpu().numpy())
                print("w value of farthest points", w[indices_d].detach().cpu().numpy())
                # print("mean farthest 500 inv w", 1 / torch.mean(sorted_w_500[:500]).detach().cpu().numpy())
                # print("farthest 50 inv w", 1 / (sorted_w_500[:50]).detach().cpu().numpy())
                #
                # print("D mean farthest 500 w", torch.mean(sorted_w_500_d[:500]).detach().cpu().numpy())
                # print("D farthest 50 w", sorted_w_500_d[:50].detach().cpu().numpy())
                # print("D mean farthest 500 inv w", 1 / torch.mean(sorted_w_500_d[:500]).detach().cpu().numpy())
                # print("D farthest 50 inv w", 1 / (sorted_w_500_d[:50]).detach().cpu().numpy())
            sorted_ws.append(torch.mean(w).detach().cpu().numpy())

            record_psnr = False
            if record_psnr:
                trainCameras = scene.getTrainCameras().copy()
                testCameras = scene.getTestCameras().copy()

                train_psnr_itr = []
                for idx, view in enumerate(trainCameras):
                    render_train = render(view, gaussians, pipe, bg)["render"]
                    gt_train = view.original_image.cuda()
                    train_psnr_itr.append(psnr(render_train, gt_train).mean().double())
                # print(train_psnr_itr)
                train_psnr.append(torch.tensor(train_psnr_itr).mean().item())


                test_psnr_itr = []
                for idx, view in enumerate(testCameras):
                    render_test = render(view, gaussians, pipe, bg)["render"]
                    gt_test = view.original_image.cuda()
                    test_psnr_itr.append(psnr(render_test, gt_test).mean().double())
                test_psnr.append(torch.tensor(test_psnr_itr).mean().item())

                # print("psnr at iteartion: ", iteration)
                # print(train_psnr, test_psnr)
                # print(torch.tensor(train_psnr_itr).sum()/len(trainCameras), torch.tensor(test_psnr_itr).sum()/len(testCameras))

    sorted_ws_output_path = dataset.model_path + "/wsLoss.jpg"
    plt.figure()
    plt.plot(sorted_ws)
    plt.xlabel("iterations")
    plt.ylabel("mean w")
    plt.title("mean of 1000 smallest w")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(20, 10)

    plt.savefig(sorted_ws_output_path, dpi=150)
    print('Train render and save images...')

    np.save(dataset.model_path + "/train_psnr.npy", train_psnr)
    np.save(dataset.model_path + "/test_psnr.npy", test_psnr)

    cameras_train = scene.getTrainCameras().copy()

    directory_img = os.path.join(dataset.model_path, "result_images_train")
    train_loss_json = os.path.join(directory_img, "train_loss.json")
    train_loss_data = {}  # 创建一个空的字典来存储所有的结果
    os.makedirs(directory_img, exist_ok=True)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    # for camera in cameras_train:
    #     image_output_path = dataset.model_path + "/result_images_train/" + camera.image_name + "_render.jpg"
    #     depth_output_path = dataset.model_path + "/depth_maps_train/" + camera.image_name + "_depth.jpg"
    #     render_pkg = render(camera, gaussians, pipe, bg)
    #     result = render_pkg["render"]
    #     gt_image = camera.original_image.to(result.device)
    #     ssim_camera = ssim(result, gt_image)
    #     psnr_camera = psnr(result, gt_image).mean()
    #     lpips_camera = lpips(result, gt_image, net_type='vgg').mean()
    #
    #     # 每次循环将当前图像的名字作为 key，loss 值作为 value 加入到字典中
    #     train_loss_data[camera.image_name] = {
    #         "ssim": ssim_camera.item(),
    #         "psnr": psnr_camera.item(),
    #         "lpips": lpips_camera.item()
    #     }
    #
    #     result = result.detach().cpu().permute(1, 2, 0).numpy()
    #     imageio.imwrite(image_output_path, to8b(result))

    # 最后一次性将字典写入 JSON 文件
    with open(train_loss_json, 'w') as f:
        json.dump(train_loss_data, f, indent=4)

    print('Train render and save images...')
    cameras_test = scene.getTestCameras().copy()

    directory_img = os.path.join(dataset.model_path, "result_images_test")
    test_loss_json = os.path.join(directory_img, "test_loss.json")
    test_loss_data = {}  # 创建一个空的字典来存储所有的结果
    os.makedirs(directory_img, exist_ok=True)

    # scene_render = Scene(dataset, gaussians, shuffle=False)
    #
    # camera_test_render = scene_render.getTestCameras()

    # for idx, camera in enumerate(camera_test_render):
    #     # image_output_path = dataset.model_path + "/result_images_test/" + camera.image_name + "_render.jpg"
    #     # render_pkg = render(camera, gaussians, pipe, bg)
    #     # result = render_pkg["render"]
    #     #
    #     # gt_image = camera.original_image.to(result.device)
    #     # ssim_camera = ssim(result, gt_image)
    #     # psnr_camera = psnr(result, gt_image).mean()
    #     # lpips_camera = lpips(result, gt_image, net_type='vgg').mean()
    #     # # 每次循环将当前图像的名字作为 key，loss 值作为 value 加入到字典中
    #     # test_loss_data[camera.image_name] = {
    #     #     "ssim": ssim_camera.item(),
    #     #     "psnr": psnr_camera.item(),
    #     #     "lpips": lpips_camera.item()
    #     # }
    #     #
    #     # result = result.detach().cpu().permute(1, 2, 0).numpy()
    #     # imageio.imwrite(image_output_path, to8b(result))
    #     name = "test"
    #     render_path = os.path.join(dataset.model_path, name, "ours_{}".format(opt.iterations), "renders")
    #     gts_path = os.path.join(dataset.model_path, name, "ours_{}".format(opt.iterations), "gt")
    #
    #     makedirs(render_path, exist_ok=True)
    #     makedirs(gts_path, exist_ok=True)
    #
    #     rendering = render(camera, gaussians, pipe, background)["render"]
    #     gt = camera.original_image[0:3, :, :]
    #     torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    #     torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    # 最后一次性将字典写入 JSON 文件
    with open(test_loss_json, 'w') as f:
        json.dump(test_loss_data, f, indent=4)

    loss_output_path = dataset.model_path + "/lossPlot.jpg"
    plt.figure()
    plt.plot(losses)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.title("lossL1 + lossDssim")

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(20, 10)

    plt.savefig(loss_output_path, dpi=150)

    plt.close()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[7_000, 10000, 20000, 30_000, 50000, 70000, 100000])
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[7_000, 10000, 20000, 30_000, 50000, 70000, 100000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
