import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import time
import numpy as np

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    render_time_list = []
    for i in range(1):
        print(f"evaluating i: {i}")
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            # import pdb; pdb.set_trace()
            torch.cuda.synchronize()
            start = time.time()
            rendering = render(view, gaussians, pipeline, background)["render"]
            torch.cuda.synchronize()
            end = time.time()

            render_time_list.append((end-start)*1000)
            # gt = view.original_image[0:3, :, :]
            # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    print(render_time_list)
    print("????", render_time_list[5:])
    print("?????", len(views) * 100 / (np.sum(render_time_list) / 1000))


    with open(os.path.join(model_path, name, 'render_time.txt'), 'w') as f:
        for t in render_time_list:
            f.write("%.2fms\n"%t)
        f.write("Mean time: %.2fms\n"%(np.mean(render_time_list[5:])))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        # import pdb; pdb.set_trace()
        cameras = scene.getTestCameras()
        # for cam in cameras:
        #     cam.image_height = 1080
        #     cam.image_width = 1920

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, cameras, gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)