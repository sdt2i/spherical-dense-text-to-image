import numpy as np
import torch
import random
import sys
import os
from os.path import join
from argparse import Namespace, ArgumentParser
import imageio
import numpy as np
import torch
from pathlib import Path
import math
import region_based as md
from original_tesellation import Tesellation, save_image
import tools

MASKS_PATH = 'masks'
TANGENT_MASKS_PATH = 'tangent_masks'
REFERENCE_PATH = 'references'
size_vals = [0, 1, 2]
mask_idx_vals = [[0], [1], [2]]
background_prompts = ['an indoor room', 'a green field', 'a busy street']
foreground_prompts = [[['a table', 'a bed'], ['a television', 'a potted plant'], ['a wardrobe', 'a door']],
                      [['a cow', 'a sheep'], ['a cat', 'a pond'], ['a tree', 'a windmill']],
                      [['a car', 'a bus'], ['a sign', 'a bicycle'], ['a building', 'a traffic light']]]
masks = [['1_small.png', '2_small.png', '3_small.png'],
         ['1_medium.png', '2_medium.png', '3_medium.png'],
         ['1_large.png', '2_large.png', '3_large.png']]
masks_projected = [['1_small_box.png', '2_small_box.png', '3_small_box.png'],
                   ['1_medium_box.png', '2_medium_box.png', '3_medium_box.png'],
                   ['1_large_box.png', '2_large_box.png', '3_large_box.png']]
sphere_utils = Tesellation()

def generate_reference_images(path, masks, prompt, fg_prompts, seed):
    if not os.path.exists(join(path, f'{seed}.png')):
        args = get_args()
        args.bg_prompt = prompt
        args.fg_prompts = fg_prompts
        args.mask_paths = masks
        args.seed = seed
        args.outdir = path
        md.main(args)

def get_args():
    args = Namespace(
        mask_paths='',
        bg_prompt='',
        bg_negative='artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image',
        fg_prompts='',
        fg_negative='artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image',
        sd_version='2.0',
        H=512,
        W=512,
        seed=0,
        steps=50,
        bootstrapping=20,
    )
    return args

def project_masks(masks, projected_masks, mask_idx):
    for i in range(len(masks)):
        if not os.path.exists(projected_masks[i]):
            mask = torch.from_numpy(imageio.imread(masks[i], pilmode='RGBA')).float()
            mask = mask.permute(2, 0, 1).unsqueeze(0) / 255.0
            width = mask.shape[3]
            height = mask.shape[2]
            x1, x2, y1, y2 = tools.mask_to_box(mask)
            theta, phi = tools.box_to_theta_phi(x1, x2, y1, y2, width, height)
            if mask_idx < 2:
                phi = phi - 20
            tangents = sphere_utils.get_tangent_images(erp_image=mask, fov_x=120, fov_y=120,
                                                       list_of_angles=[[theta, phi]])
            tangent = tangents[0].cpu()
            save_image(tangent, projected_masks[i])

def main(seed_start, seed_end):
    print(f'generating seeds {seed_start} - {seed_end}')
    for size in size_vals:
        for mask_idx in mask_idx_vals:
            for prompt_idx in range(len(background_prompts)):
                for fg_prompt_idx in range(2):
                    for seed in range(seed_start, seed_end):
                        bg_prompt = background_prompts[prompt_idx]
                        fg_prompts = [foreground_prompts[prompt_idx][idx][fg_prompt_idx] for idx in mask_idx]
                        m = [join(MASKS_PATH, masks_projected[size][idx]) for idx in mask_idx]
                        mt = [join(TANGENT_MASKS_PATH, masks_projected[size][idx]) for idx in mask_idx]
                        project_masks(m, mt, mask_idx[0])
                        mask_str = ''
                        for idx in mask_idx:
                            mask_str = mask_str + str(idx)
                        path = join(REFERENCE_PATH, str(size), mask_str)
                        path = join(path, str(prompt_idx), str(fg_prompt_idx))
                        Path(path).mkdir(parents=True, exist_ok=True)
                        generate_reference_images(path, mt, bg_prompt, fg_prompts, seed)
        mask_1_path = join(REFERENCE_PATH, str(size), '0')
        mask_2_path = join(REFERENCE_PATH, str(size), '1')
        mask_3_path = join(REFERENCE_PATH, str(size), '2')
        mask_12_path = join(REFERENCE_PATH, str(size), '01')
        if os.path.exists(mask_12_path):
            continue
        Path(mask_12_path).mkdir(parents=True, exist_ok=True)
        mask_13_path = join(REFERENCE_PATH, str(size), '02')
        Path(mask_13_path).mkdir(parents=True, exist_ok=True)
        mask_23_path = join(REFERENCE_PATH, str(size), '12')
        Path(mask_23_path).mkdir(parents=True, exist_ok=True)
        mask_123_path = join(REFERENCE_PATH, str(size), '012')
        Path(mask_123_path).mkdir(parents=True, exist_ok=True)

        os.symlink(mask_1_path, join(mask_12_path, '0'))
        os.symlink(mask_2_path, join(mask_12_path, '1'))
        os.symlink(mask_1_path, join(mask_13_path, '0'))
        os.symlink(mask_3_path, join(mask_13_path, '2'))
        os.symlink(mask_2_path, join(mask_23_path, '1'))
        os.symlink(mask_3_path, join(mask_23_path, '2'))
        os.symlink(mask_1_path, join(mask_123_path, '0'))
        os.symlink(mask_2_path, join(mask_123_path, '1'))
        os.symlink(mask_3_path, join(mask_123_path, '2'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--seed_start', type=int, default=0)
    parser.add_argument('--seed_end', type=int, default=168)
    args = parser.parse_args()
    main(args.seed_start, args.seed_end)