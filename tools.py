import math
import os.path

from PIL import Image
import numpy as np
from os import listdir
from os.path import join
from original_tesellation import Tesellation, save_image
import torch
from pathlib import Path
import cv2


def equi_mask_to_equi_box(mask_image, ext='_box', overwrite=False, bounding_box=True):
    sphere_utils = Tesellation()
    mask = sphere_utils._read_image(mask_image)
    width = mask.shape[3]
    height = mask.shape[2]
    x1, x2, y1, y2 = mask_to_box(mask)
    theta, phi = box_to_theta_phi(x1, x2, y1, y2, width, height)
    tangents = sphere_utils.get_tangent_images(erp_image=mask, fov_x=120, fov_y=120,
                                               list_of_angles=[[theta, phi]])
    tangent = tangents[0]
    Path(os.path.dirname(mask_image)).mkdir(parents=True, exist_ok=True)
    x1, x2, y1, y2 = mask_to_box(tangent)
    tangent = torch.zeros(tangent.shape)
    tangent[:, :, y1:y2, x1:x2] = 1
    tangents[0] = tangent
    if bounding_box:
        bb = tangent_mask_to_bounding_box(tangent, (255, 0, 0, 255))
    erp_image_est = sphere_utils.remap_tangents(
        tangents=tangents, list_of_angles=[[theta, phi]], erp_h=height, erp_w=width)
    erp_image_est[:, 3, :, :] = 1
    if overwrite:
        sphere_utils._save_image(erp_image_est, mask_image)
    else:
        new_file = os.path.basename(mask_image).split('/')[-1]
        new_file = new_file[:len(new_file)-4] + ext + '.png'
        new_file = os.path.join(os.path.dirname(mask_image), new_file)
        sphere_utils._save_image(erp_image_est, new_file)
    if bounding_box:
        tangents[0] = bb
        erp_image_est = sphere_utils.remap_tangents(
            tangents=tangents, list_of_angles=[[theta, phi]], erp_h=height, erp_w=width)
        new_file = os.path.basename(mask_image).split('/')[-1]
        new_file = new_file[:len(new_file) - 4] + ext + '_bb.png'
        new_file = os.path.join(os.path.dirname(mask_image), new_file)
        sphere_utils._save_image(erp_image_est, new_file)


def mask_to_bounding_box(mask, save_filename):
    x1, x2, y1, y2 = mask_to_box(torch.from_numpy(mask))
    bb = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    bb = cv2.line(bb, (x1, y1), (x1, y2), (255, 0, 0, 255), 2)
    bb = cv2.line(bb, (x1, y1), (x2, y1), (255, 0, 0, 255), 2)
    bb = cv2.line(bb, (x2, y1), (x2, y2), (255, 0, 0, 255), 2)
    bb = cv2.line(bb, (x1, y2), (x2, y2), (255, 0, 0, 255), 2)
    im = Image.fromarray(bb)
    im.save(save_filename)


def tangent_mask_to_bounding_box_np(mask, color):
    x1, x2, y1, y2 = mask_to_box(mask)
    bb = np.zeros((mask.shape[2], mask.shape[3], 4), dtype=np.uint8)
    bb = cv2.line(bb, (x1, y1), (x1, y2), color, 3)
    bb = cv2.line(bb, (x1, y1), (x2, y1), color, 3)
    bb = cv2.line(bb, (x2, y1), (x2, y2), color, 3)
    bb = cv2.line(bb, (x1, y2), (x2, y2), color, 3)
    return bb


def tangent_mask_to_bounding_box(mask, color):
    bb = tangent_mask_to_bounding_box_np(mask, color)
    bb = torch.from_numpy(bb)
    result = torch.zeros(1, 4, mask.shape[2], mask.shape[3])
    result[0] = bb.permute(2, 0, 1)
    return result


def mask_to_box(mask):
    if len(mask.shape) == 4:
        idx = torch.argwhere(mask[:, 0:3, :, :])
        idx = idx[:, 2:4]
        if idx.shape[0] == 0:
            return 0, 0, 0, 0
        x1 = torch.min(idx[:, 1])
        x2 = torch.max(idx[:, 1])
        y1 = torch.min(idx[:, 0])
        y2 = torch.max(idx[:, 0])
    else:
        idx = torch.argwhere(mask)
        if idx.shape[0] == 0:
            return 0, 0, 0, 0
        x1 = torch.min(idx[:, 1])
        x2 = torch.max(idx[:, 1])
        y1 = torch.min(idx[:, 0])
        y2 = torch.max(idx[:, 0])

    return x1.item(), x2.item(), y1.item(), y2.item()


def box_to_theta_phi(x1, x2, y1, y2, width, height):
    x = torch.tensor((x1 + x2) / 2)
    y = torch.tensor((y1 + y2) / 2)
    theta = torch.rad2deg((-2 * math.pi / (width - 1)) * x + 2 * math.pi)
    phi = torch.rad2deg((math.pi / (height - 1)) * y)
    return theta.item(), phi.item()


def union_area(a,b):
    a_area = (a[2]-a[0]) * (a[3]-a[1])
    b_area = (b[2]-b[0]) * (b[3]-b[1])
    return a_area + b_area - intersection_area(a,b)

def intersection_area(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[2], b[2]) - x
    h = min(a[3], b[3]) - y
    if w<0 or h<0: return 0
    return w * h
