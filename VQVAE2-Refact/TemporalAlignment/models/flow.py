# Code used to compute the flow of a sequence of frames 
import sys
sys.path.append('core')

import random
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft_original import RAFT
from utils import flow_viz
from utils.utils import InputPadder

from skimage import io
import os.path as osp

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

# method to compute the warp error between two videos 
def compute_error_videos(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    files = sorted(glob.glob('demo-frames/*.png'))
    # consider the second video in the reverse order

    video1 = files.copy()
    video2 = sorted(files, reverse=True) # dummy with the 2nd video reversed

    err = 0
    for i in range(1, len(files)-1):
        err += compute_error(model, video1[i-1], video1[i], video2[i-1], video2[i])

    total_error = err / (len(files)-1)
    print(f'The total flow error in the videos is : {total_error}')

def compute_error_from_tensors():


# method to compute the warp error between the two images
def compute_error(model, input_file1, input_file2, output_file1, output_file2):

    with torch.no_grad():
        input1 = load_image(input_file1)
        input2 = load_image(input_file2)

        padder = InputPadder(input1.shape)
        input1, input2 = padder.pad(input1, input2)

        output1 = load_image(output_file1)
        output2 = load_image(output_file2)

        output1, output2 = padder.pad(output1, output2)

        # compute the forward and backward flow using the input images
        _, fw_flow = model(input1, input2, iters=20, test_mode=True)
        _, bw_flow = model(input2, input1, iters=20, test_mode=True)


    # detect the occlusion using the forward backward flows 
    occlusion_mask = detect_occlusion(bw_flow, fw_flow)

    # print(f'fw flow dimensions - {fw_flow.shape}')
    # print(f'bw flow dimensions - {bw_flow.shape}')

    # print(f'occlusion : {occlusion_mask.shape}')

    output1 = output1.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    output2 = output2.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    fw_flow = fw_flow.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    warp_img_output2 = warp_flow(output2, fw_flow)

    # print(f'the dimensions of the warp image are : {warp_img_output2.shape}')
    
    noc_mask = 1 - occlusion_mask

    # compute the warping error 
    diff = np.multiply((warp_img_output2 - output1).transpose(2, 0, 1), noc_mask) # shape -> 3 x 440 x 1024

    # print(diff)

    N = np.sum(noc_mask)
    if N == 0:
        N = diff.shape[0] * diff.shape[1] * diff.shape[2]
    err = np.sum(np.square(diff)/N)

    return err

def compute_flow_magnitude(flow):

    flow_mag = flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2

    return flow_mag

# method to compute the occlusion
def detect_occlusion(fw_flow, bw_flow):
    # fw-flow -> img1 => img2 
    # bw-flow -> img2 => img1 
    # the flow is of the dimension -> b x d x h x w

    fw_flow, bw_flow = tensor2img(fw_flow), tensor2img(bw_flow)

    fw_flow_w = fw_flow.copy()

    # occlusion is the sum of the forward and backward flows
    fb_flow_sum = fw_flow_w + bw_flow
    fb_flow_mag = compute_flow_magnitude(fb_flow_sum)
    fw_flow_w_mag = compute_flow_magnitude(fw_flow_w)
    bw_flow_mag = compute_flow_magnitude(bw_flow)

    mask1 = fb_flow_mag > 0.01 * (fw_flow_w_mag + bw_flow_mag) + 0.5

    occlusion = np.zeros((fw_flow.shape[0], fw_flow.shape[1]))
    occlusion[mask1 == 1] = 1

    return occlusion

def tensor2img(img_t):

    img = img_t[0].detach().to("cpu").numpy()
    img = np.transpose(img, (1, 2, 0))

    return img

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', help="restore checkpoint")
#     parser.add_argument('--path', help="dataset for evaluation")
#     parser.add_argument('--small', action='store_true', help='use small model')
#     parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
#     parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
#     args = parser.parse_args()

#     # load_model(args)
#     compute_error_videos(args)