"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import sys
import os
sys.path.append('..')

import tqdm
import argparse
from utils import inception_utils
from dataloader.dataset import CelebAMaskDataset, WSIMaskDataset
import pickle

@torch.no_grad()
def extract_features(args, loader, inception, device):
    pbar = loader

    pools, logits = [], []

    for data in tqdm.tqdm(pbar):
        img = data['image']
        # check img dim
        if img.shape[1] != 3:
            img = img.expand(-1,3,-1,-1)

        img = img.to(device)
        pool_val, logits_val = inception(img)
        
        pools.append(pool_val.cpu().numpy())
        logits.append(F.softmax(logits_val, dim=1).cpu().numpy())

    pools = np.concatenate(pools, axis=0)
    logits = np.concatenate(logits, axis=0)

    return pools, logits


def get_dataset(args):
    if args.dataset_name == 'celeba-mask':
        unlabel_dataset = CelebAMaskDataset(args, args.path, is_label=False)
        train_val_dataset = CelebAMaskDataset(args, args.path, is_label=True, phase='train-val')
        dataset = ConcatDataset([unlabel_dataset, train_val_dataset])
    elif args.dataset_name == 'KID-MP':
        wsi_dir = '/home/cjr66/project/KID-DeepLearning/KID-Images-pyramid'
        coord_dir = '/home/cjr66/project/KID-DeepLearning/Patch_coords-1024/MP_KPMP_all-patches-stride256'
        mask_dir = '/home/cjr66/project/KID-DeepLearning/Labeled_patches/MP_1024_stride256'
        process_list = '/home/cjr66/project/KID-DeepLearning/proc_info/MP_only-KID_process_list.csv'
        class_val = {
             "Background": 0,
             "Lymphocytes": 19,
             "Neutrophils": 39,
             "Macrophage": 58,
             "PCT Nuclei": 78,
             "DCT Nuclei": 98,
             "Endothelial": 117,
             "Fibroblast": 137,
             "Mesangial": 156,
             "Parietal cells": 176,
             "Podocytes": 196,
             "Mitosis": 215,
             "Tubule Nuclei": 235
         }
        color_map = {
             0: [0, 0, 0],
             1: [0, 128, 0],
             2: [0, 255, 0],
             3: [255, 153,102],
             4: [255, 0, 255],
             5: [0, 0, 128],
             6: [0, 128, 128],
             7: [235, 206, 155],
             8: [255, 255, 0],
             9: [58, 208, 67],
             10: [0, 255, 255],
             11: [179, 26, 26],
             12: [130, 91, 37]
         }
        img_dataset = WSIMaskDataset(args,
                                     wsi_dir=wsi_dir,                   # Path to WSI directory.
                                     coord_dir=coord_dir,                 # Path to h5 coord database.
                                     mask_dir=mask_dir,
                                     class_val = class_val,
                                     color_map = color_map,
                                     process_list = process_list,       #Dataframe path of WSIs to process and their seg_levels/downsample levels that correspond to the coords
                                     wsi_exten = ['.tif', '.svs'],
                                     mask_exten = '.png',
                                     max_coord_per_wsi = 'inf',
                                     rescale_mpp = True,
                                     desired_mpp = 0.2,
                                     random_seed = 0,
                                     load_mode = 'openslide',
                                     make_all_pipelines = False,
                                     unlabel_transform=None, 
                                     latent_dir=None, 
                                     is_label=False, 
                                     phase='train', 
                                     aug=False, 
                                     resolution=1024
                                     )
        seg_dataset = WSIMaskDataset(args,
                                     wsi_dir=wsi_dir,                   # Path to WSI directory.
                                     coord_dir=coord_dir,                 # Path to h5 coord database.
                                     mask_dir=mask_dir,
                                     class_val = class_val,
                                     color_map = color_map,
                                     process_list = process_list,       #Dataframe path of WSIs to process and their seg_levels/downsample levels that correspond to the coords
                                     wsi_exten = ['.tif', '.svs'],
                                     mask_exten = '.png',
                                     max_coord_per_wsi = 'inf',
                                     rescale_mpp = True,
                                     desired_mpp = 0.2,
                                     random_seed = 0,
                                     load_mode = 'openslide',
                                     make_all_pipelines = False,
                                     unlabel_transform=None, 
                                     latent_dir=None, 
                                     is_label=True, 
                                     phase='train-val', 
                                     aug=False, 
                                     resolution=1024
                                     )
        dataset = ConcatDataset([img_dataset, seg_dataset])
    else:
        raise Exception('No such a dataloader!')
    return dataset

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(
        description='Calculate Inception v3 features for datasets'
    )
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--batch', default=64, type=int, help='batch size')
    parser.add_argument('--n_sample', type=int, default=50000)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--image_mode', type=str, default='RGB')
    parser.add_argument('--dataset_name', type=str, help='[celeba-mask, KID-MP]')
    parser.add_argument('--path', type=str, help='path to datset dir', required=False)

    args = parser.parse_args()

    inception = inception_utils.load_inception_net()

    dset = get_dataset(args)
    loader = DataLoader(dset, batch_size=args.batch, num_workers=7)

    pools, logits = extract_features(args, loader, inception, device)

    # pools = pools[: args.n_sample]
    # logits = logits[: args.n_sample]

    print(f'extracted {pools.shape[0]} features')

    print('Calculating inception metrics...')
    IS_mean, IS_std = inception_utils.calculate_inception_score(logits)
    print('Training data from dataloader has IS of %5.5f +/- %5.5f' % (IS_mean, IS_std))
    print('Calculating means and covariances...')

    mean = np.mean(pools, axis=0)
    cov = np.cov(pools, rowvar=False)

    with open(args.output, 'wb') as f:
        pickle.dump({'mean': mean, 'cov': cov, 'size': args.size, 'dataset': args.dataset_name}, f)
