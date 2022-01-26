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

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import torch
import cv2
import albumentations as A
import pandas as pd
import random
import h5py
import openslide
import pyvips
import math

class HistogramEqualization(object):
    def __call__(self, img):
        img_eq = ImageOps.equalize(img)
        
        return img_eq

class AdjustGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma
    
    def __call__(self, img):
        img_gamma = transforms.functional.adjust_gamma(img, self.gamma)

        return img_gamma

#----------------------------------------------------------------------------

class CelebAMaskDataset(Dataset):
    def __init__(self, args, dataroot, unlabel_transform=None, latent_dir=None, is_label=True, phase='train', 
                    limit_size=None, unlabel_limit_size=None, aug=False, resolution=256):

        self.args = args
        self.is_label = is_label


        if is_label == True:
            self.latent_dir = latent_dir
            self.data_root = os.path.join(dataroot, 'label_data')
        
            if phase == 'train':
                if limit_size is None:
                    self.idx_list = np.loadtxt(os.path.join(self.data_root, 'train_full_list.txt'), dtype=str)
                else:
                    self.idx_list = np.loadtxt(os.path.join(self.data_root, 
                                            'train_{}_list.txt'.format(limit_size)), dtype=str).reshape(-1)
            elif phase == 'val':
                if limit_size is None:
                    self.idx_list = np.loadtxt(os.path.join(self.data_root, 'val_full_list.txt'), dtype=str)
                else:
                    self.idx_list = np.loadtxt(os.path.join(self.data_root, 
                                            'val_{}_list.txt'.format(limit_size)), dtype=str).reshape(-1)
            elif phase == 'train-val':
                # concat both train and val
                if limit_size is None:
                    train_list = np.loadtxt(os.path.join(self.data_root, 'train_full_list.txt'), dtype=str)
                    val_list = np.loadtxt(os.path.join(self.data_root, 'val_full_list.txt'), dtype=str)
                    self.idx_list = list(train_list) + list(val_list)
                else:
                    train_list = np.loadtxt(os.path.join(self.data_root, 
                                            'train_{}_list.txt'.format(limit_size)), dtype=str).reshape(-1)
                    val_list = np.loadtxt(os.path.join(self.data_root, 
                                            'val_{}_list.txt'.format(limit_size)), dtype=str).reshape(-1)
                    self.idx_list = list(train_list) + list(val_list)
            else:
                self.idx_list = np.loadtxt(os.path.join(self.data_root, 'test_list.txt'), dtype=str)
        else:
            self.data_root = os.path.join(dataroot, 'unlabel_data')
            if unlabel_limit_size is None:
                self.idx_list = np.loadtxt(os.path.join(self.data_root, 'unlabel_list.txt'), dtype=str)
            else:
                self.idx_list = np.loadtxt(os.path.join(self.data_root, 'unlabel_{}_list.txt'.format(unlabel_limit_size)), dtype=str)

        self.img_dir = os.path.join(self.data_root, 'image')
        self.label_dir = os.path.join(self.data_root, 'label')

        self.phase = phase
        self.color_map = {
            0: [  0,   0,   0],
            1: [ 0,0,205],
            2: [132,112,255],
            3: [ 25,25,112],
            4: [187,255,255],
            5: [ 102,205,170],
            6: [ 227,207,87],
            7: [ 142,142,56]
        }

        self.data_size = len(self.idx_list)
        self.resolution = resolution

        self.aug = aug
        if aug == True:
            self.aug_t = A.Compose([
                            A.HorizontalFlip(p=0.5),
                            A.ShiftScaleRotate(shift_limit=0.1,
                                                scale_limit=0.2,
                                                rotate_limit=15,
                                                border_mode=cv2.BORDER_CONSTANT,
                                                value=0,
                                                mask_value=0,
                                                p=0.5),
                    ])

        self.unlabel_transform = unlabel_transform
        

    def _mask_labels(self, mask_np):
        label_size = len(self.color_map.keys())
        labels = np.zeros((label_size, mask_np.shape[0], mask_np.shape[1]))
        for i in range(label_size):
            labels[i][mask_np==i] = 1.0
        
        return labels

    
    @staticmethod
    def preprocess(img):
        image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5), inplace=True)
                    ]
                )
        img_tensor = image_transform(img)
        # normalize
        # img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
        # img_tensor = (img_tensor - 0.5) / 0.5

        return img_tensor
        

    def __len__(self):
        if hasattr(self.args, 'n_gpu') == False:
            return self.data_size
        # make sure dataloader size is larger than batchxngpu size
        return max(self.args.batch*self.args.n_gpu, self.data_size)
    
    def __getitem__(self, idx):
        if idx >= self.data_size:
            idx = idx % (self.data_size)
        img_idx = self.idx_list[idx]
        img_pil = Image.open(os.path.join(self.img_dir, img_idx)).convert('RGB').resize((self.resolution, self.resolution))
        mask_pil = Image.open(os.path.join(self.label_dir, img_idx)).convert('L').resize((self.resolution, self.resolution), resample=0)
        
        if self.is_label:
            if (self.phase == 'train' or self.phase == 'train-val') and self.aug:
                augmented = self.aug_t(image=np.array(img_pil), mask=np.array(mask_pil))
                aug_img_pil = Image.fromarray(augmented['image'])
                # apply pixel-wise transformation
                img_tensor = self.preprocess(aug_img_pil)

                mask_np = np.array(augmented['mask'])
                labels = self._mask_labels(mask_np)

                mask_tensor = torch.tensor(labels, dtype=torch.float)
                mask_tensor = (mask_tensor - 0.5) / 0.5

            else:
                img_tensor = self.preprocess(img_pil)
                mask_np = np.array(mask_pil)
                labels = self._mask_labels(mask_np)

                mask_tensor = torch.tensor(labels, dtype=torch.float)
                mask_tensor = (mask_tensor - 0.5) / 0.5
            
            return {
                'image': img_tensor,
                'mask': mask_tensor
            }
        else:
            img_tensor = self.unlabel_transform(img_pil)
            return {
                'image': img_tensor,
            }

#----------------------------------------------------------------------------
class WSIMaskDataset(Dataset):
    def __init__(self, 
                 args,
                 wsi_dir,                   # Path to WSI directory.
                 coord_dir,                 # Path to h5 coord database.
                 mask_dir,
                 class_val = None,
                 color_map = None,
                 process_list = None,       #Dataframe path of WSIs to process and their seg_levels/downsample levels that correspond to the coords
                 wsi_exten = '.svs',
                 mask_exten = '.png',
                 max_coord_per_wsi = 'inf',
                 rescale_mpp = False,
                 desired_mpp = 0.25,
                 random_seed = 0,
                 load_mode = 'openslide',
                 make_all_pipelines = False,
                 unlabel_transform=None, 
                 latent_dir=None, 
                 is_label=True, 
                 phase='train', 
                 aug=False, 
                 resolution=1024
                 ):

        self.args = args
        self.is_label = is_label
        
        #Grayscale value of masks and corresponding classes, could load from json file
        if class_val is None:
            print('Using kidney cell class values...')
            self.class_val = {
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
        else:
            assert isinstance(class_val, dict)
            self.class_val = class_val
        
        if color_map is None:
            print('Using kidney cell color map (13 classes)...')
            self.color_map = {
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
        else:
            assert isinstance(color_map, dict)
            self.color_map = color_map
        
        try:
            random.seed(random_seed)
        except Exception as e:
            print(e)
            random.seed(0)
        self.wsi_dir = wsi_dir
        self.wsi_exten = wsi_exten
        self.mask_exten = mask_exten
        self.coord_dir = coord_dir
        self.max_coord_per_wsi = max_coord_per_wsi
        if process_list is None:
            self.process_list = None
        else:
            self.process_list = pd.read_csv(process_list)
        self.patch_size = resolution
        self.rescale_mpp = rescale_mpp
        self.desired_mpp = desired_mpp
        self.load_mode = load_mode
        self.make_all_pipelines = make_all_pipelines
        #Implement labels here..
        #Need to load the wsi_pipelines after init for multiprocessing?
        self.wsi_pipelines = None

        if is_label == True:
            self.latent_dir = latent_dir
            assert isinstance(mask_dir, str)
            if os.path.isdir(mask_dir):
                self.mask_dir = mask_dir
            else:
                raise ValueError('{} does not exist. Verify mask_dir...'.format(mask_dir))
            #Load the coordinate dic & wsi dicts for the labeled images only...
            #Need a function that looks at masks in mask_dir, pulls out wsi_names and coords from filename
            self.coord_dict, self.wsi_names, self.wsi_props = self.createLabeledWSIData()
            
        else:
            self.coord_dict, self.wsi_names, self.wsi_props = self.createWSIData()
        
        self.data_size = len(self.coord_dict)
        print('Number of WSIs:', len(self.wsi_names))
        print('Number of patches:', self.data_size)
        if self.is_label:
            i_img, i_mask = self._load_raw_image(0, load_one=True)
            raw_shape = [self.data_size] + list(i_img.shape) 
        else:
            raw_shape = [self.data_size] + list(self._load_raw_image(0, load_one=True).shape)
        print('Raw shape of dataset:', raw_shape)
        if resolution is not None and (raw_shape[1] != resolution or raw_shape[2] != resolution):
            raise IOError('Image files do not match the specified resolution')
        #Trying to resolve picking of this dictionary for multiprocessing.....
        #Maybe there's a better way... maybe just load one image or add a 'test' parameter?
        del self.wsi_pipelines
        self.wsi_pipelines = None

        #__get_item__ params
        self.phase = phase
        self.aug = aug
        if aug == True:
            self.aug_t = A.Compose([
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                            #More conservative aug of rotating ever 90 degrees
                            # A.OneOf([
                            #     A.RandomRotate90(p=1),
                            #     A.Sequential([A.RandomRotate90(p=1),
                            #                   A.RandomRotate90(p=1)], p=1),
                            #     A.Sequential([A.RandomRotate90(p=1),
                            #                   A.RandomRotate90(p=1),
                            #                   A.RandomRotate90(p=1)], p=1),
                            #     ], p=0.75)
                            #The image is rotated in random angles 0 to 360deg. 
                            #May work fine because a lot of whitespace actually exists in core biopsies.
                            #However, I do not want the img generator to produce these images, just want the segmentation branch to learn from them
                            A.ShiftScaleRotate(shift_limit=0,
                                               scale_limit=0,
                                               rotate_limit=360,
                                               border_mode=cv2.BORDER_CONSTANT,
                                               value=[255,255,255],
                                               mask_value=0,
                                               p=0.75),
                    ])

        self.unlabel_transform = unlabel_transform
    
    def createLabeledWSIData(self):
        #Really only care about the files in the mask_dir
        mask_files = sorted([x for x in os.listdir(self.mask_dir) if x.endswith(self.mask_exten)])
        #Thin out by process list....
        
        #Will use all WSIs that have labels, regardless of process list.... simpler but needs to be modified to use only the process list files to control the number of labels used during training
        wsi_names_noext = sorted(list(set([m.split('_')[0] for m in mask_files])))
        #Get the real wsi_names from wsi_dir using a lookup. This is needed for getting correct extensions...
        wsi_names = sorted([w for w in os.listdir(self.wsi_dir) if os.path.splitext(w)[0] in wsi_names_noext])
        temp_wsi_dict = dict(zip(wsi_names_noext,wsi_names))
        #Extract wsi_names and coords from filename
        #very scary one liner to split mask filename to get list of coords to int, mask_name, and wsi_name
        tups = [(list(map(int,m.split('_')[1].split(self.mask_exten)[0].split(','))), m, temp_wsi_dict[m.split('_')[0]]) for m in mask_files]
        #Get the desired seg level for the patching based on process list
        wsi_props = {}
        for wsi_name in wsi_names:
            mpp = None
            seg_level = 0
            if self.process_list is not None:
                seg_level = int(self.process_list.loc[self.process_list['slide_id']==wsi_name,'seg_level'].iloc[0])
                if self.rescale_mpp and 'MPP' in self.process_list.columns:
                    mpp = float(self.process_list.loc[self.process_list['slide_id']==wsi_name,'MPP'].iloc[0])
                    seg_level = 0
                #if seg_level != 0:
                #    print('{} for {}'.format(seg_level, wsi_name))
            if self.rescale_mpp and mpp is None:
                try:
                    wsi = openslide.OpenSlide(os.path.join(self.wsi_dir, wsi_name))
                    mpp = float(wsi.properties['openslide.mpp-x'])
                    seg_level = 0
                except Exception as e:
                    print(e)
                    print(wsi_name)
                    raise ValueError('Cannot find slide MPP from process list ["MPP"] or Openslide properties. Set rescale_mpp to False to avoid this error or add slide MPPs to process list')
            wsi_props[wsi_name] = (seg_level, mpp)
            
        coord_dict = {}
        for i,t in enumerate(tups):
            #Make key a string so that it is less likely to have hash collisions...
            coord_dict[str(i)] = t
        
        return coord_dict, wsi_names, wsi_props
        
        
    def createWSIData(self):
        if self.process_list is None:
            #Only use WSI that have coord files....
            all_coord_files = sorted([x for x in os.listdir(self.coord_dir) if x.endswith('.h5')])
        else:
            #Only use WSI that coord files aren't excluded and are in coord_dir
            wsi_plist = list(self.process_list.loc[~self.process_list['exclude_ids'].isin(['y','yes','Y']),'slide_id'])
            coord_plist = sorted([os.path.splitext(x)[0]+'.h5' for x in wsi_plist])
            all_coord_files = sorted([x for x in os.listdir(self.coord_dir) if x.endswith('.h5') and x in coord_plist])
        #Get WSI filenames from path that have coord files/in process list
        wsi_names = sorted([w for w in os.listdir(self.wsi_dir) if w.endswith(tuple(self.wsi_exten)) and os.path.splitext(w)[0]+'.h5' in all_coord_files])
            
        #Get corresponding coord h5 files using WSI paths
        h5_names = [os.path.splitext(wsi_name)[0]+'.h5' for wsi_name in wsi_names]
        #Loop through coord files, get coord length, randomly choose X coords for each wsi (max_coord_per_wsi)
        coord_dict = {}
        wsi_props = {}
        # wsi_number = 0
        for h5, wsi_name in zip(h5_names, wsi_names):
            #All h5 paths must exist....
            h5_path = os.path.join(self.coord_dir, h5)
            with h5py.File(h5_path, "r") as f:
                attrs = dict(f['coords'].attrs)
                seg_level = int(attrs['patch_level'])
                dims = attrs['level_dim']
                #patch_size = attrs['patch_size']
                dset = f['coords']
                max_len = len(dset)
                if max_len < float(self.max_coord_per_wsi):
                    #Return all coords
                    coords = dset[:]
                else:
                    #Randomly select X coords
                    rand_ind = np.sort(random.sample(range(max_len), int(self.max_coord_per_wsi)))
                    coords = dset[rand_ind]

            #Get the desired seg level for the patching based on process list
            mpp = None
            seg_level = 0
            if self.process_list is not None:
                seg_level = int(self.process_list.loc[self.process_list['slide_id']==wsi_name,'seg_level'].iloc[0])
                if self.rescale_mpp and 'MPP' in self.process_list.columns:
                    mpp = float(self.process_list.loc[self.process_list['slide_id']==wsi_name,'MPP'].iloc[0])
                    seg_level = 0
                #if seg_level != 0:
                #    print('{} for {}'.format(seg_level, wsi_name))
            if self.rescale_mpp and mpp is None:
                try:
                    wsi = openslide.OpenSlide(os.path.join(self.wsi_dir, wsi_name))
                    mpp = float(wsi.properties['openslide.mpp-x'])
                    seg_level = 0
                except Exception as e:
                    print(e)
                    print(wsi_name)
                    raise ValueError('Cannot find slide MPP from process list ["MPP"] or Openslide properties. Set rescale_mpp to False to avoid this error or add slide MPPs to process list')
            
            #Check that coordinates and patch resolution is within the dimensions of the WSI... slow but only done once at beginning
            del_index = []
            # print(wsi_name)
            for i,coord in enumerate(coords):
                #Check that coordinates are inside dims
                changed = False
            #   old_coord = coord.copy()
                if coord[0]+self.patch_size > dims[0]:
                    coord[0] = dims[0]-self.patch_size
                #   print('X not in bounds, adjusting')
                    changed = True
                if coord[1]+self.patch_size > dims[1]:
                    coord[1] = dims[1]-self.patch_size
                #   print('Y not in bounds, adjusting')
                    changed = True
                if changed:
                #   print("Changing coord {} to {}".format(old_coord, coord))
                    coords[i] = coord
            
            if len(del_index) > 0:
                print('Removing {} coords that have black or white patches....'.format(len(del_index)))
                coords = np.delete(coords, del_index, axis=0)    
            
            #Store as dictionary with tuples {0: (coord, wsi_number), 1: (coord, wsi_number), etc.}
            dict_len = len(coord_dict)
            for i in range(coords.shape[0]):
                #Make key a string so that it is less likely to have hash collisions...
                coord_dict[str(i+dict_len)] = (coords[i], wsi_name)
            wsi_props[wsi_name] = (seg_level, mpp)
            #Storing number/index because smaller size than string??
            # wsi_number += 1
            
        return coord_dict, wsi_names, wsi_props

    def _mask_labels(self, mask_np):
        label_size = len(self.color_map.keys())
        class_keys = list(self.class_val.keys())
        labels = np.zeros((label_size, mask_np.shape[0], mask_np.shape[1]))
        for i in range(label_size):
            labels[i][mask_np==self.class_val[class_keys[i]]] = 1.0
        
        return labels

    
    @staticmethod
    def preprocess(img):
        image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5), inplace=True)     #Normalize between -1 and 1
                        #Using real data to normalize dist...
                        # transforms.Normalize((0.8153510093688965,
                        #                       0.6476525664329529,
                        #                       0.7707882523536682), 
                        #                       (0.035145699977874756,
                        #                       0.05645135045051575,
                        #                       0.028033018112182617), 
                        #                       inplace=True)
                    ]
                )
        img_tensor = image_transform(img)
        # normalize
        # img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
        # img_tensor = (img_tensor - 0.5) / 0.5

        return img_tensor
    
    @staticmethod    
    def adjPatchOOB(wsi_dim, coord, patch_size):
        #wsi_dim = (wsi_width, wsi_height)
        #coord = (x, y) with y axis inverted or point (0,0) starting in top left of image
        #patchsize = integer for square patch only
        #assume coord starts at (0,0) in line with original WSI,
        #therefore the patch is only out-of-bounds if the coord+patchsize exceeds the WSI dimensions
        #check dimensions, adjust coordinate if out of bounds
        coord = [int(coord[0]), int(coord[1])] 
        if coord[0]+patch_size > wsi_dim[0]:
            coord[0] = int(wsi_dim[0] - patch_size)
        
        if coord[1]+patch_size > wsi_dim[1]:
            coord[1] = int(wsi_dim[1] - patch_size) 
        
        return tuple(coord)

    def scalePatch(self, wsi, dims, coord, input_mpp=0.5, desired_mpp=0.25, patch_size=512, eps=0.05, level=0):
        desired_mpp = float(desired_mpp)
        input_mpp = float(input_mpp)
        #downsample > 1, upsample < 1
        factor = desired_mpp/input_mpp
        #Openslide get dimensions of full WSI
        # dims = wsi.level_dimensions[0]
        if input_mpp > desired_mpp + eps or input_mpp < desired_mpp - eps:
            #print('scale by {:.2f} factor'.format(factor))
            # if factor > 1
            #input mpp must be smaller and therefore at higher magnification (e.g. desired 40x vs input 60x) and vice versa
            #approach: shrink a larger patch by factor to the desired patch size or enlarge a smaller patch to desired patch size
            #if factor > 1 and you are downsampling the image, it can be faster to load the downsample level that is closest to the factor
            #get the level that is closest to the factor... really only care if factor > 2 because tiled images increment downsample levels in factors of 2 or 4 typically.
            downsample_at_new_level = 1
            if factor >= 2 and self.load_mode == 'openslide':
                # print('Factor:', factor)
                #Downsamples aren't integers in TCGA data..... but typically they are in increments of 2 anyways.
                level = wsi.get_best_level_for_downsample(int(math.ceil(factor))+0.5)
                downsample_at_new_level = wsi.level_downsamples[level]
                #update factor to scale based on new level. If factor was 5 and the downsample at new level is 4, 
                #then you need to still scale scaled_psize/(downsample*patch_size) == (5*1024)/(4*1024) = 1.25
                factor = factor/downsample_at_new_level
                # print('Adj Factor:', factor)
                # print('Level:', level)
                # print('Downsample at new level:', downsample_at_new_level)
            #Don't know how I could do this in pyvips unless I can get the downsample level metadata...... could try and check if the pyvips image was openslide compatible...?
            if factor >= 1.25 or factor <= 0.75:
                scaled_psize = int(patch_size*factor)
                #check and adjust dimensions of coord based on scaled patchsize relative to level 0
                coord = self.adjPatchOOB(dims, coord, int(scaled_psize*downsample_at_new_level))
                adj_patch = self._load_patch(wsi, level, coord, scaled_psize, dims=dims)
                #shrink patch down to desired mpp if factor > 1
                #enlarge if factor < 1
                #Could implement fully in vips...
                patch = cv2.resize(adj_patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
                return patch
            else:
                coord = self.adjPatchOOB(dims, coord, patch_size)
                patch = self._load_patch(wsi, level, coord, patch_size, dims=dims)
                return patch    
        else: 
            #print('skip scaling factor {:.2f}. input um per pixel ({}) within +/- {} of desired MPP ({}).'.format(factor, input_mpp, eps, desired_mpp))
            coord = self.adjPatchOOB(dims, coord, patch_size)
            patch = self._load_patch(wsi, level, coord, patch_size, dims=dims)
            return patch
    
    @staticmethod
    def fetch(region, patch_size, x, y):
        return region.fetch(x, y, patch_size, patch_size)
    
    def vips_readRegion(self, region, level, patch_size, x, y, mode="RGBA", ref_level=0):
        #Assumes that region is at desired level, therefore coordinates x,y need to be downsampled
        #because x,y coords are referenced at level=0
        assert isinstance(level, int)
        assert isinstance(ref_level, int)
        if level < 0:
            level = 0
        if ref_level < 0:
            ref_level = 0
        downsample = 2**(level-ref_level)
        x,y = x//downsample, y//downsample
        patch = self.fetch(region, patch_size, int(x), int(y))
        return Image.frombuffer(mode, (patch_size, patch_size), patch, 'raw', mode, 0, 1)
    
    def vips_crop(self, wsi_vips, level, patch_size, x, y, mode="RGBA", ref_level=0):
        #Assumes that region is at desired level, therefore coordinates x,y need to be downsampled
        #because x,y coords are referenced at level=0
        assert isinstance(level, int)
        assert isinstance(ref_level, int)
        if level < 0:
            level = 0
        if ref_level < 0:
            ref_level = 0
        downsample = 2**(level-ref_level)
        x,y = x//downsample, y//downsample
        patch = wsi_vips.crop(int(x), int(y), patch_size, patch_size)
        return Image.frombuffer(mode, (patch_size, patch_size), patch.write_to_memory(), 'raw', mode, 0, 1)
    
    def _load_wsi_pipelines(self, load_wsi_by_name = None):
        #Create all the image pipelines in a dictionary
        wsi_pipelines = {}
        
        if load_wsi_by_name is not None:
            if isinstance(load_wsi_by_name, list):
                load_WSIs = load_wsi_by_name
            else:
                load_WSIs = [load_wsi_by_name]
        else:
            load_WSIs = self.wsi_names
        
        #vips method
        if self.load_mode == 'vips':
            for wsi_name in load_WSIs:
                seg_level, mpp = self.wsi_props[wsi_name]
                if os.path.splitext(wsi_name)[1]=='.tiff' or os.path.splitext(wsi_name)[1]=='.tif':
                    wsi_vips = pyvips.Image.new_from_file(os.path.join(self.wsi_dir, wsi_name))
                else:                
                    wsi_vips = pyvips.Image.openslideload(os.path.join(self.wsi_dir, wsi_name), level=seg_level)
                reg = pyvips.Region.new(wsi_vips)
                if seg_level==0:
                    dims = [wsi_vips.width, wsi_vips.height, wsi_vips.bands]
                else:
                    assert isinstance(seg_level, int)
                    if seg_level < 0:
                        seg_level = 0
                    downsample = 2**seg_level
                    dims = [int(wsi_vips.width*downsample), int(wsi_vips.height*downsample), wsi_vips.bands]
                wsi_pipelines[wsi_name] = (reg, dims)
            return wsi_pipelines
        elif self.load_mode == 'vips-crop':
            for wsi_name in load_WSIs:
                seg_level, mpp = self.wsi_props[wsi_name]
                if os.path.splitext(wsi_name)[1]=='.tiff' or os.path.splitext(wsi_name)[1]=='.tif':
                    wsi_vips = pyvips.Image.new_from_file(os.path.join(self.wsi_dir, wsi_name))
                else:                
                    wsi_vips = pyvips.Image.openslideload(os.path.join(self.wsi_dir, wsi_name), level=seg_level)
                # reg = pyvips.Region.new(wsi_vips)
                if seg_level==0:
                    dims = [wsi_vips.width, wsi_vips.height, wsi_vips.bands]
                else:
                    assert isinstance(seg_level, int)
                    if seg_level < 0:
                        seg_level = 0
                    downsample = 2**seg_level
                    dims = [int(wsi_vips.width*downsample), int(wsi_vips.height*downsample), wsi_vips.bands]
                wsi_pipelines[wsi_name] = (wsi_vips, dims)
            return wsi_pipelines
        #openslide method (slower than vips!)
        elif self.load_mode == 'openslide':
            for wsi_name in load_WSIs:
                seg_level, mpp = self.wsi_props[wsi_name]
                wsi = openslide.OpenSlide(os.path.join(self.wsi_dir, wsi_name))
                dims = wsi.level_dimensions[seg_level]
                wsi_pipelines[wsi_name] = wsi, dims
            return wsi_pipelines
        else:
            raise ValueError('Load mode {} not supported! Modes {} are supported.'.format(self.load_mode, ['vips', 'openslide']))

    def _load_one_wsi(self, wsi_name):        
        #vips method
        if self.load_mode == 'vips':
            seg_level, mpp = self.wsi_props[wsi_name]
            wsi_vips = pyvips.Image.openslideload(os.path.join(self.wsi_dir, wsi_name), level=seg_level, access='sequential')
            reg = pyvips.Region.new(wsi_vips)
            if seg_level==0:
                dims = [wsi_vips.width, wsi_vips.height, wsi_vips.bands]
            else:
                assert isinstance(seg_level, int)
                if seg_level < 0:
                    seg_level = 0
                downsample = 2**seg_level
                dims = [int(wsi_vips.width*downsample), int(wsi_vips.height*downsample), wsi_vips.bands]
            return reg, dims
        elif self.load_mode == 'vips-crop':
            seg_level, mpp = self.wsi_props[wsi_name]
            wsi_vips = pyvips.Image.openslideload(os.path.join(self.wsi_dir, wsi_name), level=seg_level, access='sequential')
            # reg = pyvips.Region.new(wsi_vips)
            if seg_level==0:
                dims = [wsi_vips.width, wsi_vips.height, wsi_vips.bands]
            else:
                assert isinstance(seg_level, int)
                if seg_level < 0:
                    seg_level = 0
                downsample = 2**seg_level
                dims = [int(wsi_vips.width*downsample), int(wsi_vips.height*downsample), wsi_vips.bands]
            return wsi_vips, dims
        #openslide method (slower than vips!)
        elif self.load_mode == 'openslide':
            seg_level, mpp = self.wsi_props[wsi_name]
            wsi = openslide.OpenSlide(os.path.join(self.wsi_dir, wsi_name))
            dims = wsi.level_dimensions[seg_level]
            return wsi, dims
        else:
            raise ValueError('Load mode {} not supported! Modes {} are supported.'.format(self.load_mode, ['vips', 'vips-crop', 'openslide']))
    
    def _load_patch(self, wsi, level, coord, patch_size, dims=None):
        if self.load_mode == 'openslide':
            patch = np.array(wsi.read_region(coord, level, (patch_size, patch_size)).convert('RGB'))
        elif self.load_mode == 'vips':
            if dims[2] == 4:
                patch = np.array(self.vips_readRegion(wsi, level, patch_size, coord[0], coord[1], mode='RGBA').convert('RGB'))
            elif dims[2] == 3:
                #print('pyvips opened image as RGB')
                patch = np.array(self.vips_readRegion(wsi, level, patch_size, coord[0], coord[1], mode='RGB'))
            else:
                raise ValueError('Mode for image (RGB or RGBA) not specified/supported. Cannot use vips to open and scale patches..')
        elif self.load_mode == 'vips-crop':
            if dims[2] == 4:
                patch = np.array(self.vips_crop(wsi, level, patch_size, coord[0], coord[1], mode='RGBA').convert('RGB'))
            elif dims[2] == 3:
                #print('pyvips opened image as RGB')
                patch = np.array(self.vips_crop(wsi, level, patch_size, coord[0], coord[1], mode='RGB'))
            else:
                raise ValueError('Mode for image (RGB or RGBA) not specified/supported. Cannot use vips to open and scale patches..')
        else:
            raise ValueError('Load mode {} not supported! Modes {} are supported.'.format(self.load_mode, ['vips', 'openslide']))
        return patch
    
    def _load_raw_image(self, raw_idx, load_one=False):
        
        if self.is_label:
            coord, mask_name, wsi_name = self.coord_dict[str(raw_idx % self.data_size)]
        else:
            coord, wsi_name = self.coord_dict[str(raw_idx % self.data_size)]
        
        seg_level, mpp = self.wsi_props[wsi_name]
        
        #Load wsi first...
        if self.make_all_pipelines:
            if self.wsi_pipelines is None:
                #load pipelines first
                if load_one:
                    #For the test image for init
                    self.wsi_pipelines = self._load_wsi_pipelines(load_wsi_by_name=wsi_name)
                else:
                    self.wsi_pipelines = self._load_wsi_pipelines()
            wsi, dims = self.wsi_pipelines[wsi_name]
        else:
            wsi, dims = self._load_one_wsi(wsi_name)
        
        #Load wsi patch
        #Do not try to rescale the labeled patches... they should be of the desired mpp, in this implementation.
        if self.rescale_mpp and not self.is_label:
            if mpp is None and self.load_mode == 'openslide':
                try:
                    mpp = wsi.properties['openslide.mpp-x']
                except Exception as e:
                    print(e)
                    print(wsi_name)
                    raise ValueError('Cannot find slide MPP from process list or Openslide properties. Set rescale_mpp to False to avoid this error or add slide MPPs to process list.')
            elif mpp is None and self.load_mode != 'openslide':
                raise ValueError('Cannot find slide MPP from process list. Cannot use mode load_mode {} if MPP not in process list. Change load_mode to "openslide", set rescale_mpp to False, or slide MPPs to process list to avoid error.'.format(self.load_mode))
            img = self.scalePatch(wsi=wsi, dims=dims, coord=coord, input_mpp=mpp, desired_mpp=self.desired_mpp, patch_size=self.patch_size, level=seg_level) 
        else:
            img = self._load_patch(wsi, seg_level, coord, self.patch_size, dims=dims)
        
        if self.is_label:
            #Load label mask
            mask = np.array(Image.open(os.path.join(self.mask_dir, mask_name)).convert('L'))
            return img, mask
        else:
            return img

    def __len__(self):
        if hasattr(self.args, 'n_gpu') == False:
            return self.data_size
        # make sure dataloader size is larger than batchxngpu size
        return max(self.args.batch*self.args.n_gpu, self.data_size)
    
    def __getitem__(self, idx):
        if self.is_label:
            img, mask = self._load_raw_image(idx, load_one=False)
            if (self.phase == 'train' or self.phase == 'train-val') and self.aug:
                augmented = self.aug_t(image=img, mask=mask)
                aug_img_pil = Image.fromarray(augmented['image'])
                # apply pixel-wise transformation
                img_tensor = self.preprocess(aug_img_pil)

                mask_np = np.array(augmented['mask'])
                labels = self._mask_labels(mask_np)

                mask_tensor = torch.tensor(labels, dtype=torch.float)
                mask_tensor = (mask_tensor - 0.5) / 0.5

            else:
                img_pil = Image.fromarray(img)
                img_tensor = self.preprocess(img_pil)
                mask_np = mask
                labels = self._mask_labels(mask_np)

                mask_tensor = torch.tensor(labels, dtype=torch.float)
                mask_tensor = (mask_tensor - 0.5) / 0.5
            
            return {
                'image': img_tensor,
                'mask': mask_tensor
            }
        else:
            img = self._load_raw_image(idx, load_one=False)
            img_pil = Image.fromarray(img)
            if self.unlabel_transform is None:
                img_tensor = self.preprocess(img_pil)
            else:
                img_tensor = self.unlabel_transform(img_pil)
            return {
                'image': img_tensor,
            }
        
    
        
if __name__ == '__main__':
    import argparse
    import shlex
    import matplotlib.pyplot as plt
    from torchvision import utils
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.n_gpu = 4
    args.batch = 16
    wsi_dir = '/home/cjr66/project/KID-DeepLearning/KID-Images-pyramid'
    coord_dir = '/home/cjr66/project/KID-DeepLearning/Patch_coords-1024/MP_KPMP_all-patches-stride256'
    mask_dir = '/home/cjr66/project/KID-DeepLearning/Labeled_patches/MP_1024_stride256'
    process_list = '/home/cjr66/project/KID-DeepLearning/proc_info/MP_only-KID_process_list.csv'
    rescale_mpp = True
    desired_mpp = 0.2
    wsi_exten = ['.tif','.svs']
    kidData = WSIMaskDataset(args, wsi_dir, coord_dir, mask_dir, process_list = process_list,
                             wsi_exten=wsi_exten, rescale_mpp=True, desired_mpp=desired_mpp, is_label=False, 
                             aug=False, resolution=1024)
    
    
    for i in range(0,10,1):
        dget = kidData.__getitem__(i)
        # img, mask = dget['image'], dget['mask']
        img = dget['image']
        img = img.permute(1,2,0).numpy()
        # mask = torch.argmax(mask, dim=0)
        # sample_mask = torch.zeros((mask.shape[0], mask.shape[1], 3), dtype=torch.float)
        # color_map = kidData.color_map
        # for key in color_map:
            # sample_mask[mask==key] = torch.tensor(color_map[key], dtype=torch.float)
        # sample_mask = sample_mask.permute(0,3,1,2)
        
        
        # img = kidData._load_raw_image(0)
        
        plt.figure()
        # plt.subplot(1,2,1)
        plt.imshow(img)
        # plt.subplot(1,2,2)
        # plt.imshow(sample_mask)
        plt.show()
    
    # save_dir = '/home/cjr66/project/KID-DeepLearning/Labeled_patches'
    # utils.save_image(
    #         sample_mask,
    #         os.path.join(save_dir, 'test.png')
    # )
    # utils.save_image(
    #         sample_mask,
    #         os.path.join(save_dir, 'test.png'),
    #         nrow=int(args.n_sample ** 0.5),
    #         normalize=True,
    # )