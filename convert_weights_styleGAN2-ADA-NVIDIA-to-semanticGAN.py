#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 09:46:19 2022

@author: cjr66
"""

import re
from pathlib import Path
import os
os.chdir('/home/cjr66/project/semanticGAN_WSI')
from collections import OrderedDict

import click
import numpy as np
import torch

import dnnlib
import legacy


def convert_to_rgb_G(state_semG, state_G_nv, sem_name, nv_name):
    state_semG[f"{sem_name}.conv.weight"] = state_G_nv[f"{nv_name}.torgb.weight"].unsqueeze(0)
    state_semG[f"{sem_name}.bias"] = state_G_nv[f"{nv_name}.torgb.bias"].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    state_semG[f"{sem_name}.conv.modulation.weight"] = state_G_nv[f"{nv_name}.torgb.affine.weight"]
    state_semG[f"{sem_name}.conv.modulation.bias"] = state_G_nv[f"{nv_name}.torgb.affine.bias"]


def convert_conv_G(state_semG, state_G_nv, sem_name, nv_name):
    state_semG[f"{sem_name}.conv.weight"] = state_G_nv[f"{nv_name}.weight"].unsqueeze(0)
    state_semG[f"{sem_name}.activate.bias"] = state_G_nv[f"{nv_name}.bias"]
    state_semG[f"{sem_name}.conv.modulation.weight"] = state_G_nv[f"{nv_name}.affine.weight"]
    state_semG[f"{sem_name}.conv.modulation.bias"] = state_G_nv[f"{nv_name}.affine.bias"]
    state_semG[f"{sem_name}.noise.weight"] = state_G_nv[f"{nv_name}.noise_strength"].unsqueeze(0)


def convert_blur_kernel_G(state_semG, state_G_nv, level):
    """Not quite sure why there is a factor of 4 here"""
    # They are all the same
    state_semG[f"convs.{2*level}.conv.blur.kernel"] = 4*state_G_nv["synthesis.b4.resample_filter"]
    state_semG[f"to_rgbs.{level}.upsample.kernel"] = 4*state_G_nv["synthesis.b4.resample_filter"]


def convert_conv_D(state_semD, state_D_nv, j, nv_layer):
    # state_semD[f"convs.{j}.conv1.0.kernel"] = state_D_nv[f"b{nv_layer}.conv0.resample_filter"]
    state_semD[f"convs.{j}.conv1.0.weight"] = state_D_nv[f"b{nv_layer}.conv0.weight"]
    state_semD[f"convs.{j}.conv1.1.bias"] = state_D_nv[f"b{nv_layer}.conv0.bias"]
    state_semD[f"convs.{j}.conv2.0.kernel"] = state_D_nv[f"b{nv_layer}.conv1.resample_filter"]
    state_semD[f"convs.{j}.conv2.1.weight"] = state_D_nv[f"b{nv_layer}.conv1.weight"]
    state_semD[f"convs.{j}.conv2.2.bias"] = state_D_nv[f"b{nv_layer}.conv1.bias"]
    state_semD[f"convs.{j}.skip.0.kernel"] = state_D_nv[f"b{nv_layer}.skip.resample_filter"]
    state_semD[f"convs.{j}.skip.1.weight"] = state_D_nv[f"b{nv_layer}.skip.weight"]

def determine_config(state_G_nv):
    mapping_names = [name for name in state_G_nv.keys() if "mapping.fc" in name]
    sythesis_names = [name for name in state_G_nv.keys() if "synthesis.b" in name]

    n_mapping =  max([int(re.findall("(\d+)", n)[0]) for n in mapping_names]) + 1
    resolution =  max([int(re.findall("(\d+)", n)[0]) for n in sythesis_names])
    n_layers = np.log(resolution/2)/np.log(2)

    return n_mapping, n_layers


# @click.command()
# @click.argument("network-pkl")
# @click.argument("output-file")
# @click.argument("init-res")
def convert(network_pkl, output_file, init_res=1024):
    with dnnlib.util.open_url(network_pkl) as f:
        G_nvidia = legacy.load_network_pkl(f)["G_ema"]
    
    with dnnlib.util.open_url(network_pkl) as f:
        D_nvidia = legacy.load_network_pkl(f)['D']

    state_G_nv = G_nvidia.state_dict()
    state_D_nv = D_nvidia.state_dict()
    n_mapping, n_layers = determine_config(state_G_nv)

    # state_semG = OrderedDict()
    # state_semD = OrderedDict()
    torch.device('cpu')
    if init_res == 1024:
        # ckpt = torch.load('/home/cjr66/project/KID-DeepLearning/semanticGAN_results/init13_KID-MP-run-Jan26_00-01-06/ckpt/000000.pt', map_location=torch.device('cpu'))
        ckpt = torch.load('/home/cjr66/project/KID-DeepLearning/semanticGAN_results/init10_KID-MP-run-Jan26_18-21-35/ckpt/000000.pt', map_location=torch.device('cpu'))
    elif init_res == 512:
        pass
    elif init_res == 256:
        pass
    else:
        raise Exception(f"init_res={init_res} not implemented! Make the semanticGAN network and change network .pt in code....")
    state_semG = ckpt['g'].copy()
    state_semD = ckpt['d_img'].copy()
    
    #Convert generator, rgb portion only
    print('converting generator weights...')
    for i in range(n_mapping):
        print('mapping layer',i+1)
        state_semG[f"style.{i+1}.weight"] = state_G_nv[f"mapping.fc{i}.weight"]
        state_semG[f"style.{i+1}.bias"] = state_G_nv[f"mapping.fc{i}.bias"]

    for i in range(int(n_layers)):
        if i > 0:
            print('synthesis layer', i+1,': res',4*(2**i))
            for conv_level in range(2):
                convert_conv_G(state_semG, state_G_nv, f"convs.{2*i-2+conv_level}", f"synthesis.b{4*(2**i)}.conv{conv_level}")
                state_semG[f"noises.noise_{2*i-1+conv_level}"] = state_G_nv[f"synthesis.b{4*(2**i)}.conv{conv_level}.noise_const"].unsqueeze(0).unsqueeze(0)

            convert_to_rgb_G(state_semG, state_G_nv, f"to_rgbs.{i-1}", f"synthesis.b{4*(2**i)}")
            convert_blur_kernel_G(state_semG, state_G_nv, i-1)
        
        else:
            print('input layer',i+1,': res', 4*(2**i))
            state_semG["input.input"] = state_G_nv[f"synthesis.b{4*(2**i)}.const"].unsqueeze(0)
            convert_conv_G(state_semG, state_G_nv, "conv1", f"synthesis.b{4*(2**i)}.conv1")
            state_semG[f"noises.noise_{2*i}"] = state_G_nv[f"synthesis.b{4*(2**i)}.conv1.noise_const"].unsqueeze(0).unsqueeze(0)
            convert_to_rgb_G(state_semG, state_G_nv, "to_rgb1", f"synthesis.b{4*(2**i)}")
    
    #Convert discriminator for real/fake images
    #b1024.fromrgb.weight =? convs.0.0 [96] 
    #b1024.fromrgb.bias =? convs.0.1 [32]
    #...
    #b8.conv0 == convs.8.conv1
    #b8.conv1 == convs.8.conv2
    #b8.skip == convs.8.skip
    #b4.conv.weight =? final_conv.0.weight [2363904]
    #b4.conv.bias =? final_conv.1.bias [512]
    #b4.fc == final_linear.0
    #b4.out == final_linear.1
    
    #Not as many resample filters/kernels in semGAN discriminator compared to NV discriminator
    print('converting discriminator weights...')
    for i in range(int(n_layers)):
        # print(i)
        nv_layer = int(2**(n_layers+1-i))
        if i > 0 and i < n_layers-1:
            print('working on', i+1,': res',nv_layer)
            convert_conv_D(state_semD, state_D_nv, i+1, nv_layer)        
        #Last layers: final conv, final linear
        elif i == n_layers-1:
            print('working on last layer', i+1,': res',nv_layer)
            state_semD["final_conv.0.weight"] = state_D_nv['b4.conv.weight']
            state_semD["final_conv.1.bias"] = state_D_nv['b4.conv.bias']
            #b4.conv.resample_filter???
            state_semD["final_linear.0.weight"] = state_D_nv['b4.fc.weight']
            state_semD["final_linear.0.bias"] = state_D_nv['b4.fc.bias']
            state_semD["final_linear.1.weight"] = state_D_nv['b4.out.weight']
            state_semD["final_linear.1.bias"] = state_D_nv['b4.out.bias']
            
        else:
            print('working on', i+1,': res',nv_layer)
            state_semD[f"convs.{i}.0.weight"] = state_D_nv[f"b{nv_layer}.fromrgb.weight"]
            state_semD[f"convs.{i}.1.bias"] = state_D_nv[f"b{nv_layer}.fromrgb.bias"]
            #b1024.fromrgb.resample_filter???
            convert_conv_D(state_semD, state_D_nv, i+1, nv_layer)
            
    #count kernels and resample_filter
    # import re
    # r = re.compile(".*kernel")
    # newlist = list(filter(r.match, list(state_semD.keys()))) # Read Note below
    # len(newlist)
    # r = re.compile(".*resample_filter")
    # newlist = list(filter(r.match, list(state_D_nv.keys()))) # Read Note below
    # len(newlist)
            
    
    # https://github.com/yuval-alaluf/restyle-encoder/issues/1#issuecomment-828354736
    latent_avg = state_G_nv['mapping.w_avg']
    state_dict = OrderedDict()
    state_dict['g'] = state_semG
    state_dict['d_img'] = state_semD
    state_dict['d_seg'] = ckpt['d_seg']
    state_dict['g_ema'] = state_semG
    state_dict['args'] = ckpt['args']
    state_dict['latent_avg'] = latent_avg
    print('saving to {}'.format(output_file))
    torch.save(state_dict, output_file)

if __name__ == "__main__":
    network_pkl = '/home/cjr66/project/KID-DeepLearning/NV-StyleGANv2-ADA_results/00144-KID_1024-MP-NV-StyleGANv2-ADA-mirror-KIDgan-g_lr2.5e-05-d_lr3e-05-gamma10-ema20-batch64-resumecustom/network-snapshot-013075.pkl'
    output_file = '/home/cjr66/project/KID-DeepLearning/semanticGAN_results/KID_1024-MP-semGAN10cell-from-013075.pt'
    convert(network_pkl, output_file, init_res=1024)
    torch.device('cpu')
    check = torch.load('/home/cjr66/project/KID-DeepLearning/semanticGAN_results/KID_1024-MP-semGAN10cell-from-013075.pt', map_location=torch.device('cpu'))
    print(check.keys())
