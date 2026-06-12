import argparse
import os
import shutil
import time
import warnings
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
from utils.loader import get_training_data

from M290_MachineSimu_GPU.complex_field_tools_GPU.complex_field_tools import *
from M290_MachineSimu_GPU.optical_components_GPU.Apertures import CircularAperture
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('-b', '--batch_size', default=20, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--beamshape', default='RecTophat', type=str)
parser.add_argument('--caustic_plane', default='prefoc', type=str)
parser.add_argument('--optimizer', default='adam', type=str,
                    help='Optimizer algorithm - "adam" for Adam optimizer, "sgd" for SGD optimizer.')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--step_size', default=50, type=int,
                    help='step size (default: 50)')
parser.add_argument('--full_scene', default=False, type=bool)


iteration = 100

if __name__ == '__main__':
    
    args = parser.parse_args()
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

    beamshape = args.beamshape
    plane = args.caustic_plane
    batchsize = args.batch_size
    fullscene = args.full_scene
    
    if (fullscene==False):
        from M290_MachineSimu_GPU.M290 import M290
    if (fullscene==True):
        from M290_MachineSimu_GPU.M290_fullscene import M290

    if(fullscene==False):
        lightsource_path = 'lightsource_norm.npy'
        print("Using normalized aperture-scene lightsource!")
    else:
        lightsource_path = 'lightsource_norm_fullscene.npy'
        print("Using normalized full-scene lightsource!")
    
    vis_dir = 'prediction_result/'

    train_dataset = get_training_data('./')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize, shuffle=False,
        num_workers=1, pin_memory=True, sampler=None)
    
    machine = M290(batchsize, beamshape, lightsource_path, device, plane).to(device)

    near_field = machine.nearField
    
    criterion = nn.MSELoss().to(device)
    if args.lr > 0:
        lr = args.lr
    
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(machine.parameters(), lr=lr,
                                     weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(machine.parameters(), lr=lr, momentum=0.9,
                                    weight_decay=1e-4)
    if args.step_size > 0:
        step_size = args.step_size
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.5)
    
    for i, (beamshape, Z, name) in enumerate(train_loader):
        print(name)
        
        beamshape = beamshape.to(device)
        
        #-----------------------------Sanity check start---------------------------------------------------------
        
        Zernike_Coeffs = Z.to(device, non_blocking=True).squeeze()
        with torch.no_grad():
            machine.zernike_coeffs.copy_(Zernike_Coeffs.view(batchsize, 12, 1, 1))
        imaging_field, phase = machine(near_field)
        
        I = torch.abs(imaging_field)**2
        max_per_sample = I.amax(dim=(1,2), keepdim=True)
        I = I / max_per_sample
        I = I * 255.0
        I = I.detach().cpu().numpy()
        
        phase = CircularAperture(phase, machine.apertureRadius, machine.gridSize)
        phase = phase[: , machine.start_idx:machine.end_idx, machine.start_idx:machine.end_idx]
        phase = phase.detach().cpu().numpy()
        print(I.shape)
        
        for j in range(len(I)):
            np.save(vis_dir+"/gt_I/"+name[j][:7]+'_intensity.npy', I[j])
            np.save(vis_dir+"/gt_phi/"+name[j][:7]+'_phase.npy', phase[j])
        
        #-----------------------------Sanity check over---------------------------------------------------------
        
        with torch.no_grad():
            machine.zernike_coeffs.zero_()
        
        for j in range(iteration):
            
            imaging_field, phase = machine(near_field)

            I = torch.abs(imaging_field)**2
            max_per_sample = I.amax(dim=(1,2), keepdim=True)
            I = I / max_per_sample
            I = I * 255.0
            
            loss = criterion(I, beamshape)
            
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            
            print("Batch "+str(i)+" Iteration "+str(j))
            print(loss.item())
        
        phase = CircularAperture(phase, machine.apertureRadius, machine.gridSize)
        phase = phase[: , machine.start_idx:machine.end_idx, machine.start_idx:machine.end_idx]
        phase = phase.detach().cpu().numpy()
        I = I.cpu().detach().numpy()
        z_pred = machine.zernike_coeffs.squeeze().cpu().detach().numpy()
        
        for j in range(len(I)):
            mpimg.imsave(vis_dir+name[j][:7]+'_intensity.png', -I[j], cmap='Greys')
            np.save(vis_dir+"/pred_I/"+name[j][:7]+'_intensity.npy', I[j])
            mpimg.imsave(vis_dir+name[j][:7]+'_phase.png', phase[j], cmap='Greys')
            np.save(vis_dir+"/pred_phi/"+name[j][:7]+'_phase.npy', phase[j])
            np.save(vis_dir+"/pred_z/"+name[j][:7]+'_zernike.npy', z_pred[j].squeeze())
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr