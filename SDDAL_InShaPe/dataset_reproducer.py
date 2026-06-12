import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import sys
import os

from utils.loader import get_reproduce_data

from M290_MachineSimu_GPU.complex_field_tools_GPU.complex_field_tools import *
from M290_MachineSimu_GPU.optical_components_GPU.Apertures import CircularAperture


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--beamshape', default='rec', type=str)
parser.add_argument('--caustic_plane', default='prefoc', type=str)


if __name__ == '__main__':
    
    seed = 12345
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    
    beamshape = args.beamshape
    intrinsic = args.beamshape[:-2]
    vis_dir = "Design_" + beamshape
    plane = args.caustic_plane
    
    lightsource_path = 'M290_MachineSimu_GPU/lightsource_full_scene.npy'
    
    from M290_Repro.M290_full_scene import M290
    machine = M290(1, intrinsic, lightsource_path, device)
    machine = machine.to(device)
    near_field = machine.nearField
    
    batchsize = 1
    train_dataset = get_reproduce_data(vis_dir)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        sampler=None
    )
    
    intensity_npy_dir = os.path.join(vis_dir, "training_set", "intensity", "npy")
    intensity_img_dir = os.path.join(vis_dir, "training_set", "intensity", "img")
    phase_npy_dir = os.path.join(vis_dir, "training_set", "phase", "npy")
    phase_img_dir = os.path.join(vis_dir, "training_set", "phase", "img")

    os.makedirs(intensity_npy_dir, exist_ok=True)
    os.makedirs(intensity_img_dir, exist_ok=True)
    os.makedirs(phase_npy_dir, exist_ok=True)
    os.makedirs(phase_img_dir, exist_ok=True)
    
    for i, (Z, name) in enumerate(train_loader):
        
        print(name)
        
        Zernike_Coeffs = Z.to(device, non_blocking=True).squeeze()
        
        with torch.no_grad():
            machine.zernike_coeffs.copy_(Zernike_Coeffs.view(batchsize, 12, 1, 1))
        
        imaging_field, phase = machine(near_field)
        
        I = torch.abs(imaging_field) ** 2
        max_per_sample = I.amax(dim=(1, 2), keepdim=True)
        I = I / max_per_sample
        I = I * 255.0
        I = I.detach().cpu().numpy()
        
        phase = CircularAperture(phase, machine.apertureRadius, machine.gridSize)
        phase = phase[:, machine.start_idx:machine.end_idx, machine.start_idx:machine.end_idx]
        phase = phase.detach().cpu().numpy()
        
        for j in range(len(I)):
            
            nname = os.path.basename(name[j])
            
            intensity_npy_path = os.path.join(
                intensity_npy_dir,
                "intensity" + nname[8:]
            )
            
            intensity_img_path = os.path.join(
                intensity_img_dir,
                "intensity" + nname[8:-3] + "png"
            )
            
            phase_npy_path = os.path.join(
                phase_npy_dir,
                "phase" + nname[8:]
            )
            
            phase_img_path = os.path.join(
                phase_img_dir,
                "phase" + nname[8:-3] + "png"
            )
            
            np.save(intensity_npy_path, I[j])
            mpimg.imsave(intensity_img_path, -I[j], cmap='Greys')
            
            np.save(phase_npy_path, phase[j])
            mpimg.imsave(phase_img_path, phase[j], cmap='Greys')