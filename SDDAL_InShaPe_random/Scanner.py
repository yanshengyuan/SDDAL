import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import StepLR

from Official_batched_fitting_DIC_fullscene.utils.loader import get_training_data

from M290_MachineSimu_GPU.complex_field_tools_GPU.complex_field_tools import *
from M290_MachineSimu_GPU.optical_components_GPU.Apertures import CircularAperture

from QuantUNetT_model import QuantUNetT as PImodel

import sys

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('-b', '--batch_size', default=20, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--beamshape', default='rec', type=str)
parser.add_argument('--caustic_plane', default='prefoc', type=str)
parser.add_argument('--optimizer', default='adam', type=str,
                    help='Optimizer algorithm - "adam" for Adam optimizer, "sgd" for SGD optimizer.')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--step_size', default=50, type=int,
                    help='step size (default: 50)')
parser.add_argument('--pth_name', default='',type=str)
parser.add_argument('--round_sampling', type=int)
parser.add_argument('--vis_path', default='', type=str)
parser.add_argument('--init_seed', default=None, type=int,
                    help='Random seed used only for initializing zernike coefficients.')
parser.add_argument('--reset_seed', default=None, type=int,
                    help='Random seed used only for random reset of out-of-bound zernike coefficients.')

iteration = 50

class UtilityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, low, high):
        
        interval = high-low
        interval = torch.clamp(interval, min = 0.0)
        mean_interval = torch.mean(interval)
        
        loss = -mean_interval
        return loss

class uniformity_loss(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, z, bins=20, rag=(-1.5, 1.5)):
        
        hist = torch.histc(z, bins=bins, min=rag[0], max=rag[1])
        hist = hist / torch.sum(hist)
        uniform = torch.ones_like(hist) / bins
        loss = torch.sum((hist - uniform) ** 2)
        return loss

if __name__ == '__main__':
    
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    args = parser.parse_args()
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        init_gen = torch.Generator(device=device)
    init_gen.manual_seed(args.init_seed)
    
    if torch.cuda.is_available():
        reset_gen = torch.Generator(device=device)
    reset_gen.manual_seed(args.reset_seed)
    
    vis_dir = args.vis_path
    beamshape = args.beamshape
    plane = args.caustic_plane
    batchsize = args.batch_size
    
    lightsource_path = 'M290_MachineSimu_GPU/lightsource_full_scene.npy'
    
    '''
    #-------------------------Sanity check with RecTophat starts---------------------------
    
    from M290_MachineSimu_GPU.M290_sanity_check import M290
    machine = M290(1 ,args.beamshape, lightsource_path, device)
    machine = machine.to(device)
    near_field = machine.nearField
    
    batchsize = 1
    train_dataset = get_training_data('Official_batched_fitting_DIC_fullscene')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize, shuffle=False,
        num_workers=1, pin_memory=True, sampler=None)
    
    print("Sanity check sample generation with RecTophat starts now!")
    for i, (beamshape, Z, name) in enumerate(train_loader):
        
        print(name)
        
        beamshape = beamshape.to(device)
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
        
        for j in range(len(I)):
            np.save("rec_sanity_check/"+name[j][:7]+'_intensity.npy', I[j])
            np.save("rec_sanity_check/"+name[j][:7]+'_phase.npy', phase[j])
    print("Sanity check sample generation with RecTophat has ended!")
    
    #-------------------------Sanity check with RecTophat ends---------------------------
    '''
    
    from M290_MachineSimu_GPU.M290_full_scene import M290
    machine = M290(batchsize ,args.beamshape, lightsource_path, device)
    machine = machine.to(device)
    near_field = machine.nearField
    
    model = PImodel()
    model = model.to(device)
    
    checkpoint_name = vis_dir + '/models/' + args.pth_name + '.pth.tar'
    print("=> loading checkpoint '{}'".format(checkpoint_name))
    if args.gpu is None:
        checkpoint = torch.load(checkpoint_name)
    elif torch.cuda.is_available():
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(checkpoint_name, map_location=loc)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(checkpoint_name, checkpoint['epoch']))
    
    for p in model.parameters():
        p.requires_grad = False
    
    utility = UtilityLoss().to(device)
    distribution = uniformity_loss().to(device)
    
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
    
    '''
    # Extreme initialization
    with torch.no_grad():
            machine.zernike_coeffs.fill_(1.5)
    '''
    
    '''
    # Zero initialization
    with torch.no_grad():
            machine.zernike_coeffs.zero_()
    '''
    
    '''
    # Warm start
    coeffs = np.load(vis_dir+'/training_set/zernikes_init.npy')
    coeffs = torch.tensor(coeffs, dtype=machine.zernike_coeffs.dtype, device=device)
    with torch.no_grad():
        machine.zernike_coeffs.copy_(coeffs)
    '''
    
    lower, upper = -1.5, 1.5
    
    #'''
    # Random initialization
    with torch.no_grad():
        machine.zernike_coeffs.uniform_(lower, upper, generator=init_gen)
    #'''
    
    #records = []
    
    #Initial freezing mask, should be all False at the beginning
    mask_freeze = (machine.zernike_coeffs < lower) | (machine.zernike_coeffs > upper)
    
    for j in range(iteration):
        
        imaging_field, phase = machine(near_field)
        I = torch.abs(imaging_field)**2
        max_per_sample = I.amax(dim=(1,2), keepdim=True)
        I = I / max_per_sample
        I = I * 255.0
        I = I.unsqueeze(1)
        
        low, mu, high = model(I)
        #loss = criterion(low, high) + torch.mean((machine.zernike_coeffs)**2)
        #distribution_loss = distribution(machine.zernike_coeffs, bins=30, rag=(-1.5, 1.5))
        
        #'''
        distribution_loss = distribution(machine.zernike_coeffs[0].squeeze(), bins=12, rag=(-1.5, 1.5))
        for i in range(1, batchsize):
            distribution_loss += distribution(machine.zernike_coeffs[i].squeeze(), bins=12, rag=(-1.5, 1.5))
        #'''
        
        utility_loss = utility(low, high)
        
        loss = 10*distribution_loss + utility_loss
        #loss = utility_loss
        
        '''
        # ---- record current state ----
        with torch.no_grad():
            records.append({
                'iter': j,
                'zernike': machine.zernike_coeffs.detach().clone(),
                'uniformity_loss': distribution_loss.item(),
                'utility_loss': utility_loss.item()
            })
        '''
        
        optimizer.zero_grad()
        loss.backward()
        
        # Design_reset0grad
        machine.zernike_coeffs.grad[mask_freeze] = 0.0
        
        optimizer.step()
        scheduler.step()
        
        '''
        with torch.no_grad():    # Design_clamp
            mask_freeze = (machine.zernike_coeffs < lower) | (machine.zernike_coeffs > upper)
            machine.zernike_coeffs.clamp_(min=lower+0.01, max=upper-0.01)

            #if mask_freeze.any():
                #print(f"⚠️  Some Zernike coefficients reached limit and are frozen: {mask_freeze.nonzero(as_tuple=True)}")
        '''
        
        #'''
        with torch.no_grad():    # Design_reset
            mask_freeze = (machine.zernike_coeffs < lower) | (machine.zernike_coeffs > upper)

            if mask_freeze.any():
                temp = torch.empty_like(machine.zernike_coeffs)
                temp.uniform_(lower, upper, generator=reset_gen)
                machine.zernike_coeffs[mask_freeze] = temp[mask_freeze]
        #'''
        
        print(f"Sampling round {args.round_sampling} Iteration {j}    Utility loss: {utility_loss.item():.5f}    Distribution loss: {distribution_loss.item():.5f}")
        #print(loss.item())
    
    '''
    records_sorted = sorted(records, key=lambda x: x['uniformity_loss'])
    top5 = records_sorted[:5]
    best_record = min(top5, key=lambda x: x['utility_loss'])
    
    print(
    f"Iter {best_record['iter']}\n"
    f"Uniformity loss : {best_record['uniformity_loss']:.6f}\n"
    f"Utility loss    : {best_record['utility_loss']:.6f}")
    
    with torch.no_grad():
        machine.zernike_coeffs.copy_(best_record['zernike'])
    '''
    
    imaging_field, phase = machine(near_field)
    
    I = torch.abs(imaging_field)**2
    max_per_sample = I.amax(dim=(1,2), keepdim=True)
    I = I / max_per_sample
    I = I * 255.0
    
    I_ = I.unsqueeze(1)
    low, mu, high = model(I_)
    
    if(batchsize>1):
        I = I.squeeze()
    if(batchsize==1):
        I = I.squeeze(1)
    I = I.cpu().detach().numpy()
    
    phase = CircularAperture(phase, machine.apertureRadius, machine.gridSize)
    phase = phase[: , machine.start_idx:machine.end_idx, machine.start_idx:machine.end_idx]
    phase = phase.detach().cpu().numpy()
    
    z = machine.zernike_coeffs.squeeze()
    if(batchsize==1):
        z = z.unsqueeze(0)
    z = z.cpu().detach().numpy()
    
    uncertainty = high-low
    uncertainty = torch.clamp(uncertainty, min = 0.0)
    
    if(batchsize>1):
        uncertainty = uncertainty.squeeze()
    if(batchsize==1):
        uncertainty = uncertainty.squeeze(1)
    uncertainty = uncertainty.cpu().detach().numpy()
    
    for j in range(len(I)):
        mpimg.imsave(vis_dir+'/training_set/intensity/img/'+'intensity_'+ str(args.round_sampling) + '_' + str(j) +'.png', -I[j], cmap='Greys')
        np.save(vis_dir+'/training_set/intensity/npy/'+'intensity_'+ str(args.round_sampling) + '_' + str(j) +'.npy', I[j])
        mpimg.imsave(vis_dir+'/training_set/phase/img/'+'phase_'+ str(args.round_sampling) + '_' + str(j) +'.png', phase[j], cmap='Greys')
        np.save(vis_dir+'/training_set/phase/npy/'+'phase_'+ str(args.round_sampling) + '_' + str(j) +'.npy', phase[j])
        np.save(vis_dir+'/training_set/zernikes/'+'zernikes_'+ str(args.round_sampling) + '_' + str(j) +'.npy', z[j].squeeze())
        mpimg.imsave(vis_dir+'/latest_uncertainty/'+'uncertainty_'+ str(args.round_sampling) + '_' + str(j) +'.png', -uncertainty[j], cmap='Greys')
        
    #coeffs_np = machine.zernike_coeffs.detach().cpu().numpy()
    #np.save(vis_dir+'/training_set/zernikes_init.npy', coeffs_np)
    
    '''
    z_coeffs = machine.zernike_coeffs.squeeze().cpu().detach().numpy()

    bins = 30
    range_hist = (-1.5, 1.5)

    hist, bin_edges = np.histogram(z_coeffs, bins=bins, range=range_hist, density=True)

    np.save(vis_dir + f'/training_set/zernikes_hist_round{args.round_sampling}.npy', hist)

    plt.figure(figsize=(6,4))
    plt.bar(bin_edges[:-1], hist, width=(bin_edges[1]-bin_edges[0]), align='edge', color='skyblue', edgecolor='black')
    plt.xlabel('Zernike coefficient value')
    plt.ylabel('Density')
    plt.title(f'Zernike Coefficient Histogram - Round {args.round_sampling}')
    plt.xlim(range_hist)
    plt.tight_layout()
    plt.savefig(vis_dir + f'/training_set/zernikes_hist_round{args.round_sampling}.png')
    plt.close()
    '''