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
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--step_size', default=50, type=int,
                    help='step size (default: 50)')
parser.add_argument('--pth_name', default='', type=str)
parser.add_argument('--round_sampling', type=int)
parser.add_argument('--vis_path', default='', type=str)
parser.add_argument('--init_seed', default=None, type=int,
                    help='Random seed used only for initializing zernike coefficients.')

iteration = 100


class UtilityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, low, high):
        interval = high - low
        interval = torch.clamp(interval, min=0.0)
        mean_interval = torch.mean(interval)

        loss = -mean_interval
        return loss

'''
class uniformity_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, bins=20, rag=(-1.5, 1.5)):
        hist = torch.histc(z, bins=bins, min=rag[0], max=rag[1])
        hist = hist / torch.sum(hist)
        uniform = torch.ones_like(hist) / bins
        loss = torch.sum((hist - uniform) ** 2)
        return loss
'''

class MMDUniformLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def rbf_kernel(self, x, y, sigma):
        """
        x: [N]
        y: [M]
        return: [N, M]
        """
        x = x.view(-1, 1)
        y = y.view(-1, 1)
        dist2 = (x - y.T) ** 2
        return torch.exp(-dist2 / (2.0 * sigma * sigma))

    def forward(self, z, lower=-1.5, upper=1.5, num_ref=128, sigmas=(0.15, 0.3, 0.6)):
    #Computes MMD^2 between z and Uniform(lower, upper).
    
        ref = lower + (upper - lower) * torch.rand(
            num_ref, device=z.device, dtype=z.dtype
        )

        mmd2 = 0.0
        for sigma in sigmas:
            Kxx = self.rbf_kernel(z, z, sigma)
            Kyy = self.rbf_kernel(ref, ref, sigma)
            Kxy = self.rbf_kernel(z, ref, sigma)
            mmd2 = mmd2 + Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()

        mmd2 = mmd2 / len(sigmas)
        return mmd2


if __name__ == '__main__':

    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        init_gen = torch.Generator(device=device)
    else:
        init_gen = torch.Generator()
    init_gen.manual_seed(args.init_seed)

    vis_dir = args.vis_path
    beamshape = args.beamshape
    plane = args.caustic_plane
    batchsize = args.batch_size

    lightsource_path = 'M290_MachineSimu_GPU/lightsource_full_scene.npy'

    '''
    #-------------------------Sanity check with RecTophat starts---------------------------

    from M290_MachineSimu_GPU.M290_sanity_check import M290
    machine = M290(1, args.beamshape, lightsource_path, device)
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
        Zernike_Coeffs = Z.to(device, non_blocking=True).view(batchsize, 12, 1, 1)

        # Convert physical zernike coeffs in [-1.5, 1.5] to raw_zernike
        with torch.no_grad():
            eps = 1e-6
            x = (Zernike_Coeffs / 1.5).clamp(-1 + eps, 1 - eps)
            machine.raw_zernike.copy_(torch.atanh(x))

        imaging_field, phase = machine(near_field)
        I = torch.abs(imaging_field)**2
        max_per_sample = I.amax(dim=(1, 2), keepdim=True)
        I = I / max_per_sample
        I = I * 255.0
        I = I.detach().cpu().numpy()

        phase = CircularAperture(phase, machine.apertureRadius, machine.gridSize)
        phase = phase[:, machine.start_idx:machine.end_idx, machine.start_idx:machine.end_idx]
        phase = phase.detach().cpu().numpy()

        for j in range(len(I)):
            np.save("rec_sanity_check/" + name[j][:7] + '_intensity.npy', I[j])
            np.save("rec_sanity_check/" + name[j][:7] + '_phase.npy', phase[j])
    print("Sanity check sample generation with RecTophat has ended!")

    #-------------------------Sanity check with RecTophat ends---------------------------
    '''

    from M290_MachineSimu_GPU.M290_full_scene import M290
    machine = M290(batchsize, args.beamshape, lightsource_path, device)
    machine = machine.to(device)
    near_field = machine.nearField

    model = PImodel()
    model = model.to(device)

    checkpoint_name = vis_dir + '/models/' + args.pth_name + '.pth.tar'
    print("=> loading checkpoint '{}'".format(checkpoint_name))
    if args.gpu is None:
        checkpoint = torch.load(checkpoint_name)
    elif torch.cuda.is_available():
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(checkpoint_name, map_location=loc)
    else:
        checkpoint = torch.load(checkpoint_name, map_location='cpu')

    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(checkpoint_name, checkpoint['epoch']))

    for p in model.parameters():
        p.requires_grad = False

    utility = UtilityLoss().to(device)
    #distribution = uniformity_loss().to(device)
    distribution = MMDUniformLoss().to(device)

    if args.lr > 0:
        lr = args.lr

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam([machine.raw_zernike], lr=lr,
                                     weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([machine.raw_zernike], lr=lr, momentum=0.9,
                                    weight_decay=1e-4)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    if args.step_size > 0:
        step_size = args.step_size
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.5)

    lower, upper = -1.5, 1.5
    bound = 1.5
    eps = 1e-6

    '''
    # Extreme initialization in physical zernike space
    with torch.no_grad():
        z0 = torch.full_like(machine.raw_zernike, 1.5)
        x = (z0 / bound).clamp(-1 + eps, 1 - eps)
        machine.raw_zernike.copy_(torch.atanh(x))
    '''

    '''
    # Zero initialization in physical zernike space
    with torch.no_grad():
        z0 = torch.zeros_like(machine.raw_zernike)
        x = (z0 / bound).clamp(-1 + eps, 1 - eps)
        machine.raw_zernike.copy_(torch.atanh(x))
    '''

    '''
    # Warm start: coeffs file should contain physical zernike coeffs in [-1.5, 1.5]
    coeffs = np.load(vis_dir + '/training_set/zernikes_init.npy')
    coeffs = torch.tensor(coeffs, dtype=machine.raw_zernike.dtype, device=device)
    if coeffs.dim() == 2:
        coeffs = coeffs.unsqueeze(-1).unsqueeze(-1)
    x = (coeffs / bound).clamp(-1 + eps, 1 - eps)
    with torch.no_grad():
        machine.raw_zernike.copy_(torch.atanh(x))
    '''

    #'''
    # Random initialization in physical zernike space, then convert to raw_zernike
    with torch.no_grad():
        z0 = torch.empty_like(machine.raw_zernike)
        z0.uniform_(lower, upper, generator=init_gen)

        x = (z0 / bound).clamp(-1 + eps, 1 - eps)
        machine.raw_zernike.copy_(torch.atanh(x))
    #'''

    with torch.no_grad():
        z_init = machine.get_zernike_coeffs().squeeze()
    if batchsize == 1:
        z_init = z_init.unsqueeze(0)
    z_init = z_init.cpu().detach().numpy()

    records = []

    for j in range(iteration):

        imaging_field, phase = machine(near_field)
        I = torch.abs(imaging_field)**2
        max_per_sample = I.amax(dim=(1, 2), keepdim=True)
        I = I / max_per_sample
        I = I * 255.0
        I = I.unsqueeze(1)

        low, mu, high = model(I)

        zernike_coeffs = machine.get_zernike_coeffs()
        
        '''
        distribution_loss = distribution(zernike_coeffs[0].squeeze(), bins=12, rag=(-1.5, 1.5))
        for i in range(1, batchsize):
            distribution_loss += distribution(zernike_coeffs[i].squeeze(), bins=12, rag=(-1.5, 1.5))
        '''
        
        #'''
        distribution_loss = distribution(zernike_coeffs[0].squeeze())
        for i in range(1, batchsize):
            distribution_loss += distribution(zernike_coeffs[i].squeeze())
        #'''
            
        utility_loss = utility(low, high)

        loss = 1 * distribution_loss + utility_loss
        #loss = utility_loss

        #'''
        # ---- record current state ----
        with torch.no_grad():
            records.append({
                'iter': j,
                'raw_zernike': machine.raw_zernike.detach().clone(),
                'uniformity_loss': distribution_loss.item(),
                'utility_loss': utility_loss.item(),
                'total_loss': loss.item()
            })
        #'''

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f"Sampling round {args.round_sampling} Iteration {j}    Utility loss: {utility_loss.item():.5f}    Distribution loss: {distribution_loss.item():.5f}")

    ''' # optimize with uniform loss or utility loss as priority over the other
    records_sorted = sorted(records, key=lambda x: x['uniformity_loss'])
    top5 = records_sorted[:5]
    best_record = min(top5, key=lambda x: x['utility_loss'])
    
    print(
    f"Iter {best_record['iter']}\n"
    f"Uniformity loss : {best_record['uniformity_loss']:.6f}\n"
    f"Utility loss    : {best_record['utility_loss']:.6f}\n"
    f"Total loss    : {best_record['total_loss']:.6f}"
    )
    
    with torch.no_grad():
        machine.raw_zernike.copy_(best_record['raw_zernike'])
    '''
    
    #''' # optimize for total loss
    best_record = min(records, key=lambda x: x['total_loss'])
    
    print(
    f"Iter {best_record['iter']}\n"
    f"Uniformity loss : {best_record['uniformity_loss']:.6f}\n"
    f"Utility loss    : {best_record['utility_loss']:.6f}\n"
    f"Total loss    : {best_record['total_loss']:.6f}"
    )
    
    with torch.no_grad():
        machine.raw_zernike.copy_(best_record['raw_zernike'])
    #'''

    imaging_field, phase = machine(near_field)

    I = torch.abs(imaging_field)**2
    max_per_sample = I.amax(dim=(1, 2), keepdim=True)
    I = I / max_per_sample
    I = I * 255.0

    I_ = I.unsqueeze(1)
    low, mu, high = model(I_)

    if batchsize > 1:
        I = I.squeeze()
    if batchsize == 1:
        I = I.squeeze(1)
    I = I.cpu().detach().numpy()

    phase = CircularAperture(phase, machine.apertureRadius, machine.gridSize)
    phase = phase[:, machine.start_idx:machine.end_idx, machine.start_idx:machine.end_idx]
    phase = phase.detach().cpu().numpy()

    with torch.no_grad():
        z = machine.get_zernike_coeffs().squeeze()
    if batchsize == 1:
        z = z.unsqueeze(0)
    z = z.cpu().detach().numpy()

    uncertainty = high - low
    uncertainty = torch.clamp(uncertainty, min=0.0)

    if batchsize > 1:
        uncertainty = uncertainty.squeeze()
    if batchsize == 1:
        uncertainty = uncertainty.squeeze(1)
    uncertainty = uncertainty.cpu().detach().numpy()

    for j in range(len(I)):
        mpimg.imsave(vis_dir + '/training_set/intensity/img/' + 'intensity_' + str(args.round_sampling) + '_' + str(j) + '.png', -I[j], cmap='Greys')
        np.save(vis_dir + '/training_set/intensity/npy/' + 'intensity_' + str(args.round_sampling) + '_' + str(j) + '.npy', I[j])
        mpimg.imsave(vis_dir + '/training_set/phase/img/' + 'phase_' + str(args.round_sampling) + '_' + str(j) + '.png', phase[j], cmap='Greys')
        np.save(vis_dir + '/training_set/phase/npy/' + 'phase_' + str(args.round_sampling) + '_' + str(j) + '.npy', phase[j])
        np.save(vis_dir + '/training_set/zernikes/' + 'zernikes_' + str(args.round_sampling) + '_' + str(j) + '.npy', z[j].squeeze())
        np.save(vis_dir + '/training_set/init_zernikes/' + 'zernikes_' + str(args.round_sampling) + '_' + str(j) + '.npy', z_init[j].squeeze())
        mpimg.imsave(vis_dir + '/latest_uncertainty/' + 'uncertainty_' + str(args.round_sampling) + '_' + str(j) + '.png', -uncertainty[j], cmap='Greys')

    #coeffs_np = machine.get_zernike_coeffs().detach().cpu().numpy()
    #np.save(vis_dir + '/training_set/zernikes_init.npy', coeffs_np)

    '''
    z_coeffs = machine.get_zernike_coeffs().squeeze().cpu().detach().numpy()

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