"""
Shengyuan Yan, TU/e, Weeze Flughalfen, Deutsland, 16:52pm, 01/02/2026
"""

import numpy as np
import torch
from torch import nn
from torch.fft import fft2 as _fft2
from torch.fft import ifft2 as _ifft2
from scipy.special import fresnel as _fresnel


def Fresnel_Direct_Integration_as_Convolution_GPU(Fin, gridsize, wavelength, z):
    B = Fin.shape[0] if Fin.ndim == 3 else 1
    
    device = Fin.device
    
    size = gridsize
    lam = wavelength
    
    N = Fin.size(-1)
    No2 = int(N/2)
    dx = size / (N-1)
    
    kz = 2.*3.141592654/lam * z
    kz = torch.tensor(kz, dtype=torch.float32, device=device)
    
    cokz = torch.cos(kz)
    sikz = torch.sin(kz)
    
    in_outF = torch.zeros((B, 2*N, 2*N), dtype = torch.complex64, device=device)
    in_outK = torch.zeros((2*N, 2*N), dtype = torch.complex64, device=device)
    
    ii2N = torch.ones(2*N, dtype=torch.float32, device=device)
    ii2N[1::2] = -1
    iiij2N = torch.ger(ii2N, ii2N)
    iiij2No2 = iiij2N[:2*No2,:2*No2]
    iiijN = iiij2N[:N, :N]

    RR = np.sqrt(1/(2*lam*z))*dx*2
    io = np.arange(0, (2*No2)+1)
    R1 = RR*(io - No2)
    fs, fc = _fresnel(R1)
    fss = np.outer(fs, fs)
    fsc = np.outer(fs, fc)
    fcs = np.outer(fc, fs)
    fcc = np.outer(fc, fc)
    
    temp_re = (fsc[1:, 1:]
               + fcs[1:, 1:])
    temp_re -= fsc[:-1, 1:]
    temp_re -= fcs[:-1, 1:]
    temp_re -= fsc[1:, :-1]
    temp_re -= fcs[1:, :-1]
    temp_re += fsc[:-1, :-1]
    temp_re += fcs[:-1, :-1]
    
    temp_im = (-fcc[1:, 1:]
               + fss[1:, 1:])
    temp_im += fcc[:-1, 1:]
    temp_im -= fss[:-1, 1:]
    temp_im += fcc[1:, :-1]
    temp_im -= fss[1:, :-1]
    temp_im -= fcc[:-1, :-1]
    temp_im += fss[:-1, :-1]
    
    temp_K = 1j * temp_im
    temp_K += temp_re
    
    if Fin.is_cuda:
        temp_K = torch.from_numpy(temp_K)
        temp_K = temp_K.to(device)
    
    temp_K *= iiij2No2
    temp_K *= 0.5
    in_outK[(N-No2):(N+No2), (N-No2):(N+No2)] = temp_K
    
    in_outF[:, (N-No2):(N+No2), (N-No2):(N+No2)] \
        = Fin[:, (N-2*No2):N,(N-2*No2):N]
    
    in_outF[:, (N-No2):(N+No2), (N-No2):(N+No2)] *= iiij2No2
    
    in_outK = _fft2(in_outK, dim=(-2, -1))
    in_outF = _fft2(in_outF, dim=(-2, -1))
    
    in_outF *= in_outK
    
    in_outF *= iiij2N
    in_outF = _ifft2(in_outF, dim=(-2, -1))
    
    Ftemp = (in_outF[:, No2:N+No2, No2:N+No2]
             - in_outF[:, No2-1:N+No2-1, No2:N+No2])
    Ftemp += in_outF[:, No2-1:N+No2-1, No2-1:N+No2-1]
    Ftemp -= in_outF[:, No2:N+No2, No2-1:N+No2-1]
    
    comp = cokz + 1j * sikz
    comp = comp.to(torch.complex64)
    
    Ftemp *= 0.25 * comp
    
    Ftemp *= iiijN
    
    return Ftemp

class Fresnel_propagator(nn.Module):
    def __init__(self, wavelength):
        super(Fresnel_propagator, self).__init__()
        self.wavelength = wavelength

    def forward(self, Fin, z, gridsize):
        
        Fout = Fresnel_Direct_Integration_as_Convolution_GPU(Fin, gridsize, self.wavelength, z)

        return Fout