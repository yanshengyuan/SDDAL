import numpy as np
import os
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import random
import re

def extract_indices(filename):
    # extract numbers from intensity_a_b.npy
    nums = re.findall(r'\d+', filename)
    return int(nums[0]), int(nums[1])

##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, num_training_samples):
        super(DataLoaderTrain, self).__init__()
        
        gt_dir = os.path.join('training_set', 'phase', 'npy')
        input_dir = os.path.join('training_set', 'intensity', 'npy')
        
        I_files = os.listdir(os.path.join(rgb_dir, input_dir))
        Phi_files = os.listdir(os.path.join(rgb_dir, gt_dir))
        
        I_files = sorted(I_files, key=extract_indices)
        Phi_files = sorted(Phi_files, key=extract_indices)
        
        self.I_filenames = [os.path.join(rgb_dir, input_dir, x) for x in I_files]
        self.Phi_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in Phi_files]
        
        if(num_training_samples==0):
            self.tar_size = len(self.I_filenames)
        else:
            self.tar_size = num_training_samples

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        I=np.load(self.I_filenames[index]).astype(np.float32)
        Phi=np.load(self.Phi_filenames[index]).astype(np.float32)
        I=torch.tensor(I)
        Phi=torch.tensor(Phi)
        I=I.unsqueeze(0)
        Phi=Phi.unsqueeze(0)
        fname=self.I_filenames[index]
        #print(self.I_filenames[index])
        #print(self.Phi_filenames[index])

        return I, Phi, fname

##################################################################################################
class DataLoaderReproduce(Dataset):
    def __init__(self, rgb_dir):
        super(DataLoaderReproduce, self).__init__()
        
        input_dir = os.path.join('training_set', 'zernikes')
        
        z_files = os.listdir(os.path.join(rgb_dir, input_dir))
        
        z_files = sorted(z_files, key=extract_indices)
        
        self.z_filenames = [os.path.join(rgb_dir, input_dir, x) for x in z_files]
        
        self.tar_size = len(self.z_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        z=np.load(self.z_filenames[index])
        z=torch.tensor(z)
        zname=self.z_filenames[index]
        #print(self.I_filenames[index])
        #print(self.Phi_filenames[index])

        return z, zname

##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir):
        super(DataLoaderVal, self).__init__()
        
        gt_dir = os.path.join('test_set', 'phase', 'npy')
        input_dir = os.path.join('test_set', 'intensity', 'npy')
        
        I_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        self.Phi_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        
        self.I_filenames = [os.path.join(rgb_dir, input_dir, x) for x in I_files]
        self.Phi_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in self.Phi_files]
        self.tar_size = len(self.I_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        I=np.load(self.I_filenames[index]).astype(np.float32)
        Phi=np.load(self.Phi_filenames[index]).astype(np.float32)
        I=torch.tensor(I)
        Phi=torch.tensor(Phi)
        I=I.unsqueeze(0)
        Phi=Phi.unsqueeze(0)
        name=self.Phi_files[index]

        return I, Phi




    
##################################################################################################
class DataLoaderTrainCPU(Dataset):
    def __init__(self, rgb_dir):
        super(DataLoaderTrain, self).__init__()
        
        gt_dir = os.path.join('training_set', 'phase', 'npy')
        input_dir = os.path.join('training_set', 'intensity', 'npy')
        
        I_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        Phi_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        
        self.I_filenames = [os.path.join(rgb_dir, input_dir, x) for x in I_files]
        self.Phi_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in Phi_files]
        self.tar_size = len(self.I_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        I=np.load(self.I_filenames[index]).astype(np.float32)
        Phi=np.load(self.Phi_filenames[index]).astype(np.float32)

        return I, Phi
    
class DataLoaderValCPU(Dataset):
    def __init__(self, rgb_dir):
        super(DataLoaderValCPU, self).__init__()
        
        gt_dir = os.path.join('test_set', 'phase', 'npy')
        input_dir = os.path.join('test_set', 'intensity', 'npy')
        
        I_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        self.Phi_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        
        self.I_filenames = [os.path.join(rgb_dir, input_dir, x) for x in I_files]
        self.Phi_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in self.Phi_files]
        self.tar_size = len(self.I_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        I=np.load(self.I_filenames[index]).astype(np.float32)
        Phi=np.load(self.Phi_filenames[index]).astype(np.float32)
        name=self.Phi_files[index]

        return I, Phi
##################################################################################################