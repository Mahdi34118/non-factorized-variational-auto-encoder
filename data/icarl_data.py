import os

import scipy
import numpy as np

import torch
from torch.utils.data import Dataset

class ICARLDataset(Dataset):
    def __init__(self, root, data_postfix, max_num_data=15000):
        self.root = root 

        self.x = np.load(os.path.join(self.root,f"x{data_postfix}.npy"))
        self.u = np.load(os.path.join(self.root, f"u{data_postfix}.npy"))
        self.s = np.load(os.path.join(self.root, f"s{data_postfix}.npy"))

        assert self.x.shape[0]==self.u.shape[0], "Inconsistency in number of data in x.npy and u.npy"
        assert self.x.shape[0]==self.s.shape[0], "Inconsistency in number of data in x.npy and s.npy"
        
        self.x = self.x.reshape([self.x.shape[0]*self.x.shape[1], self.x.shape[2]])
        self.u = self.u.reshape([self.u.shape[0]*self.u.shape[1], self.u.shape[2]])
        self.s = self.s.reshape([self.s.shape[0]*self.s.shape[1], self.s.shape[2]])

        indecies = np.arange(self.x.shape[0])
        np.random.shuffle(indecies)
        if max_num_data!=None and max_num_data < self.x.shape[0]:
            indecies = indecies[:max_num_data]
        
        self.x = self.x[indecies]
        self.u = self.u[indecies]
        self.s = self.s[indecies]

    def get_dims(self):
        return self.x.shape[1], self.u.shape[1], self.s.shape[1]

    def __getitem__(self, index):
        return torch.tensor(self.x[index]), torch.tensor(self.u[index]), torch.tensor(self.s[index])
    
    def __len__(self):
        return self.x.shape[0]