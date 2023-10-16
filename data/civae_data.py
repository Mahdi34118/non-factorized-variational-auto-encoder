import os

import scipy
import numpy as np

import torch
from torch.utils.data import Dataset

class CIVAEDataset(Dataset):
    def __init__(self, root, data_postfixe, max_num_data=15000):
        self.root = root 

        self.x = np.load(f"{self.root}x_{data_postfixe}.npy")
        self.u = np.load(f"{self.root}u_{data_postfixe}.npy")
        self.s = np.load(f"{self.root}s_{data_postfixe}.npy")

        assert self.x.shape[0]==self.u.shape[0], "Inconsistency in number of data in x.npy and u.npy"
        assert self.x.shape[0]==self.s.shape[0], "Inconsistency in number of data in x.npy and z.npy"

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