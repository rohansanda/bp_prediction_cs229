
import torch
from torch.utils.data import Dataset
from typing import Union
import numpy as np
import pandas as pd

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class segmentsData(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data).to(device)
        self.targets = torch.tensor(targets).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
class SegmentsDataBoth(Dataset):
    def __init__(self, segments, targets_sbp, targets_dbp):
        self.segments = torch.tensor(segments).to(device)
        self.targets_sbp = torch.tensor(targets_sbp).to(device)
        self.targets_dbp = torch.tensor(targets_dbp).to(device)

    def __getitem__(self, index):
        segment = self.segments[index]
        target_sbp = self.targets_sbp[index]
        target_dbp = self.targets_dbp[index]
        return segment, target_sbp, target_dbp

    def __len__(self):
        return len(self.segments)