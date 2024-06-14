#!/usr/bin/env python3
from torch.utils.data import Dataset
import torch
import numpy as np

def parse(strn):
    A = ['vhigh', 'high', 'med', 'low']
    B = ['small', 'med', 'big']
    C = ['unacc', 'acc', 'good', 'vgood']
    if strn in A:
        return A.index(strn) + 1
    if strn in B:
        return B.index(strn) + 1
    if strn in C:
        return C.index(strn) + 1       
    if strn == '5more':
        return 5
    if strn == 'more':
        return 10
    else:
        return int(strn)

class CarEvaluationDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        #Assume datasets are going in
        self.features = features.applymap(parse).values.astype(float) #np.array(parse(features.values[:, 1], features.), dtype=float)
        self.labels = labels.applymap(parse).values.astype(float).flatten() #np.array(labels.values[:, 1], dtype=int)
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        # feature, label = self.features[index], self.labels[index]

        # if self.transform is not None:
        #     feature = self.transform(feature)

        # return feature, label
        return torch.tensor(self.features.values[index], dtype=torch.float32), torch.tensor(self.labels.values[index], dtype=torch.long)