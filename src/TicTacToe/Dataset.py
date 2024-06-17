#!/usr/bin/env python3
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder

# def parse(strn):
#     A = ['vhigh', 'high', 'med', 'low']
#     B = ['small', 'med', 'big']
#     C = ['unacc', 'acc', 'good', 'vgood']
#     if strn in A:
#         return A.index(strn) + 1
#     if strn in B:
#         return B.index(strn) + 1
#     if strn in C:
#         return C.index(strn) + 1       
#     if strn == '5more':
#         return 5
#     if strn == 'more':
#         return 10
#     else:
#         return int(strn)

def parse(strn):
    if strn < 200:
        return 0
    if strn < 600:
        return 1
    if strn < 1000:
        return 2
    return 3

class TicTacToaDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        #Assume datasets are going in
        encoder = LabelEncoder()
        features['top-left-square'] = encoder.fit_transform(features['top-left-square'])
        features['top-middle-square'] = encoder.fit_transform(features['top-middle-square'])
        features['top-right-square'] = encoder.fit_transform(features['top-right-square'])
        features['middle-left-square'] = encoder.fit_transform(features['middle-left-square'])
        features['middle-middle-square'] = encoder.fit_transform(features['middle-middle-square'])
        features['middle-right-square'] = encoder.fit_transform(features['middle-right-square'])
        features['bottom-left-square'] = encoder.fit_transform(features['bottom-left-square'])
        features['bottom-middle-square'] = encoder.fit_transform(features['bottom-middle-square'])
        features['bottom-right-square'] = encoder.fit_transform(features['bottom-right-square'])
        labels = encoder.fit_transform(labels)
        self.features = features.values.astype(float) # features.applymap(parse).values.astype(float) #np.array(parse(features.values[:, 1], features.), dtype=float)
        self.labels = labels.astype(float).flatten() # labels.applymap(parse).values.astype(float).flatten() #np.array(labels.values[:, 1], dtype=int)
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        # feature, label = self.features[index], self.labels[index]

        # if self.transform is not None:
        #     feature = self.transform(feature)

        # return feature, label
        return torch.tensor(self.features.values[index], dtype=torch.float32), torch.tensor(self.labels.values[index], dtype=torch.long)