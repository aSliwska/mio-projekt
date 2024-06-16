import matplotlib.pyplot as plt
import math, torch, torchvision, os, glob, time, random, io, sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn.metrics
import seaborn as sns
import sklearn
import pandas as pd

from torchvision.ops import MLP
from PIL import Image
from torchvision.transforms import v2
from torchvision.datasets import CIFAR10
from sklearn.metrics import confusion_matrix 
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from EStopper import EarlyStopper
from Dataset import ukmDataset


batch_size = 32

##
## Fetching the dataset
##

from ucimlrepo import fetch_ucirepo 

rice = fetch_ucirepo(id=257)  # not rice
  
# data (as pandas dataframes) 
X = rice.data.features 
y = rice.data.targets

##
## Setting up the device for CUDA
##

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("device type:", device)
print("current device:", torch.cuda.get_device_name(torch.cuda.current_device()))

torch.multiprocessing.set_sharing_strategy('file_system')

##
## Setting up the datasets
##

#data_train, data_valid, targets_train, targets_valid = train_test_split(cifar10.data, cifar10.targets, test_size=0.2, stratify=cifar10.targets)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42, stratify=y_temp)

train_dataset = ukmDataset(X_train, y_train)
valid_dataset = ukmDataset(X_val, y_val)
test_dataset = ukmDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

##
## Main loop for training and testing
##
if __name__=='__main__':
    
    model = MLP(in_channels=5, hidden_channels=[90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90], activation_layer=nn.ReLU)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    epochs = 1200
    epoch_print_step = 100

    loss_train = []
    loss_validate = []
    accuracy_train = []
    accuracy_validate = []
    f1_train = []
    f1_validate = []

    for epoch in range(epochs):
        
        epoch_loss = [0, 0]
        accuracy = [0, 0]
        f1_score = [0, 0]

        # Training loop
        for dataset, is_τraining in [(train_dataset, True), (valid_dataset, False)]:
            correct = 0
            running_loss = 0.0
            all_input_labels = []
            all_predicted_labels = []

            if is_τraining:
                model.train()
            else:
                model.eval()

            if is_τraining:
                optimizer.zero_grad()

            data = torch.from_numpy(dataset.features).type(torch.FloatTensor)
            labels = torch.from_numpy(dataset.labels).type(torch.LongTensor)

            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            loss = criterion(outputs, labels)

            if is_τraining:
                loss.backward()
                optimizer.step()

            # Statistics for the current epoch
            running_loss += loss.item() * data.size(0)
            correct += torch.sum(predicted == labels.data)
            all_predicted_labels.append(predicted)
            all_input_labels.append(labels)
            
            index = 0 if is_τraining else 1
            epoch_loss[index] = running_loss / len(dataset)
            accuracy[index] = correct.double() / len(dataset)
            all_input_labels = np.concatenate(all_input_labels)
            all_predicted_labels = np.concatenate(all_predicted_labels)
            f1_score[index] = sklearn.metrics.f1_score(all_input_labels, all_predicted_labels, average='macro')

        if epoch % epoch_print_step == epoch_print_step-1:
            print("\r" + f'EPOCH: [{epoch + 1:2d}/{epochs}]   TRAIN: [loss: {epoch_loss[0]:.3f}, acc: {accuracy[0]:.3f}, f1: {f1_score[0]:.3f}]   VAL: [loss: {epoch_loss[1]:.3f}, acc: {accuracy[1]:.3f}, f1: {f1_score[1]:.3f}]') 
        
        loss_train.append(epoch_loss[0])
        loss_validate.append(epoch_loss[1])
        accuracy_train.append(accuracy[0])
        accuracy_validate.append(accuracy[1])
        f1_train.append(f1_score[0])
        f1_validate.append(f1_score[1])

    print('done')


    ##
    ## Evaluating the model
    ##
    model.eval()

    test_running_loss = 0
    correct = 0
    all_predicted_labels = []
    all_input_labels = []

    data = torch.from_numpy(dataset.features).type(torch.FloatTensor)
    labels = torch.from_numpy(dataset.labels).type(torch.LongTensor)

    outputs = model(data)
    _, predicted = torch.max(outputs, 1)
    loss = criterion(outputs, labels)

    test_running_loss += loss.item() * data.size(0)
    correct += torch.sum(predicted == labels.data)
    all_predicted_labels.append(predicted)
    all_input_labels.append(labels)

    final_loss = test_running_loss / len(test_dataset)
    accuracy = correct.double() / len(test_dataset)
    all_input_labels = np.concatenate(all_input_labels)
    all_predicted_labels = np.concatenate(all_predicted_labels)
    f1_score = sklearn.metrics.f1_score(all_input_labels, all_predicted_labels, average='macro')

    ##
    ## Visualising the results of training and evaluation
    ##
    print(f'[TEST] loss: {final_loss:.3f}, accuracy: {accuracy:.3f}, f1_score: {f1_score:.3f}')

    print(sns.heatmap(confusion_matrix(all_input_labels, all_predicted_labels, labels=[i for i in range(4)]), annot=True, cmap="crest", fmt='g'))

    plt.figure(figsize = (15, 15))

    plt.subplot(2, 2, 1)
    plt.title('loss history')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xlim(0, len(loss_train))
    plt.ylim(0, max(max(loss_train), max(loss_validate)))
    plt.plot(loss_train)
    plt.plot(loss_validate)
    plt.legend(['training', 'validation'])

    plt.subplot(2, 2, 2)
    plt.title('accuracy history')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.xlim(0, len(accuracy_train))
    plt.ylim(0, 1)
    plt.plot(accuracy_train)
    plt.plot(accuracy_validate)
    plt.legend(['training', 'validation'])

    plt.subplot(2, 2, 3)
    plt.title('f1 score history')
    plt.xlabel('epoch')
    plt.ylabel('f1 score')
    plt.xlim(0, len(f1_train))  
    plt.ylim(0, 1)
    plt.plot(f1_train)
    plt.plot(f1_validate)
    plt.legend(['training', 'validation'])

    plt.show()