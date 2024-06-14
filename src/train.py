import matplotlib.pyplot as plt
import math, torch, torchvision, os, glob, time, random, io, sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn.metrics

from torchvision.ops import MLP
from PIL import Image
from torchvision.transforms import v2
from torchvision.datasets import CIFAR10
from sklearn.metrics import confusion_matrix 
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from EStopper import EarlyStopper
from Dataset import CarEvaluationDataset

# Settings fro IO to use UTF-8 encoding. Otherwise fail on line 37, 40
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')   
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


batch_size = 32

##
## Fetching the dataset
##

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
car_evaluation = fetch_ucirepo(id=19) 
  
# data (as pandas dataframes) 
X = car_evaluation.data.features 
y = car_evaluation.data.targets 
  
# metadata 
#print(car_evaluation.metadata) 
  
# variable information 
#print(car_evaluation.variables) 

#print(X, y)

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

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.111111, random_state=42, stratify=y_temp)


# # 66.66%
# trainDataset = Cifar10Dataset(data_train, targets_train, transform=cifar10.transform)
# trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)

# # 16.66%
# validDataset = Cifar10Dataset(data_valid, targets_valid, transform=cifar10.transform)
# validLoader = DataLoader(validDataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

# # 16.66%
# testDataset = Cifar10Dataset(cifar10_test.data, cifar10_test.targets, transform=cifar10_test.transform)
# testLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

#print(X_train)

train_dataset = CarEvaluationDataset(X_train, y_train)
valid_dataset = CarEvaluationDataset(X_val, y_val)
test_dataset = CarEvaluationDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)


if __name__=='__main__':
    
    model = MLP(in_channels=6, hidden_channels=[100,100,100,100,100], activation_layer=nn.ReLU)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    epochs = 1500
    epoch_print_step = 100

    best_train_loss = float('inf')
    best_val_loss = float('inf')

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    train_f1_history = []
    val_f1_history = []
    timestart = time.time()

    for epoch in range(epochs):
        
        epoch_loss = [0, 0]
        accuracy = [0, 0]
        f1_score = [0, 0]
        epochtime = time.time()

        for dataset, isTraining in [(train_dataset, True), (valid_dataset, False)]:
            correct = 0
            running_loss = 0.0
            all_input_labels = []
            all_predicted_labels = []

            if isTraining:
                model.train()
            else:
                model.eval()

            if isTraining:
                # zero the parameter gradients
                optimizer.zero_grad()

            data = torch.from_numpy(dataset.features).type(torch.FloatTensor)
            labels = torch.from_numpy(dataset.labels).type(torch.LongTensor)

            # forward 
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            loss = criterion(outputs, labels)

            if isTraining:
                # backward + optimize
                loss.backward()
                optimizer.step()

            # print statistics
            running_loss += loss.item() * data.size(0)
            correct += torch.sum(predicted == labels.data)
            all_predicted_labels.append(predicted)
            all_input_labels.append(labels)
            
            index = 0 if isTraining else 1
            epoch_loss[index] = running_loss / len(dataset)
            accuracy[index] = correct.double() / len(dataset)
            all_input_labels = np.concatenate(all_input_labels)
            all_predicted_labels = np.concatenate(all_predicted_labels)
            f1_score[index] = sklearn.metrics.f1_score(all_input_labels, all_predicted_labels, average='macro')

        if epoch % epoch_print_step == epoch_print_step-1:
            print("\r" + f'EPOCH: [{epoch + 1:2d}/{epochs}, {epoch_print_step}_time: {math.floor(time.time() - timestart)}s]   TRAIN: [loss: {epoch_loss[0]:.3f}, acc: {accuracy[0]:.3f}, f1: {f1_score[0]:.3f}]   VAL: [loss: {epoch_loss[1]:.3f}, acc: {accuracy[1]:.3f}, f1: {f1_score[1]:.3f}]') 
            timestart = time.time()
        train_loss_history.append(epoch_loss[0])
        val_loss_history.append(epoch_loss[1])
        train_acc_history.append(accuracy[0])
        val_acc_history.append(accuracy[1])
        train_f1_history.append(f1_score[0])
        val_f1_history.append(f1_score[1])

    print('done')