import matplotlib.pyplot as plt
import math, torch, torchvision, os, glob, time, random, io, sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn.metrics

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

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)


# # 66.66%
# trainDataset = Cifar10Dataset(data_train, targets_train, transform=cifar10.transform)
# trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)

# # 16.66%
# validDataset = Cifar10Dataset(data_valid, targets_valid, transform=cifar10.transform)
# validLoader = DataLoader(validDataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

# # 16.66%
# testDataset = Cifar10Dataset(cifar10_test.data, cifar10_test.targets, transform=cifar10_test.transform)
# testLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

train_dataset = CarEvaluationDataset(X_train, y_train)
valid_dataset = CarEvaluationDataset(X_val, y_val)
test_dataset = CarEvaluationDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

class CarNet(nn.Module):
    def __init__(self):
        super(CarNet, self).__init__()
        self.inp = nn.Linear(6, 16)
        self.conv1 = nn.Conv2d(6, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(576, 288)
        self.fc2 = nn.Linear(288, 72)
        self.fc3 = nn.Linear(72, 10)
        self.outp = nn.Linear(32, 4)  # 4 output classes

    def forward(self, x):
        x = F.relu(self.inp(x))
        x = F.relu(self.pool(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.outp(x)
        return x


if __name__=='__main__':
    model = CarNet().to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 6e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9) # , weight_decay=1e-5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=0, threshold_mode='abs') 
    early_stopper = EarlyStopper(patience=2, min_change=0, mode='min')
    batch_print_step = 500
    epochs = 50

    best_train_loss = float('inf')
    best_val_loss = float('inf')

    train_loss_history = []
    val_loss_history = []
    train_f1_history = []
    val_f1_history = []


    for epoch in range(epochs):
        # index 0 is for training, index 1 for validation
        epoch_loss = [0, 0]
        accuracy = [0, 0]
        f1_score = [0, 0]
        epochtime = time.time()
        
        for dataset, dataloader, isTraining in [(train_dataset, train_loader, True), (valid_dataset, valid_loader, False)]:
            correct = 0
            running_loss = 0.0
            all_input_labels = []
            all_predicted_labels = []

            if isTraining:
                model.train()
            else:
                model.eval()

            for i, data in enumerate(dataloader, 0):        
                # get the inputs
                text_data, labels = data
                text_data = text_data.to(device)
                labels = labels.to(device)

                if isTraining:
                    # zero the parameter gradients
                    optimizer.zero_grad()

                # forward 
                outputs = model(text_data)
                _, predicted = torch.max(outputs, 1)
                
                loss = criterion(outputs, labels)

                if isTraining:
                    # backward + optimize
                    loss.backward()
                    optimizer.step()

                # print statistics
                running_loss += loss.item() * text_data.size(0)
                correct += torch.sum(predicted == labels.data)
                all_predicted_labels.append(predicted.cpu())
                all_input_labels.append(labels.cpu())
                
                if i % batch_print_step == batch_print_step-1:
                    print("\r", f'{"TRAINING" if isTraining else "VALIDATION"} PROGRESS [epoch: {epoch + 1}, {100*batch_size*(i + 1)/len(dataset):.3f}%, {math.floor(time.time() - epochtime):3d}s] loss: {running_loss / len(dataset):.3f}', end="")
                    timestart = time.time()
            
            index = 0 if isTraining else 1
            epoch_loss[index] = running_loss / len(dataset)
            accuracy[index] = correct.double() / len(dataset)
            all_input_labels = np.concatenate(all_input_labels)
            all_predicted_labels = np.concatenate(all_predicted_labels)
            f1_score[index] = sklearn.metrics.f1_score(all_input_labels, all_predicted_labels, average='macro')

        print("\r" + f'EPOCH: [{epoch + 1:2d}/{epochs}, time: {math.floor(time.time() - epochtime)}s]   TRAIN: [loss: {epoch_loss[0]:.3f}, acc: {accuracy[0]:.3f}, f1: {f1_score[0]:.3f}]   VAL: [loss: {epoch_loss[1]:.3f}, acc: {accuracy[1]:.3f}, f1: {f1_score[1]:.3f}]   learning_rate: {lr}') 
        train_loss_history.append(epoch_loss[0])
        val_loss_history.append(epoch_loss[1])
        train_f1_history.append(f1_score[0])
        val_f1_history.append(f1_score[1])

        if early_stopper.stop_early(epoch_loss[1]): 
            print("stopping early")
            break

        scheduler.step(epoch_loss[1])
        lr = scheduler.get_last_lr()

    print('done')