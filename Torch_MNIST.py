#https://github.com/AquibPy/Pytorch/blob/master/MNIST%20Using%20ANN%20on%20GPU%20with%20Pytorch.ipynb

import torch
import torchvision
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split


dataset = MNIST(root = 'data/',train=True,transform=ToTensor())

val_size = 10000
train_size = len(dataset) - val_size
train_ds,val_ds = random_split(dataset,[train_size,val_size])

print(len(train_ds), len(val_ds))


batch_size = 128
train_loder = DataLoader(train_ds,batch_size,shuffle=True,num_workers=4,pin_memory=True)
val_loader = DataLoader(val_ds,batch_size,shuffle=True,num_workers=4,pin_memory=True)


def accuracy(outputs,labels):
    _,preds = torch.max(outputs,dim = 1) ## _ here max prob will come and we don't require it now
    return torch.tensor(torch.sum(preds == labels).item()/len(preds))

class MnistModel(nn.Module):
    def __init__(self,input_size,hidden_size,out_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,out_size)
    def forward(self,xb):
        xb = xb.view(xb.size(0),-1) ## same as .reshape()
        out = self.linear1(xb)
        out = F.relu(out)
        out = self.linear2(out)
        return out
    
    def training_step(self,batch):
        images,labels = batch
        out = self(images)
        loss = F.cross_entropy(out,labels)
        return loss
    
    def validation_step(self,batch):
        images,labels = batch
        out = self(images)
        loss = F.cross_entropy(out,labels)
        acc = accuracy(out,labels)
        return {'val_loss': loss, 'val_acc': acc}
    
    def validation_epoch_end(self,outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {'val_loss': epoch_loss.item(),'val_acc': epoch_acc.item()}
    def epoch_end(self,epoch,result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch+1, result['val_loss'], result['val_acc']))

input_size = 784
num_classes = 10

model = MnistModel(input_size,hidden_size = 32,out_size = num_classes)

print(torch.cuda.is_available())
print('--------------------------------------')


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_default_device()
print(device)


def to_device(data,device):
    if isinstance(data, (list,tuple)): #The isinstance() function returns True if the specified object is of the specified type, otherwise False.
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)

class DeviceDataLoader():
    def __init__(self,dl,device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b,self.device)
    
    def __len__(self):
        return len(self.dl)

train_loder = DeviceDataLoader(train_loder,device)
val_loader = DeviceDataLoader(val_loader,device)


def evaluate(model,val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs,lr,model,train_loder,val_loader,opt_func = torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(),lr)
    for epoch in range(epochs):
        for batch in train_loder:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()#used to update the parameters
            optimizer.zero_grad()#Clears the gradients of  optimizer
        result = evaluate(model,val_loader)
        model.epoch_end(epoch,result)
        history.append(result)
    return history

model = MnistModel(input_size, hidden_size=32, out_size=num_classes)


to_device(model, device)

evaluate(model,val_loader)

history = fit(15,0.5,model,train_loder,val_loader)





print("==================================")