import os
import pandas as pd
import numpy as np
import random
import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms
from torch.autograd import Variable

os.system('nvidia-smi')
train_data = pd.read_csv('data_cleaning/train_clean.csv', index_col=0)
val_data = pd.read_csv('data_cleaning/val_clean.csv', index_col=0)
test_data = pd.read_csv('data_cleaning/test_clean.csv', index_col=0)

drug_dictionary = {}
drug_csv = pd.read_csv('data_cleaning/drug_dict.csv')

for i in range(len(drug_csv)):
  drug_dictionary[drug_csv.iloc[i][1]] = drug_csv.iloc[i][0]

train_data = train_data.reset_index(drop=True)

class cell_dataset(Dataset):
    def __init__(self, df):
        self.metadata_well = list(df['Metadata_Well'])
        self.metadata_plate = list(df['plate'])
        del df['Metadata_Well']
        del df['plate']
        self.y = [drug_dictionary[x] for x in list(df['Metadata_broad_sample'])]
        del df['Metadata_broad_sample']
        self.X = torch.FloatTensor(df.values)
        self.y = torch.LongTensor(self.y)

   
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        item = self.X[idx]
        label = self.y[idx]
        n = len(drug_dictionary)
        onehot = np.zeros(n)
        onehot[label] = 1
        onehot = torch.FloatTensor(onehot)
        return (item, label, onehot)

train_dataset = cell_dataset(train_data)
val_dataset = cell_dataset(val_data)
test_dataset = cell_dataset(test_data)

batch_size = 32

train_loader = DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
val_loader = DataLoader(dataset=val_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
                                          
## https://github.com/danielgreenfeld3/XIC/blob/master/hsic.py
class HSICLoss(nn.Module):
    def __init__(self):
        super(HSICLoss,self).__init__()

    def pairwise_distances(self, x):
      #x should be two dimensional
      instances_norm = torch.sum(x**2,-1).reshape((-1,1))
      return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

    def GaussianKernelMatrix(self, x, sigma=1):
        pairwise_distances_ = self.pairwise_distances(x)
        return torch.exp(-pairwise_distances_ /sigma)

    def forward(self, x, y, s_x=1.5, s_y=1.5):
        m,_ = x.shape #batch size
        K = self.GaussianKernelMatrix(x,s_x)
        L = self.GaussianKernelMatrix(y,s_y)
        H = torch.eye(m) - 1.0/m * torch.ones((m,m))
        H = H.cuda()
        HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
        return HSIC

class Net(nn.Module):
    def __init__(self, features, num_classes):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(features, 300, bias = True)
        self.bn1 = nn.BatchNorm1d(300)

        self.fc2 = nn.Linear(300, 200, bias=True)
        self.bn2 = nn.BatchNorm1d(200)
        # self.do2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(200, 100, bias=True)
        self.bn3 = nn.BatchNorm1d(100)
        # self.do3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(100, num_classes, bias=True)
        

        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)




    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = (self.relu(self.bn2(self.fc2(x))))
        x = (self.relu(self.bn3(self.fc3(x))))
        x = self.soft(self.fc4(x))
        return x

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

"""# Train model"""

def predict(dataLoader):
    net.eval()
    correct = 0
    total = 0
    running_loss = []
    with torch.no_grad():
        for  i, (inputs, labels, onehots) in enumerate(dataLoader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            onehots = onehots.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            residuals = onehots - outputs
            loss = criterion(inputs, residuals)
            running_loss.append(loss.item())
            total += labels.size(0)
    return (100.0*correct/total,np.mean(running_loss))

train_losses=[]
val_lossses=[]
val_accs=[]
train_accs=[] 
def train(): 
    for epoch in range(num_epochs):
        net.train()
        for i, (inputs, labels, onehots) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            onehots = onehots.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            residuals = onehots - outputs
            loss = criterion(inputs, residuals)
            loss.backward()
            optimizer.step()

            
      
        train_predict = predict(train_loader)
        val_predict = predict(val_loader)
        train_losses.append(train_predict[1])
        train_accs.append(train_predict[0])
        val_lossses.append(val_predict[1])
        val_accs.append(val_predict[0])
        torch.save(net, 'hsic_net.pt')


        print('epoch:', epoch+1)
        print("\ttrain : accuracy: ", train_predict[0], "  loss:",train_predict[1])
        print("\tvalidation : accuracy: ", val_predict[0], "  loss:",val_predict[1])

num_epochs = 100
learning_rate = 0.001
net=Net(train_data.shape[1], len(drug_dictionary))
net = net.to(device)
criterion = HSICLoss()
criterion = criterion.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

train()

"""## plot results"""

plt.plot(train_accs, label = "train") 
plt.plot(val_accs, label = "validation") 
  

plt.xlabel('iteration') 

plt.ylabel('accuracy') 
plt.legend() 

plt.savefig('hsic_acc.png')
plt.clf()

plt.plot(train_losses, label = "train") 
plt.plot(val_lossses, label = "validation") 
  
plt.xlabel('iteration') 

plt.ylabel('loss') 
plt.legend() 

plt.savefig('hsic_loss.png')

"""# Results on test set"""

test_predict = predict(test_loader)
print("test accuracy: ",test_predict[0])

"""# save model"""

torch.save(net, 'hsic.pt')
