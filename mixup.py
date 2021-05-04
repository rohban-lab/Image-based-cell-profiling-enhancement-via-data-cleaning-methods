import os
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
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

class Net(nn.Module):
    def __init__(self, features, num_classes):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(features, 300, bias = True)
        self.bn1 = nn.BatchNorm1d(300)

        self.fc2 = nn.Linear(300, 200, bias=True)
        self.bn2 = nn.BatchNorm1d(200)
        self.do2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(200, 100, bias=True)
        self.bn3 = nn.BatchNorm1d(100)
        self.do3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(100, num_classes, bias=True)
        

        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)




    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = (self.relu(self.do2(self.bn2(self.fc2(x)))))
        x = (self.relu(self.do3(self.bn3(self.fc3(x)))))
        x = self.soft(self.fc4(x))
        return x

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    use_cuda = True
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    use_cuda = False
    print("Running on the CPU")

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam



def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

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
            inputs, targets_a, targets_b, lam = mixup_data(inputs, onehots,
                                                       alpha, use_cuda)
            inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            _, targets_a = torch.max(targets_a.data, 1)
            _, targets_b = torch.max(targets_b.data, 1)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            total += labels.size(0)
            correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
            running_loss.append(loss.item())

            
            correct = correct.item()
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

            inputs, targets_a, targets_b, lam = mixup_data(inputs, onehots,
                                                       alpha, use_cuda)
            inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
            outputs = net(inputs)
            _, targets_a = torch.max(targets_a.data, 1)
            _, targets_b = torch.max(targets_b.data, 1)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            loss.backward()
            optimizer.step()

            
      
        train_predict = predict(train_loader)
        val_predict = predict(val_loader)
        train_losses.append(train_predict[1])
        train_accs.append(train_predict[0])
        val_lossses.append(val_predict[1])
        val_accs.append(val_predict[0])


        print('epoch:', epoch+1)
        print("\ttrain : accuracy: ", train_predict[0], "  loss:",train_predict[1])
        print("\tvalidation : accuracy: ", val_predict[0], "  loss:",val_predict[1])

num_epochs = 200
learning_rate = 0.001
alpha = 0.01

net=Net(train_data.shape[1], len(drug_dictionary))
net = net.to(device)
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)  
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

train()

plt.plot(train_accs, label = "train") 
plt.plot(val_accs, label = "validation") 
  

plt.xlabel('iteration') 

plt.ylabel('accuracy') 
plt.legend() 

plt.savefig('train_accs_mixup.png')

plt.clf()
plt.plot(train_losses, label = "train") 
plt.plot(val_lossses, label = "validation") 
  
plt.xlabel('iteration') 

plt.ylabel('loss') 
plt.legend() 

plt.savefig('train_losses_mixup.png')

test_predict = predict(test_loader)
print("test accuracy: ",test_predict[0])

torch.save(net, 'mixup.pt')
