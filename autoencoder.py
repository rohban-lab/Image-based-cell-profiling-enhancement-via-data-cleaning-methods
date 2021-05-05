import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt 


os.system('nvidia-smi -L')

"""# Load Data

"""
train_data = pd.read_csv('data_cleaning/train_clean.csv', index_col=0)
val_data = pd.read_csv('data_cleaning/val_clean.csv', index_col=0)
test_data = pd.read_csv('data_cleaning/test_clean.csv', index_col=0)
drug_dictionary = {}
drug_csv = pd.read_csv('data_cleaning/drug_dict.csv')

for i in range(len(drug_csv)):
  drug_dictionary[drug_csv.iloc[i][1]] = drug_csv.iloc[i][0]

train_data = train_data.reset_index(drop=True)

"""# Dataset

"""

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
        return (item, label)

train_dataset = cell_dataset(train_data)
val_dataset = cell_dataset(val_data)
test_dataset = cell_dataset(test_data)

batch_size = 128

train_loader = DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
val_loader = DataLoader(dataset=val_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
test_loader = DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

"""# Define Model"""

class Autoencoder(nn.Module):
    def __init__(self, features, num_classes):
        super(Autoencoder, self).__init__()
        self.z_dim = 100
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(features, 400),
            nn.ReLU(),
            nn.Linear(350, 200),
            nn.ReLU(),
            nn.Linear(250, self.z_dim),

        )
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 250),
            nn.ReLU(),
            nn.Linear(200,350),
            nn.ReLU(),
            nn.Linear(400, features),

        )
    def add_noise(self, x):
        noise = torch.randn(x.size()) * 0.7
        noise = noise.to(device)
        noisy_x = x + noise
        return noisy_x
    def find_representation(self, x):
      x = self.encoder(x)
      return x

    def forward(self, x):
        x = self.add_noise(x)
        x = self.encoder(x)
        x = self.decoder(x)
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
    # correct = 0
    # total = 0
    running_loss = []
    with torch.no_grad():
        for  i, (inputs, labels) in enumerate(dataLoader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            # _, predicted = torch.max(outputs.data, 1)
            # correct += (predicted == labels).sum().item()
            loss = criterion(outputs, inputs)
            running_loss.append(loss.item())
            # total += labels.size(0)
    return np.mean(running_loss)

train_losses=[]
val_lossses=[]
val_accs=[]
train_accs=[] 
def train(): 
    for epoch in range(num_epochs):
        net.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
    
        train_predict = predict(train_loader)
        val_predict = predict(val_loader)
        train_losses.append(train_predict)
        val_lossses.append(val_predict)


        print('epoch:', epoch+1)
        print("\ttrain :  ",  "  loss:",train_predict)
        print("\tvalidation :  ",  "  loss:",val_predict)

num_epochs = 100
learning_rate = 0.0001
net=Autoencoder(train_data.shape[1], len(drug_dictionary))
net = net.to(device)
criterion = nn.MSELoss()
criterion = criterion.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay = 1e-5)

train()

"""## plot results"""

plt.plot(train_losses, label = "train") 
plt.plot(val_lossses, label = "validation") 
  
plt.xlabel('iteration') 

plt.ylabel('loss') 
plt.legend() 

plt.savefig('AEloss.png')

"""# Results on test set"""

test_predict = predict(test_loader)
print("test loss: ",test_predict)

"""# save model"""

torch.save(net, 'dae.pt')
