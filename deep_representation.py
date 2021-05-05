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
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms
from torch.autograd import Variable
from scipy import stats

os.system('nvidia-smi')
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")



class Autoencoder(nn.Module):
    def __init__(self, features, num_classes):
        super(Autoencoder, self).__init__()
        self.z_dim = 100
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(features, 350),
            # nn.ReLU(),
            # nn.Linear(350, 200),
            # nn.ReLU(),
            # nn.Linear(250, self.z_dim),

        )
        self.decoder = nn.Sequential(
            # nn.Linear(self.z_dim, 250),
            # nn.ReLU(),
            # nn.Linear(200,350),
            # nn.ReLU(),
            nn.Linear(350, features),

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

class Net(nn.Module):
    def __init__(self, features, num_classes):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(features, 300, bias = True)
        self.bn1 = nn.BatchNorm1d(300)

        self.fc2 = nn.Linear(300, 200, bias=True)
        self.bn2 = nn.BatchNorm1d(200)
        self.do2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(200, 100, bias=True)
        self.bn3 = nn.BatchNorm1d(100)
        self.do3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(100, num_classes, bias=True)
        

        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)



    def find_representation(self, x):
        x = self.relu(self.fc1(x))
        x = (self.relu(self.fc2(x)))
        x = (self.relu(self.fc3(x)))
        return x



    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = (self.relu(self.bn2(self.fc2(x))))
        x = (self.relu(self.bn3(self.fc3(x))))
        x = (self.fc4(x))

        return x


class custom_dataset(Dataset):
    def __init__(self, df):
        self.metadata_broad_sample = list(df['Metadata_broad_sample'])
        self.meta_data_well = list(df['Metadata_Well'])
        self.plate = list(df['plate'])
        del df['Metadata_broad_sample']
        del df['Metadata_Well']
        del df['plate']
        self.X = torch.FloatTensor(df.values)

   
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        item = self.X[idx]
        return (item)        

path = 'mixup.pt'
model = torch.load(path)
print(model.parameters)

broadsample_moa = pd.read_csv('moa.txt', sep = "\t")
broad_sample = broadsample_moa['Metadata_broad_sample']
moa = broadsample_moa['Metadata_moa']
moa_dict = {}
for i in range (len(broad_sample)):
  sample = broad_sample[i]
  m = moa[i].split("|")
  if sample in moa_dict:
    if len(moa_dict[sample]) > len(m):
      continue
  moa_dict[sample] = m

drugs = list(moa_dict.keys()) 
outliers = list(pd.read_csv('median_outlier_drugs.txt', sep = "\t", header = None)[0])
for d in outliers:
  drugs.remove(d)

count_per_drug = pd.read_csv('count_per_drug.csv')
num_removing = int(len(count_per_drug) * 0.05)
for i in range(num_removing):
      drugs.remove(count_per_drug.iloc[i]['Metadata_broad_sample'])


print("total drugs number: ", len(drugs))

dataDir = '../../../../mnt/sda1/project/htm/celldata/normalized'
files = os.listdir(dataDir)
print("number of files: ", len(files))

dff = pd.read_csv('final_features.txt', sep="\t", header=None)
selected_cols  = list(dff.iloc[:, 0])

model.eval()
batch_size = 32

frames = []
for f in files:
  representation = []
  path = dataDir + '/' + f
  data = pd.read_csv(path,index_col=0)
  data['plate'] = f
  data = data[data['Metadata_broad_sample'].isin(drugs)]
  data = data[data.columns.intersection(selected_cols)]

  
  cell_area_col = list(data['Cells_AreaShape_Area'])
  cell_area = np.array(cell_area_col).reshape(-1,1)
  del data['Cells_AreaShape_Area']
  for col in data.columns:
       if col not in ['Metadata_Well','plate','Metadata_broad_sample']:
           feature = list(data[col])
           model_reg = LinearRegression(fit_intercept=False).fit(cell_area, feature)
           new_feature = feature - model_reg.predict(cell_area)
           data[col] =  new_feature
  data['Cells_AreaShape_Area'] = cell_area_col

  X = pd.read_csv('data_cleaning/outlier_without_regress/'+f)
  print(data.shape, X.shape)
  data['outlier'] = list(X['outlier'])
  data = data[data['outlier'] == 0]
  #print(data.colmns)
  del data['outlier']

  dataset = custom_dataset(data.copy())

  data_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=False)

  for i, (inputs) in enumerate(data_loader):
    inputs = inputs.to(device)
    rep = model.find_representation(inputs)
    representation += rep.tolist()

  print("data shape: ",data.shape, " rep shape: ",np.array(representation).shape)

  total = pd.DataFrame(np.array(representation))
  total['Metadata_broad_sample'] = dataset.metadata_broad_sample
  total['Metadata_Well'] = dataset.meta_data_well
  total['plate'] = dataset.plate

  total = total.groupby(['Metadata_Well','plate','Metadata_broad_sample']).agg(np.median)
  total = total.reset_index(drop = False, inplace = False)
  frames.append(total)






total = pd.concat(frames)
del total['Metadata_Well']
del total['plate']
print("number of cols: ",len(total.columns))
print(total.head())


total = total.groupby(['Metadata_broad_sample']).agg('mean')
total = total.reset_index(drop = False, inplace = False)

drugs = list(total['Metadata_broad_sample'])
del total['Metadata_broad_sample']

corr_matrix = np.corrcoef(total.values, rowvar= True)

print("correlation matrix: ",corr_matrix.shape)

print("min: ",np.min(corr_matrix), "max: ",np.max(corr_matrix))

temp = np.ones_like(corr_matrix) * -2
temp = np.tril(temp)
corr_matrix = np.triu(corr_matrix, 1)
corr_matrix += temp

n = corr_matrix.shape[0]
ind_corr = np.dstack(np.unravel_index(np.argsort(corr_matrix.ravel()), (n,n)))[0][int(n*(n-1)/2)+n:]

def function(percentage):
  fisher_table = np.zeros((2,2))
  low_corrs = ind_corr[:int(len(ind_corr)*percentage)+1]
  top_corrs = ind_corr[int(len(ind_corr)*percentage)+1:]
  for item in top_corrs:
    i = item[0]
    j = item[1]
    first_moa = moa_dict[drugs[i]]
    second_moa = moa_dict[drugs[j]]
    ###### check if two list have at least one common element
    if set(first_moa) & set(second_moa):
      moa_found = True
    else:
      moa_found = False
    # moa_found = not set(first_moa).isdisjoint(second_moa)
    if moa_found:
      fisher_table[0,0] += 1
    else:
      fisher_table[0,1] += 1
  
  for item in low_corrs:
    i = item[0]
    j = item[1]
    first_moa = moa_dict[drugs[i]]
    second_moa = moa_dict[drugs[j]]
    ###### check if two list have at least one common element
    if set(first_moa) & set(second_moa):
      moa_found = True
    else:
      moa_found = False
    # moa_found = not set(first_moa).isdisjoint(second_moa)
    if moa_found:
      fisher_table[1,0] += 1
    else:
      fisher_table[1,1] += 1
  oddsratio, pvalue = stats.fisher_exact(fisher_table)
  return oddsratio


x = [0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5]
y =[]
for e in x:
  percentage = (100 -e)/ 100
  ratio = function(percentage)
  y.append(ratio)

import matplotlib.pyplot as plt 
  
print(y)
plt.plot(x, y, marker = 'o') 
 
plt.xlabel('k') 

plt.ylabel('odds ratio') 
  
plt.title('top k percent') 
  
plt.savefig('rep_hsic.png')
