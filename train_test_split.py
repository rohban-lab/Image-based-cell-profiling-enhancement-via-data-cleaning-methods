import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LinearRegression


dff = pd.read_csv('../final_features.txt', sep="\t", header=None)
selected_cols  = list(dff.iloc[:, 0])

broadsample_moa = pd.read_csv('../moa.txt', sep = "\t")
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


                          #######
outlier_drugs = list(set(pd.read_csv('../median_outlier_drugs.txt', sep = "\t", header = None)[0]))

print('outlier drugs =',len(outlier_drugs))
for d in outlier_drugs:
    drugs.remove(d)


count_per_drug = pd.read_csv('../count_per_drug.csv')
num_removing = int(len(count_per_drug) * 0.05)
for i in range(num_removing):
    drugs.remove(count_per_drug.iloc[i]['Metadata_broad_sample'])
                                  


######

dataDir = '../../../../../mnt/sda1/project/htm/celldata/normalized'
files = os.listdir(dataDir)
print("number of files: ", len(files), '\n\n')


### choose 5 0r 6 moa for training

moa = pd.read_csv('../moa.txt', sep = "\t")
selected_moa = ['calcium channel blocker','adrenergic receptor agonist','glucocorticoid receptor agonist','Cyclooxygenase inhibitor','protein synthesis inhibitor       ','histamine receptor antagonist        ']
outliers = list(pd.read_csv('../median_outlier_drugs.txt', sep = "\t", header = None)[0])
moa = moa[~moa['Metadata_broad_sample'].isin(outliers)]
moa = moa[moa['Metadata_moa'].isin(selected_moa)]
moas = list(moa['Metadata_broad_sample'])


np.random.seed(0)
num_drugs = 50
selected_drugs = list(np.random.choice(moas,num_drugs))
print(selected_drugs)

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

"""# Load Data"""

dataDir = '../../../../../mnt/sda1/project/htm/celldata/normalized'
files = os.listdir(dataDir)
print("number of files: ", len(files))
s =0
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
           model = LinearRegression(fit_intercept=False).fit(cell_area, feature)
           new_feature = feature - model.predict(cell_area)
           data[col] =  new_feature
  data['Cells_AreaShape_Area'] = cell_area_col


  X = pd.read_csv('outlier_without_regress/'+f)
  print(data.shape, X.shape)
  data['outlier'] = list(X['outlier'])
  data = data[data['outlier'] == 0]
  #print(data.colmns)
  del data['outlier']

  ##### choose train drugs
  data = data[data['Metadata_broad_sample'].isin(selected_drugs)]

  frames.append(data)


data = pd.concat(frames)
print("shape of all data: ",data.shape)


print('unique  ',len(pd.unique(data['Metadata_broad_sample'])))

print("shape of all data: ",data.shape)

print("number of different drugs: ",len(pd.unique(data['Metadata_broad_sample'])))


groups = data.groupby(['Metadata_Well', 'plate'])

total_index = [i for i in range(len(groups))]
random.shuffle(total_index)

train_inds = total_index[:int(len(total_index)*train_ratio)]
val_inds = total_index[int(len(total_index)*train_ratio): int(len(total_index)*(1-test_ratio))]
test_inds = total_index[int(len(total_index)*(1-test_ratio)):]

def get_cell(indexes, groups):
  cells = []
  temp = []
  for key in indexes:
    # print(key)
    _,df = list(groups)[key]
    # print(df)
    max_num = len(df)
    temp.append(max_num)
    # print(max_num)
    select_num = 300
    if select_num > max_num:
      select_num = max_num
    # randomlist = random.sample(range(0, max_num), max_num)
    randomlist = random.sample(range(0, max_num), select_num)
    selected_cells = df.iloc[randomlist]
    cells.append(selected_cells)
  # print(min(temp), max(temp))
  return pd.concat(cells)

train_data = get_cell(train_inds, groups)
val_data = get_cell(val_inds, groups)
test_data = get_cell(test_inds, groups)

train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)

train_data.head()

print("train has all drugs? ",len(train_data['Metadata_broad_sample'].unique()) == len(data['Metadata_broad_sample'].unique()))

print("number of uniques plates in train:" ,len(pd.unique(train_data['plate'])) )

print("train shape: ",train_data.shape ,"  test shape:", test_data.shape)

test_len = len(test_data)
train_len = len(train_data)
val_len = len(val_data)



total = pd.concat([train_data, val_data, test_data])
# total = total.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

train_data = total.iloc[[i for i in range(train_len)]]
val_data = total.iloc[[i for i in range(train_len, train_len + val_len)]]
test_data = total.iloc[[i for i in range(train_len + val_len, train_len + val_len + test_len)]]

broadsamples = pd.unique(data['Metadata_broad_sample'])
drug_dictionary = {}
for i in range(len(broadsamples)):
  drug_dictionary[broadsamples[i]] = i

train_data.to_csv('train_clean.csv')
val_data.to_csv('val_clean.csv')
test_data.to_csv('test_clean.csv')

d = pd.DataFrame(drug_dictionary.items())
d.to_csv('drug_dict.csv')

print("columns",train_data.columns)
