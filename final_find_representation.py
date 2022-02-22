
import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression



dff = pd.read_csv('../final_features.txt', sep="\t", header=None)
selected_cols  = list(dff.iloc[:, 0])




broadsample_moa = pd.read_csv('../moa.txt', sep = "\t")
broad_sample = broadsample_moa['Metadata_broad_sample']
moa = broadsample_moa['Metadata_moa']
moa_dict = {}
for i in range (len(broad_sample)):
  sample = broad_sample[i]
  m = moa[i].lower().split("|")
  if sample in moa_dict:
    if len(moa_dict[sample]) > len(m):
      continue
  moa_dict[sample] = m

drugs = list(moa_dict.keys()) 


####### remove outlier drugs
outlier_drugs = list(set(pd.read_csv('../median_outlier_drugs.txt', sep = "\t", header = None)[0]))

print('outlier drugs =',len(outlier_drugs))
for d in outlier_drugs:
    drugs.remove(d)

temp_l = []

######### remove toxic drugs
count_per_drug = pd.read_csv('../count_per_drug.csv')
num_removing = int(len(count_per_drug) * 0.05)
for i in range(num_removing):
  drugs.remove(count_per_drug.iloc[i]['Metadata_broad_sample'])
###########

print("total drugs number:", len(drugs))
dataDir = '../../../../../mnt/sda1/project/htm/celldata/normalized'
files = os.listdir(dataDir)
print("number of files: ", len(files), '\n\n')
frames = []
for f in files:
  representation = []
  path = dataDir + '/' + f
  data = pd.read_csv(path,index_col=0)
  data['plate'] = f
  data = data[data['Metadata_broad_sample'].isin(drugs)]
  data = data[data.columns.intersection(selected_cols)]

  ######### OUTLIER DETECTION
  X = pd.read_csv('outlier_without_regress/'+f)
  print(data.shape, X.shape)
  data['outlier'] = list(X['outlier'])
  data = data[data['outlier'] == 0]
  del data['outlier']
  #################

  ###### REGRESS OUT
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
  ##############

  temp_l.append(len(data))



  ##### aggregate on well, plate 
  data = data.groupby(['Metadata_Well','plate','Metadata_broad_sample']).agg(np.median)
  data = data.reset_index(drop = False, inplace = False)
  print(f)
  frames.append(data)
print(temp_l)

total = pd.concat(frames)



del total['Metadata_Well']
del total['plate']

print(total.shape)
print("number of cols: ",len(total.columns))
print(total.head())

##### aggregate on broad sample
total = total.groupby(['Metadata_broad_sample']).agg('mean')

total = total.reset_index(drop = False, inplace = False)
print(total.shape)




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
### argsort of upper triangle of the correlation matrix
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
    if moa_found:
      fisher_table[1,0] += 1
    else:
      fisher_table[1,1] += 1
  oddsratio, pvalue = stats.fisher_exact(fisher_table)
  return oddsratio


x = [0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5]
y =[]
for e in x:
  percentage = (100 - e) / 100
  ratio = function(percentage)
  y.append(ratio)

import matplotlib.pyplot as plt 
  
print(y)
plt.plot(x, y, marker = 'o') 
 
plt.xlabel('k') 
plt.ylabel('odds ratio') 
  
plt.title('top k percent') 
  
plt.show()