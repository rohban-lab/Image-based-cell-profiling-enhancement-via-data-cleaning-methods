import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from scipy import stats
from sklearn.linear_model import LinearRegression
from pyod.models.hbos import HBOS


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


######

count_per_drug = pd.read_csv('../count_per_drug.csv')
num_removing = int(len(count_per_drug) * 0.05)
for i in range(num_removing):
  drugs.remove(count_per_drug.iloc[i]['Metadata_broad_sample'])

#############

print("total drugs number:", len(drugs))
dataDir = '../../../../../mnt/sda1/project/htm/celldata/normalized'
files = os.listdir(dataDir)
print("number of files: ", len(files), '\n\n')

for f in files:
  representation = []
  path = dataDir + '/' + f
  data = pd.read_csv(path,index_col=0)
  data['plate'] = f
  data = data[data['Metadata_broad_sample'].isin(drugs)]
  data = data[data.columns.intersection(selected_cols)]


  b = data['Metadata_broad_sample']
  w = data['Metadata_Well']
  p =  data['plate']
  del data['Metadata_broad_sample']
  del data['Metadata_Well']
  del data['plate']
  
  outliers_fraction = 0.01
  clf = HBOS (contamination= outliers_fraction)
  clf.fit(data)
  y_pred = clf.predict(data)
  X = pd.DataFrame()
  X['outlier'] = y_pred.tolist()
  X['Metadata_broad_sample'] = b
  X['Metadata_Well'] = w
  X['plate'] = p
  X.to_csv('outlier_without_regress/'+f)


  #target = y_pred.tolist()
  #tsne = TSNE(n_components= 2, verbose=1, perplexity=40, n_iter=2000)
  #tsne_results = tsne.fit_transform(data)
  #fig = plt.figure()
  #ax = fig.add_subplot(111, projection='3d')
  #ax.scatter(tsne_results[:,0], tsne_results[:,1],tsne_results[:,2], cmap = "coolwarm", edgecolor = "None" , c = target)
  #plt.scatter(tsne_results[:,0],tsne_results[:,1], c=target,
   #                   cmap = "coolwarm", edgecolor = "None")
  #plt.savefig(f+"out.png")
  #break
