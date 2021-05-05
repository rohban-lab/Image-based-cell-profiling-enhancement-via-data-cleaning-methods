import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

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


#################
outlier_drugs = list(set(pd.read_csv('median_outlier_drugs.txt', sep = "\t", header = None)[0]))

print('outlier drugs =',len(outlier_drugs))
for d in outlier_drugs:
    drugs.remove(d)
################


dataDir = '../../../../mnt/sda1/project/htm/celldata/normalized'

files = os.listdir(dataDir)
print("number of files: ", len(files))

frames = []
for f in files:
  representation = []
  path = dataDir + '/' + f
  data = pd.read_csv(path,index_col=0)
  data = data[data.columns.intersection(['Metadata_Well','plate','Metadata_broad_sample'])]
  data['plate'] = f
  data = data[data['Metadata_broad_sample'].isin(drugs)]

  temp = data.groupby(['Metadata_Well','plate','Metadata_broad_sample']).size().reset_index(name='counts')

  frames.append(temp)

total = pd.concat(frames)

print(total.shape)
print(total.columns)
print(total.head())
print('-------------------------')

total = total.groupby(['Metadata_broad_sample']).agg('median')

total = total.reset_index(drop = False, inplace = False)
print(total.shape)
print(total.head())
print('----------------------')

total = total.sort_values(by=['counts'])
total.to_csv('count_per_drug.csv')



