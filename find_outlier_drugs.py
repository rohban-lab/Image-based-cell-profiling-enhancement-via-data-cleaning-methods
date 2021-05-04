import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

broadsample_moa = pd.read_csv('moa.txt', sep="\t")
broad_sample = broadsample_moa['Metadata_broad_sample']
moa = broadsample_moa['Metadata_moa']
moa_dict = {}
for i in range(len(broad_sample)):
    sample = broad_sample[i]
    m = moa[i].split("|")
    if sample in moa_dict:
        if len(moa_dict[sample]) > len(m):
            continue
    moa_dict[sample] = m

drugs = list(moa_dict.keys())

dataDir = '../../../../mnt/sda1/project/htm/celldata/normalized'
files = os.listdir(dataDir)
print("number of files: ", len(files))

dff = pd.read_csv('final_features.txt', sep="\t", header=None)
selected_cols  = list(dff.iloc[:, 0])

frames = []
for f in files:
    representation = []
    path = dataDir + '/' + f
    data = pd.read_csv(path, index_col=0)
    data['plate'] = f
    data = data[data.columns.intersection(selected_cols)]
    data = data[data['Metadata_broad_sample'].isin(drugs)]
    data = data.groupby(['Metadata_Well', 'plate', 'Metadata_broad_sample']).agg(np.median)
    data = data.reset_index(drop=False, inplace=False)

    frames.append(data)

total = pd.concat(frames)


N = 1000
medians = []
for i in range(N):
    # from each compound choose one random well
    df = total.groupby("Metadata_broad_sample").sample(n=1, random_state=1)
    df = df.reset_index(drop=True, inplace=False)
    # choose 3 random compounds
    rows = np.random.choice(df.index.values, 3, replace=False)
    df = df.iloc[rows]
    del df['Metadata_Well']
    del df['plate']
    del df['Metadata_broad_sample']
    corr = df.T.corr()
    # print(corr.shape == (3,3))
    up_tri = np.asarray(corr)[np.triu_indices(3, k=1)]
    medians.append(np.median(up_tri))


s = pd.Series(sorted(medians))
threshold = s.quantile(.95)
plt.hist(s, bins=30, color='c', edgecolor='k', alpha=0.65)
plt.axvline(s.quantile(.8), color='red', linewidth=1)
plt.savefig('histogram_d1_median.png')


outliers = []

for drug in drugs:
    temp = total[total['Metadata_broad_sample'] == drug]
    del temp['Metadata_Well']
    del temp['plate']
    del temp['Metadata_broad_sample']
    corr = temp.T.corr()
    # print(corr.shape, end='\t')
    up_tri = np.asarray(corr)[np.triu_indices(temp.shape[0], k=1)]
    if np.median(up_tri) < threshold:
        outliers.append(drug)


print(len(outliers), 'outlier out of total of', len(drugs), 'drugs!')

with open("median_outlier_drugs.txt", "w") as outfile:
    outfile.write("\n".join(outliers))
