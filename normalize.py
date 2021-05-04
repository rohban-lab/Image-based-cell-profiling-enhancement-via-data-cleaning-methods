import pandas as pd
import numpy as np
from statsmodels import robust
from scipy import stats

plates = [24277, 24280, 24293, 24294, 24295, 24296, 24297, 24300, 24301, 24302, 24303, 24304, 24305, 24306, 24307, 24308, 24309, 24310, 24311 ,24313,24319,24320,24321,24352,24357,25937,25938,25939,25943,25944,25945,25949,25955,25962,25965,25966,25967,25968,25983,25984,25985,25986,25987,25988,25989,25990,25991,25992,26224,26232,26239, 26247,]

for pl in plates:
  data = pd.read_csv('../plate'+str(pl)+'.csv', index_col = 0)

  dmso = data[data['Metadata_broad_sample']=='DMSO']
  del dmso['Metadata_broad_sample']

  dmso = dmso.groupby(['Metadata_Well']).agg(np.median)
  dmso = dmso.reset_index(drop = False, inplace = False)
  del dmso['Metadata_Well']
  dmso = dmso.replace('nan', np.nan).dropna()




  ## noramalize other rows using dmso
  print('####### normalizing')
  d = data[data['Metadata_broad_sample']!='DMSO']
  d = d.replace('nan', np.nan).dropna()
  Metadata_broad_sample =  d['Metadata_broad_sample']
  Metadata_Well =  d['Metadata_Well']
  del d['Metadata_broad_sample']
  del d['Metadata_Well']
  cols = d.columns
  values = d.values
  dmsoval = dmso.values
  median = np.median(dmsoval, axis = 0)
  mad = stats.median_absolute_deviation(dmsoval)
  values -= median
  values /= mad

  new_df = pd.DataFrame(values , columns= cols)
  new_df['Metadata_broad_sample'] = list(Metadata_broad_sample)
  new_df['Metadata_Well'] = list(Metadata_Well)

  new_df.to_csv('norm_plate'+str(pl)+'.csv')
  print('####### plate saved!')