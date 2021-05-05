# Data Cleaning for image-based profiling Enhancement 
## Abstract
With the advent of high-throughput assays, a large number of biological experiments can be carried out. Image-based profiling is among the most accessible and inexpensive technologies for this purpose. High-throughput image-based assays have earlier shown to be effective in characterizing unknown functions of genes and small molecules. CellProfiler is a popular and commonly used  tool to process and quantify the data that is produced by these assays by making various measurements, or features, for each cell. However, there may be several sources of error in the quantification pipeline that is developed in CellProfiler. In this work, we examined different steps to improve profiles that are taken from CellProfiler to identify mechanisms of action of different drugs. These steps mainly consist of data cleaning, cell level outlier detection, toxic drug detection, and regressing out the cell area from all other features, as many of them are widely affected by the cell area. We also examined deep learning based methods to improve the CellProfiler output, and finally we suggest possible avenues for future research.
## Prerequisites
Prerequisites can be found in `requirements.txt`.

## Dataset
Dataset was from [GIGADB](http://gigadb.org/dataset/view/id/100351). Each plate was downloaded separately and tables of cell, nuclei and cytoplasm merged for each plate and all features saved in a CSV file in this format: plate\<plate_id>. 
The code is available in `download.py` .

The normalization by DMSO features is in `normalize.py` .
After downloading all plates, each plate was normalized individually.

Metadata can be found in `moa.txt`and `final_features.txt`.
## Outlier Cell Detection
In `outlierdetection.py` outlier cells in each plate was found by histrogm-based technique and the metadata for distinguishing outliers from others saved in a CSV file for each plate.
## Removing toxic drugs
Output of `remove_toxic_drugs.py` is a CSV file that sorts drugs by median of cell counts of the wells treated by them. In the profile creation pipeline, 5% of drugs with lowest cell counts were removed.

## Creating profile and evaluation
In `simple_find_representation.py` profiles were created without any data cleaning method. After the profile creation, aggregation on well, aggregation on broad sample and calculating similarity and odds ratio and plotting the results added.

In `final_find_representation.py` profiles were created with different data cleaning methods, such as Histrogram-based outlier detection, removing outlier and toxic drugs. Regressing out was another step in the profile creation too. Other steps for evaltion was exactly similar to `simple_find_representation.py`.

## Deep learning
In `train_test_split.py` five mechanisms were chosen and fifty random drugs with these mechnisms were picked for train/test data.
After splittind and savind train, validation and test data, different models such as simple fully connected network with mixup technique (`mixup.py`), simple fully connected network with HSIC loss (`hsic.py`) and various versions of AutoEncoder(`autoencoder.py`) like denoising autoencoder trained on data.
When trained model saved, by using `deep_representation.py`, the profile for final evalution is the output of the model (The input of the model is the cleaned profile created in previous step).



