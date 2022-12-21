""" IMDB DATASET
    -------------
    Download the imdb dataset and save the features and targets.
"""


# -------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------------------------------------


import os
import tarfile
import pyprind
import numpy as np
import pandas as pd


# -------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# -------------------------------------------------------------------------------


# Unpack the gzip compressed tarball archive

with tarfile.open('imdb dataset/original/aclImdb_v1.tar.gz', 'r:gz') as tar:

    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(tar, path="imdb dataset/original")


# Assemble the text documents from the decompressed archive into a single csv file

pbar = pyprind.ProgBar(50000)

labels = {'pos': 1, 'neg': 0}
df = pd.DataFrame()

for s in ('test', 'train'):

    for l in ('pos', 'neg'):

        path = os.path.join('imdb dataset/original/aclImdb', s, l)

        for file in sorted(os.listdir(path)):

            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:

                txt = infile.read()

            df = df.append([[txt, labels[l]]], ignore_index=True)
            pbar.update()

df.columns = ['review', 'sentiment']

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('imdb dataset/extracted/imdb_data.csv', index=False, encoding='utf-8')
