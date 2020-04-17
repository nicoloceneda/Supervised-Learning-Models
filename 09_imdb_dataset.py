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

    tar.extractall(path='imdb dataset/original')


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
