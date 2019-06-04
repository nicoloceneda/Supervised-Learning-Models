""" aclImdb
    -------
    Extract and convert to csv the aclImdb database.
"""


# ------------------------------------------------------------------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES AND/OR MODULES
# ------------------------------------------------------------------------------------------------------------------------------------------


import pyprind
import os
import pandas as pd
import numpy as np


# ------------------------------------------------------------------------------------------------------------------------------------------
# 2. PREPARE THE DATA
# ------------------------------------------------------------------------------------------------------------------------------------------


labels = {'pos': 1, 'neg': 0}

pbar = pyprind.ProgBar(50000)

df = pd.DataFrame()

for s in ('test', 'train'):

    for l in ('pos', 'neg'):

        path = os.path.join('aclImdb', s, l)

        for file in sorted(os.listdir(path)):

                with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:

                    txt = infile.read()

                df = df.append([[txt, labels[l]]], ignore_index=True)

                pbar.update()

df.columns = ['review', 'sentiment']

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv', index=False, encoding='utf-8')