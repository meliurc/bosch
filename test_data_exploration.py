# -*- coding:utf-8 -*-
import re
import time
import collections
import numpy as np
import pandas as pd
from pandas import DataFrame

# import matplotlib.pyplot as plt
# from matplotlib_venn import venn2, venn3, venn3_circles

df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
                       'foo', 'bar', 'foo', 'foo'],
                 'B' : ['one', 'one', 'two', 'three',
                           'two', 'two', 'one', 'three'],
                'C' : np.random.randn(8),
                  'D' : np.random.randn(8)},
                  index=[3, 0, 1, 2, 5, 4, 7, 6])
df = df.reset_index()
print df, '\n-----------'
# print df['index'].diff().fillna(999999).astype(int)
print df['index'].iloc[::-1]

# df2 = DataFrame([[1, 0, 1, 0], [1, 1, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0],
#                  [0, 1, 0, 1], [1, 1, 0, 1], [1, 0, 0, 1], [0, 1, 0, 0]])

print df.columns.values