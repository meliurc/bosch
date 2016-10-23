

import numpy as np
import pandas as pd
from pandas import DataFrame

df0 = DataFrame()
df1 = DataFrame(np.array(range(8)).reshape(4, 2))
df2 = DataFrame(np.array(range(8, 16)).reshape(4, 2))

df2.to_csv('df2')
df1.to_csv('df2')

