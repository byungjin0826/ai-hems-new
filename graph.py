import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotnine as p9


import warnings
warnings.filterwarnings('ignore')

n = 10
df = pd.DataFrame({'x': np.arange(n),
                   'y': np.arange(n),
                   'yfit': np.arange(n) + np.tile([-.2, .2], n//2),
                   'cat': ['a', 'b']*(n//2)})

(p9.ggplot(data=df, mapping=p9.aes(x='x', y='y'))+p9.geom_point())