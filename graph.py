import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import settings
import plotnine as p9
import warnings
warnings.filterwarnings('ignore')

device_id = '000D6F000C1382DE1'

sql = f"""
SELECT *
FROM AH_USE_LOG_BYMINUTE
WHERE 1=1
AND DEVICE_ID = '{device_id}'
AND COLLECT_DATE >= '20191101'"""

df = pd.read_sql(sql, con = settings.conn)



(p9.ggplot(data=df, mapping=p9.aes(x='COLLECT_TIME', y='COLLECT_DATE'))+
     p9.geom_tile(p9.aes(fill='APPLIANCE_STATUS')))