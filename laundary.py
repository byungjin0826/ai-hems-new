import pandas as pd


df = pd.read_csv('./sample_data/laundary_09.csv')
df['collected_date'] = df['collected_date'].astype(str)
df['collected_time'] = df['collected_time'].astype(str)
df['date_time'] = df['collected_date']+' '+df['collected_time']

pd.to_datetime(df.date_time, format = '%Y%m%d %H%M')


#
