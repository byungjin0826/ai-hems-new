# 그냥 중첩해서 사용안하는 시간
from sklearn.metrics import accuracy_score
from utils import *
import datetime

sql = """
SELECT *
FROM AH_USE_LOG_BYMINUTE_LABLED
WHERE 1=1
AND device_address = '00158D000151B1F9'
"""
df = get_table_from_db(sql, db = 'aihems_api_db')
df = binding_time(df, format='%Y%m%d %H:%M')

df['2019-02-28']

df.index

# pivot_table로 만들기
def decomposition_datetime(df):
    df.loc[:'year'] = [str(x) for x in df.index.year]
    df.loc[:'month'] = [str(x) for x in df.index.month]
    df.loc[:'day'] = [str(x) for x in df.index.day]
    df.loc[:'hour'] = [str(x) for x in df.index.hour]
    df.loc[:'year'] = [str(x) for x in df.index.year]



    return df