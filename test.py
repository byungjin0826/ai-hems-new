from flask import Flask
from flask_restful import Resource, Api, reqparse
import utils
from joblib import load, dump
import datetime
import sklearn as sk
import matplotlib.pyplot as plt
# from silvercare import silvercare_api
import pymysql
import pandas as pd
import numpy as np

gateway_id = 'ep17470201'
device_id = '00158D000151B4721'
date = '20191030'  # 오늘 날짜.

db = 'aihems_api_db'

conn = pymysql.connect(host='aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com',
                       port=3306, user='aihems', passwd='#cslee1234', db=db,
                       charset='utf8')

sql = f"""
SELECT *
FROM AH_USE_LOG_BYMINUTE
WHERE 1=1
AND GATEWAY_ID = '{gateway_id}'
AND DEVICE_ID = case when (   SELECT SCHEDULE_ID
FROM AH_DEVICE_MODEL
WHERE 1=1
AND DEVICE_ID = '{device_id}') is null then '{device_id}' else (   SELECT SCHEDULE_ID
FROM AH_DEVICE_MODEL
WHERE 1=1
AND DEVICE_ID = '{device_id}') end
AND COLLECT_DATE >= DATE_FORMAT( DATE_ADD( STR_TO_DATE( '{date}', '%Y%m%d'),INTERVAL -28 DAY), '%Y%m%d')
"""

df = utils.get_table_from_db(sql)
df = utils.binding_time(df)

schedule = df.pivot_table(values='appliance_status', index=df.index.time, columns=df.index.dayofweek,
                          aggfunc='max')

schedule = schedule.reset_index()

schedule_unpivoted = schedule.melt(id_vars=['index'], var_name='date',
                                   value_name='appliance_status')  # todo: sql 문으로 처리할 수 있도록 수정

schedule_unpivoted.loc[:,
'status_change'] = schedule_unpivoted.appliance_status == schedule_unpivoted.appliance_status.shift(1)

# schedule_unpivoted.columns = ['time', 'date','time', 'appliance_status', 'status_change']

subset = schedule_unpivoted.loc[
    (schedule_unpivoted.status_change == False) | (schedule_unpivoted.index % 1440 == 0), ['date', 'index',
                                                                                           'appliance_status']]

subset.columns = ['dayofweek', 'time', 'appliance_status']

subset.loc[:, 'minutes'] = [x.hour * 60 + x.minute for x in subset.time]

subset.loc[:, 'minutes'] = subset.dayofweek * 1440 + subset.minutes

subset.loc[:, 'duration'] = subset.minutes - subset.minutes.shift(1)
subset.loc[:, 'duration'] = subset.minutes.shift(-1) - subset.minutes

subset = subset.loc[
         ((subset.appliance_status == 0) & (subset.duration < 120) | (subset.time == '00:00:00')) == False, :]
# subset = subset.loc[subset.duration > 120, :]

subset.loc[:, 'status_change'] = subset.appliance_status == subset.appliance_status.shift(1)

# subset = subset.loc[(subset.status_change == False), ['dayofweek', 'time', 'appliance_status']]

subset.loc[:, 'dayofweek'] = [str(x) for x in subset.loc[:, 'dayofweek']]

subset.loc[:, 'time'] = [str(x) for x in subset.loc[:, 'time']]

subset = subset.reset_index(drop=True)

result = subset.to_dict('index')
