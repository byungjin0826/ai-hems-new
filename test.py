from utils import *
from datetime import date, timedelta
import os

yesterday = date.today() - timedelta(1)     #하루 전 날짜 계산
yesterday = yesterday.strftime('%Y%m%d')

Path = './sample_data/joblib/'

device_id = '00158D000151B2131'

file_path = Path + device_id + '_labeling.joblib'
if os.path.isfile(file_path):  # model load / 없을경우 continue
    gs = load(file_path)
else:
    print("no model : " + device_id)
sql = f"""
SELECT AUL.*
FROM
AH_USE_LOG_BYMINUTE AS AUL
JOIN
AH_GATEWAY AS AG
ON AUL.GATEWAY_ID = AG.GATEWAY_ID
WHERE 
AUL.DEVICE_ID = '{device_id}'
AND
AUL.COLLECT_DATE = '{yesterday}'
"""
df = get_table_from_db(sql)
df['appliance_status'] = 0

if df.empty:  # 데이터가 없을경우 continue
    print('empty : ' + device_id)

lag = 10
gs = load(file_path)                                                       #모델 로드
x, y = split_x_y(df, x_col='energy_diff', y_col='appliance_status')
x, y = sliding_window_transform(x, y, lag=lag, step_size=30)

df['appliance_status'] = gs.predict(x)                                     #모델 적용

df = df.iloc[:-lag]
df.to_csv('C:/Users/user/Desktop/model_load_result.csv',)