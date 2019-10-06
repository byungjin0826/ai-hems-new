import pandas as pd
import utils
from joblib import load, dump
sql = f"""
SELECT *
FROM SC_DEVICE_LOG
WHERE HOUSE_NO = '20190805000030'
"""
df = utils.get_table_from_db(sql, db='silver_service_db')

model_sql = f"""
SELECT HD.HOUSE_NO, DM.MODEL_ID
FROM 
	silver_service_db.SC_HOUSE_DEVICE AS HD
	JOIN
	aihems_api_db.AH_DEVICE_MODEL AS DM
	ON HD.DEVICE_ID = DM.DEVICE_ID
WHERE HD.HOUSE_NO = '20190805000030'
"""

model_id = utils.get_table_from_db(model_sql, db='silver_service_db')
model_name = model_id['model_id'][0]

gs = load(f'./sample_data/joblib/silvercare/{model_name}_labeling.joblib')

lag = 4
x, y = utils.split_x_y(df, x_col='energy_diff', y_col='label_status')
x, y = utils.sliding_window_transform(x, y, lag=lag, step_size=10)

y_pred = gs.predict(x)
df = df.iloc[:-lag]
df.loc[:, 'pred'] = y_pred
