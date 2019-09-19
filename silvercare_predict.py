from utils import *

sql = f'''
SELECT *
FROM AH_USE_LOG_BYMINUTE
WHERE GATEWAY_ID = 'ep18270403'
AND COLLECT_DATE = '20190918'
'''

df = get_table_from_db(sql, db='aihems_api_db')

df['appliance_status'] = 0

"""
start = 20190806 2349
end = 20190918 2222e
"""

lag = 4

model = load('./sample_data/joblib/silvercare_model_2.joblib')

x, y = split_x_y(df, x_col='energy_diff')

"""
x = 53037
y = 53037
"""

x, y = sliding_window_transform(x, y, lag=lag, step_size=10)
"""
x = 53034
y = 53034
"""
y = model.predict(x)
df = df.iloc[:-lag]

"""
start = 20190806 2349
end = 20190918 2219
"""
df.loc[:, 'appliance_status'] = y