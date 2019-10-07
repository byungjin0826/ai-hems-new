from utils import *

# sql = f'''
# SELECT *
# FROM SC_DEVICE_LOG
# WHERE HOUSE_NO = '20190805000009'
# AND COLLECT_DATE = '20190910'
# '''
collect_date = "20190808"
start = collect_date + '0000'
end = collect_date + '2359'
gateway_id = 'ep18270334'
device_id = "00158D0001A427D81"
sql = f"""
SELECT    *
FROM      AH_USE_LOG_BYMINUTE
WHERE      1=1
   AND   GATEWAY_ID = '{gateway_id}'
   AND   DEVICE_ID = '{device_id}'
   AND   COLLECT_DATE = '{collect_date}'
ORDER BY COLLECT_DATE, COLLECT_TIME
"""
df = get_table_from_db(sql, db='aihems_api_db')

# df['appliance_status'] = 0
# pre_arr = np.ndarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
"""
start = 20190806 2349
end = 20190918 2222e
"""
x, y = split_x_y(df, x_col='energy_diff')
pre_arr = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
post_arr = np.array([0,0,0,0,0,0,0,0,0,0])
x = np.concatenate((pre_arr, x), axis=None)
x = np.concatenate((x, post_arr), axis=None)
lag = 10
pre = 20
post = 10
length = post + pre
#
x = [x[i:i + length] for i in range(len(x) - (pre + post))]
#
model = load('./sample_data/joblib/by_device/00158D0001A44CC51_labeling.joblib')
# # model = load(f'./sample_data/joblib/by_device/{device_id}_labeling.joblib')
#
#
# """
# x = 53037
# y = 53037
# """
#
# # x, y = sliding_window_transform(x, y, lag=lag, step_size=30)
# """
# x = 53034
# y = 53034
# # """
# # x = [[1,1,1,1,1,1,0,0,0,0]]
#
y = model.predict(x)
# df = df.iloc[:-length]
# # y = [int(x) for x in y]
# """
# start = 20190806 2349
# end = 20190918 2219
# """
df.loc[:, 'appliance_status'] = y