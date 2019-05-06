from utils import *

# 일단 모델을 로컬에 저장
member_name = input('member name : ')
appliance_name = input('appliance name : ')
# device_address = input('device_address:')

device_address = search_device_address(member_name, appliance_name)

gs = load('./sample_data/joblib/'+device_address+'.joblib')

lag = 10

sql = """
SELECT *
FROM AH_USE_LOG_BYMINUTE_LABLED
WHERE 1=1
"""

# gateway_id = ""
device_address_condition = f"AND device_address = '{device_address}'"
# gateway_id_condition = f"AND gateway_id = {gateway_id}"

sql += device_address_condition
# sql += gateway_id_conditon

df = get_table_from_db(sql, db='aihems_service_db')

x, y = split_x_y(df, x_col='energy_diff', y_col='appliance_status')

x, y = sliding_window_transform(x,y,lag=lag,step_size=30)

Y = gs.predict(x)


