from utils import *

# query = """
# SELECT distinct(device_id)
# FROM AH_USE_LOG_BYMINUTE_LABELED_cc
# """
# device_address_list = get_table_from_db(query, db='aihems_api_db')

# 일단 모델을 로컬에 저장
member_name = input('member name : ')
appliance_name = input('appliance name : ')
# device_address = input('device_address:')

device_address = search_device_address(member_name, appliance_name)

print(device_address)

# device_address = device_address[:-1]
# for device_address in device_address_list.device_id:
lag = 10

sql = """
SELECT *
FROM AH_USE_LOG_BYMINUTE_LABELED_cc
WHERE 1=1
"""

# gateway_id = ""
device_address_condition = f"AND device_id = '{device_address}'"
# gateway_id_condition = f"AND gateway_id = {gateway_id}"
sql += device_address_condition
# sql += gateway_id_conditon

df = get_table_from_db(sql, db='aihems_api_db')

x, y = split_x_y(df, x_col='energy_diff', y_col='appliance_status')

x, y = sliding_window_transform(x,y,lag=lag,step_size=30)

model, params = select_classification_model('random forest')

gs = sk.model_selection.GridSearchCV(estimator=model,
                                     param_grid=params,
                                     cv=5,
                                     scoring='accuracy',
                                     n_jobs=-1)

gs.fit(x, y)

print(round(gs.best_score_*100, 2), '%', sep = '')

df = df.iloc[:-lag]

df.loc[:, 'appliance_status_predicted'] = gs.predict(x)
# df['appliance_status'] = gs.predict(x)
# dump(gs, './sample_data/joblib/'+device_address+'_labeling.joblib') # 저장