from utils import *

# 일단 모델을 로컬에 저장

device_address = input('device_address:')
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

df = get_table_from_db(sql, db='aihems_api_db')

x, y = split_x_y(df, x_col='ENERGY_DIFF', y_col='APPLIANCE_STATUS')

x, y = sliding_window_transform(x,y,lag=lag,step_size=30)

model, params = select_classification_model('random forest')

gs = sk.model_selection.GridSearchCV(estimator=model,
                                     param_grid=params,
                                     cv=5,
                                     scoring='accuracy')

gs.fit(x, y)

print(round(gs.best_score_*100, 2), '%', sep = '')

df = df.iloc[:-lag]

df.loc[:, 'appliance_status_predicted'] = gs.predict(x)

# dump(gs, './sample_data/joblib/test.joblib') # 저장