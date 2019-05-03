from utils import *

# 일단 모델을 로컬에 저장

device_address = '00158D000151B1F9'

sql = """
SELECT *
FROM AH_USE_LOG_BYMINUTE_LABLED
WHERE 1=1
"""

# gateway_id = ""
device_address_condition = f"AND device_address = '{device_address}'"

sql += device_address_condition

df = get_table_from_db(sql, db='aihems_api_db')

x, y = split_x_y(df, x_col='ENERGY_DIFF', y_col='APPLIANCE_STATUS')

x, y = sliding_window_transform(x,y,lag=3)

model, params = select_classification_model('random forest')

gs = sk.model_selection.GridSearchCV(estimator=model,
                                     param_grid=params,
                                     cv=5,
                                     scoring='accuracy')

gs.fit(x, y)

dump(gs, './sample_data/joblib/test.joblib')