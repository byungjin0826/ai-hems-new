from utils import *

lag = 10

device_id = '00158D000151B3061'
# 00158D000151B4441: 공기청정기
# 00158D0001524BC71 : 셋톱


sql = f"""
SELECT *
FROM AH_USE_LOG_BYMINUTE_LABELED_cc
WHERE 1=1
AND GATEWAY_ID = (
	SELECT GATEWAY_ID
	FROM AH_GATEWAY
	WHERE GATEWAY_NAME = '박재훈'
	)
AND DEVICE_ID = '{device_id}'
"""

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

dump(gs, f'./sample_data/joblib/{device_id}_labeling.joblib') # 저장