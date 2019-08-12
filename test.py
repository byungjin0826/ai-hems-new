from utils import *

import datetime

lag = 10

sql = f"""
SELECT distinct device_id
FROM AH_USE_LOG_BYMINUTE_LABELED_sbj
"""

df = get_table_from_db(sql)

for i in df.iterrows():
    device_id = i[1][0]

    print(device_id)

    sql = f"""
SELECT *
FROM AH_USE_LOG_BYMINUTE_LABELED_sbj
WHERE 1=1
AND device_id = '{device_id}'
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

    dump(gs, f'./sample_data/test/{device_id}_labeling.joblib') # 저장