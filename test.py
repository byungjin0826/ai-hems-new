from utils import *

import datetime
import utils


# device_id = args['device_id']

device_id = '000D6F000E4B27B81'
lag = 10

sql = f"""
SELECT
	GATEWAY_ID
	, DEVICE_ID
	, COLLECT_DATE
	, COLLECT_TIME
	, QUALITY
	, ONOFF
	, ENERGY
	, ENERGY_DIFF
	, case when APPLIANCE_STATUS is null then 0 else APPLIANCE_STATUS end APPLIANCE_STATUS
	, CREATE_DATE
FROM
	AH_USE_LOG_BYMINUTE_LABELED_sbj
WHERE
	1 = 1
-- 	AND GATEWAY_ID = 'ep18270363'
	AND DEVICE_ID = '000D6F000E4B27B81'
	AND COLLECT_DATE in (
		SELECT
			t1.COLLECT_DATE
		FROM
			(SELECT
				COLLECT_DATE
				, sum(APPLIANCE_STATUS) APPLIANCE_STATUS_SUM
			FROM 
				AH_USE_LOG_BYMINUTE_LABELED_sbj
			GROUP by
				COLLECT_DATE) t1
		WHERE 1=1
		AND t1.APPLIANCE_STATUS_SUM is not null)
-- 	AND CONCAT( COLLECT_DATE, COLLECT_TIME) >= DATE_FORMAT( DATE_ADD( STR_TO_DATE( '201909110000', '%Y%m%d%H%i'), INTERVAL -20 MINUTE), '%Y%m%d%H%i')
-- 	AND CONCAT( COLLECT_DATE, COLLECT_TIME) <= DATE_FORMAT( DATE_ADD( STR_TO_DATE( '201909112359', '%Y%m%d%H%i'), INTERVAL 10 MINUTE), '%Y%m%d%H%i')
-- ORDER BY
-- 	COLLECT_DATE,
-- 	COLLECT_TIME
"""

df = utils.get_table_from_db(sql, db='aihems_api_db')



x, y = utils.split_x_y(df, x_col='energy_diff', y_col='appliance_status')

x, y = utils.sliding_window_transform(x, y, lag=lag, step_size=30)

model, params = utils.select_classification_model('random forest')

gs = sk.model_selection.GridSearchCV(estimator=model,
                                     param_grid=params,
                                     cv=5,
                                     scoring='accuracy',
                                     n_jobs=-1)

gs.fit(x, y)

gs.best_score_

print(round(gs.best_score_ * 100, 2), '%', sep='')

df = df.iloc[:-lag]

df.loc[:, 'appliance_status_predicted'] = gs.predict(x)
# df['appliance_status'] = gs.predict(x)

dump_path = f'./sample_data/joblib/{device_id}_labeling.joblib'

dump(gs, dump_path)  # 저장