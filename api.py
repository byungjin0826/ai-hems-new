from flask import Flask
from flask_restful import Resource, Api, reqparse
import utils
from joblib import load, dump
import datetime
import sklearn as sk
import matplotlib.pyplot as plt
# from silvercare import silvercare_api
import pymysql
import pandas as pd
import numpy as np

plt.style.use('seaborn-whitegrid')

app = Flask(__name__)
api = Api(app)


class PredictElec(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('house_no', type=str)
            parser.add_argument('date', type=str)
            args = parser.parse_args()

            house_no = args['house_no']
            date = args['date']
            # month = args['month']

            sql = f"""
            SELECT   * 
            FROM      AH_USAGE_DAILY_PREDICT
            WHERE      HOUSE_NO = '{house_no}'
             AND      USE_DATE >= DATE_FORMAT( DATE_ADD( STR_TO_DATE( '{date}', '%Y%m%d'),INTERVAL -7 DAY), '%Y%m%d')
             AND      USE_DATE < '{date}'
            ORDER BY USE_DATE
            """

            df = utils.get_table_from_db(sql)

            elec = [x for x in df.use_energy_daily.values[-7:]]

            model = load(f'./sample_data/joblib/usage_daily/{house_no}.joblib')

            y = utils.iter_predict(x=elec, n_iter=31, model=model)

            return {'flag_success': True, 'PREDICT_USE_ENERGY': y}

        except Exception as e:
            return {'flag_success': False, 'error': str(e)}


class Labeling(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('device_id', type=str)
            parser.add_argument('gateway_id', type=str)
            parser.add_argument('collect_date', type=str)
            args = parser.parse_args()

            device_id = args['device_id']
            gateway_id = args['gateway_id']
            collect_date = args['collect_date']

            start = collect_date + '0000'
            end = collect_date + '2359'

            sql = f"""
            SELECT    *
            FROM      AH_USE_LOG_BYMINUTE
            WHERE      1=1
               AND   GATEWAY_ID = '{gateway_id}'
               AND   DEVICE_ID = '{device_id}'
               AND   CONCAT( COLLECT_DATE, COLLECT_TIME) >= DATE_FORMAT( DATE_ADD( STR_TO_DATE( '{start}', '%Y%m%d%H%i'),INTERVAL -20 MINUTE), '%Y%m%d%H%i')
                 AND   CONCAT( COLLECT_DATE, COLLECT_TIME) <= DATE_FORMAT( DATE_ADD( STR_TO_DATE( '{end}', '%Y%m%d%H%i'),INTERVAL 10 MINUTE), '%Y%m%d%H%i')
            ORDER BY COLLECT_DATE, COLLECT_TIME
            """

            df = utils.get_table_from_db(sql)
            print(df.head())
            print('df:', len(df))

            x, y = utils.split_x_y(df, x_col='energy_diff')

            pre = 20
            post = 10
            length = post + pre

            x = [x[i:i + length] for i in range(len(x) - (pre + post))]

            model = load(f'./sample_data/joblib/by_device/{device_id}_labeling.joblib')

            y = model.predict(x)

            y = [int(x) for x in y]

            print(len(y))

            return {'flag_success': True, 'predicted_status': y}

        except Exception as e:
            return {'flag_success': False, 'error': str(e)}


class DeviceMatching(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('device_id', type=str)
            args = parser.parse_args()

            device_id = args['device_id']

            db = 'aihems_api_db'

            conn = pymysql.connect(host='aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com',
                                   port=3306, user='aihems', passwd='#cslee1234', db=db,
                                   charset='utf8')

            sql = f"""
            SELECT
            	DEVICE_ID
            	, avg(ENERGY_DIFF) * 60 ENERGY_DIFF_PER_HOUR
            FROM 
            (SELECT
            	DEVICE_ID
            	, COLLECT_DATE
            	, SUBSTR(COLLECT_TIME, 1, 2) HH
            	, case when POWER >= 5 then 1 else 0 end APPLIANCE_STATUS
            	, ENERGY_DIFF
            FROM 
            	AH_USE_LOG_BYMINUTE
            WHERE 1=1
            AND APPLIANCE_STATUS = 1
            AND COLLECT_DATE >= DATE_FORMAT(DATE_ADD(NOW(), INTERVAL -14 DAY), '%Y%m%d')
            AND DEVICE_ID in 	(
            					SELECT DEVICE_ID
            					FROM AH_APPLIANCE_CONNECT
            					WHERE 1=1
            					AND FLAG_DELETE = 'N'
            					AND APPLIANCE_NO in 
            						(SELECT
            							APPLIANCE_NO
            						FROM
            						(SELECT
            							*
            						FROM AH_APPLIANCE
            						WHERE 1=1
            							) tt
            						WHERE 1=1
            						AND APPLIANCE_TYPE = (SELECT APPLIANCE_TYPE
            FROM AH_APPLIANCE
            WHERE 1=1
            AND APPLIANCE_NO = (SELECT
            	APPLIANCE_NO
            FROM AH_APPLIANCE_CONNECT
            WHERE 1=1 
            AND DEVICE_ID = '{device_id}'
            AND FLAG_DELETE = 'N')))
            					)
            AND DEVICE_ID != '{device_id}'
            ) t
            WHERE 1=1
            AND DEVICE_ID in (SELECT DISTINCT MODEL_ID
            FROM AH_DEVICE_MODEL
            WHERE 1=1
            AND MODEL_ID is not null
            AND MODEL_ID != '제외')
            GROUP BY 
            	DEVICE_ID
            """

            df = pd.read_sql(sql, con=conn, index_col='DEVICE_ID')

            sql = f"""
            SELECT 
            	avg(ENERGY_DIFF) * 60
            FROM (SELECT
            	ENERGY_DIFF
            	, case when POWER > 5 then 1 else 0 end APPLIANCE_STATUS
            FROM AH_USE_LOG_BYMINUTE
            WHERE 1=1
            AND DEVICE_ID = '{device_id}'
            AND COLLECT_DATE >= DATE_FORMAT(DATE_ADD(NOW(), INTERVAL -14 DAY), '%Y%m%d')) t
            WHERE APPLIANCE_STATUS = 1
            GROUP BY APPLIANCE_STATUS
            """

            energy_diff = pd.read_sql(sql, con=conn).iloc[0, 0]
            df['diff1'] = np.abs(df.ENERGY_DIFF_PER_HOUR - energy_diff)
            model_id = df.loc[df.diff1 == df.diff1.min(), :].index[0]

            return {'flag_success': True,
                    'device_id': str(device_id),
                    'model_id': str(model_id)}

        except Exception as e:
            return {'flag_success': False, 'error': str(e)}


class AISchedule(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('gateway_id', type=str)
            parser.add_argument('device_id', type=str)
            # parser.add_argument('dayofweek', type=str)
            args = parser.parse_args()
            date = datetime.datetime.now().date().strftime('%Y%m%d')

            gateway_id = args['gateway_id']
            device_id = args['device_id']

            sql = f"""
            SELECT *
            FROM AH_USE_LOG_BYMINUTE
            WHERE 1=1
            AND GATEWAY_ID = '{gateway_id}'
            AND DEVICE_ID = case when (   SELECT SCHEDULE_ID
            FROM AH_DEVICE_MODEL
            WHERE 1=1
            AND DEVICE_ID = '{device_id}') is null then '{device_id}' else (   SELECT SCHEDULE_ID
            FROM AH_DEVICE_MODEL
            WHERE 1=1
            AND DEVICE_ID = '{device_id}') end
            AND COLLECT_DATE >= DATE_FORMAT( DATE_ADD( STR_TO_DATE( '{date}', '%Y%m%d'),INTERVAL -28 DAY), '%Y%m%d')
            """

            df = utils.get_table_from_db(sql)
            df = utils.binding_time(df)

            schedule = df.pivot_table(values='appliance_status', index=df.index.time, columns=df.index.dayofweek,
                                      aggfunc='max')

            schedule = schedule.reset_index()

            schedule_unpivoted = schedule.melt(id_vars=['index'], var_name='date',
                                               value_name='appliance_status')  # todo: sql 문으로 처리할 수 있도록 수정

            schedule_unpivoted.loc[:,
            'status_change'] = schedule_unpivoted.appliance_status == schedule_unpivoted.appliance_status.shift(1)

            # schedule_unpivoted.columns = ['time', 'date','time', 'appliance_status', 'status_change']

            subset = schedule_unpivoted.loc[
                (schedule_unpivoted.status_change == False) | (schedule_unpivoted.index % 1440 == 0), ['date', 'index',
                                                                                                       'appliance_status']]

            subset.columns = ['dayofweek', 'time', 'appliance_status']

            subset.loc[:, 'minutes'] = [x.hour * 60 + x.minute for x in subset.time]

            subset.loc[:, 'minutes'] = subset.dayofweek * 1440 + subset.minutes

            subset.loc[:, 'duration'] = subset.minutes - subset.minutes.shift(1)
            subset.loc[:, 'duration'] = subset.minutes.shift(-1) - subset.minutes

            subset = subset.loc[
                     ((subset.appliance_status == 0) & (subset.duration < 120) | (subset.time == '00:00:00')) == False,:]
            # subset = subset.loc[subset.duration > 120, :]

            subset.loc[:, 'status_change'] = subset.appliance_status == subset.appliance_status.shift(1)

            # subset = subset.loc[(subset.status_change == False), ['dayofweek', 'time', 'appliance_status']]

            subset.loc[:, 'dayofweek'] = [str(x) for x in subset.loc[:, 'dayofweek']]

            subset.loc[:, 'time'] = [str(x) for x in subset.loc[:, 'time']]

            subset = subset.reset_index(drop=True)

            result = subset.to_dict('index')

            return {
                'flag_success': True,
                'device_id': device_id,
                'result': result
            }

        except Exception as e:
            return {'flag_success': False, 'error': str(e)}


class CBL_INFO(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('house_no', type=str)
            parser.add_argument('request_dr_no', type=str)
            args = parser.parse_args()

            # gateway_id = args['gateway_id']

            house_no = args['house_no']
            request_dr_no = args['request_dr_no']
            # house_name = ''

            db = 'aihems_api_db'

            conn = pymysql.connect(host='aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com',
                                   port=3306, user='aihems', passwd='#cslee1234', db=db,
                                   charset='utf8')
            sql = f"""
            SELECT sum(s)/4
            from (	SELECT *
            		FROM (	SELECT 
            					sum(ENERGY_DIFF) s
            				FROM AH_USE_LOG_BYMINUTE
            				WHERE 1=1
            				AND GATEWAY_ID = (
            					SELECT GATEWAY_ID
            					FROM AH_GATEWAY_INSTALL
            					WHERE 1=1
            					AND HOUSE_NO = '{house_no}'
            				)
            				AND COLLECT_DATE >= DATE_FORMAT(DATE_ADD((SELECT START_DATE FROM AH_DR_REQUEST WHERE 1=1 AND REQUEST_DR_NO = '{request_dr_no}'), INTERVAL -5 DAY), '%Y%m%d')
            				AND COLLECT_DATE < (SELECT DATE_FORMAT(START_DATE, '%Y%m%d') FROM AH_DR_REQUEST WHERE 1=1 AND REQUEST_DR_NO = '{request_dr_no}')
            				AND COLLECT_TIME >= (SELECT DATE_FORMAT(START_DATE, '%H%i') FROM AH_DR_REQUEST WHERE 1=1 AND REQUEST_DR_NO = '{request_dr_no}')
            				AND COLLECT_TIME <= (SELECT DATE_FORMAT(END_DATE, '%H%i') FROM AH_DR_REQUEST WHERE 1=1 AND REQUEST_DR_NO = '{request_dr_no}')
            				GROUP BY
            					COLLECT_DATE) t1
            		WHERE 1=1
            		ORDER BY s desc
            		limit 4) t2
            """

            cbl = pd.read_sql(sql, con=conn).iloc[0, 0]

            if cbl <= 500:
                reduction_energy = cbl * 0.3
            elif cbl <= 1500:
                reduction_energy = cbl * 0.15 + 75
            else:
                reduction_energy = 300

            conn.close

            return {'flag_success': True, 'cbl': cbl, 'reduction_energy': reduction_energy}

        except Exception as e:
            return {'flag_success': False, 'error': str(e)}


class DR_RECOMMEND(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('house_no', type=str)
            parser.add_argument('request_dr_no', type=str)
            args = parser.parse_args()

            # gateway_id = args['gateway_id']

            house_no = args['house_no']
            request_dr_no = args['request_dr_no']
            # house_name = ''

            db = 'aihems_api_db'

            conn = pymysql.connect(host='aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com',
                                   port=3306, user='aihems', passwd='#cslee1234', db=db,
                                   charset='utf8')
            sql = f"""
            SELECT sum(s)/4
            from (	SELECT *
            		FROM (	SELECT 
            					sum(ENERGY_DIFF) s
            				FROM AH_USE_LOG_BYMINUTE
            				WHERE 1=1
            				AND GATEWAY_ID = (
            					SELECT GATEWAY_ID
            					FROM AH_GATEWAY_INSTALL
            					WHERE 1=1
            					AND HOUSE_NO = '{house_no}'
            				)
            				AND COLLECT_DATE >= DATE_FORMAT(DATE_ADD((SELECT START_DATE FROM AH_DR_REQUEST WHERE 1=1 AND REQUEST_DR_NO = '{request_dr_no}'), INTERVAL -5 DAY), '%Y%m%d')
            				AND COLLECT_DATE < (SELECT DATE_FORMAT(START_DATE, '%Y%m%d') FROM AH_DR_REQUEST WHERE 1=1 AND REQUEST_DR_NO = '{request_dr_no}')
            				AND COLLECT_TIME >= (SELECT DATE_FORMAT(START_DATE, '%H%i') FROM AH_DR_REQUEST WHERE 1=1 AND REQUEST_DR_NO = '{request_dr_no}')
            				AND COLLECT_TIME <= (SELECT DATE_FORMAT(END_DATE, '%H%i') FROM AH_DR_REQUEST WHERE 1=1 AND REQUEST_DR_NO = '{request_dr_no}')
            				GROUP BY
            					COLLECT_DATE) t1
            		WHERE 1=1
            		ORDER BY s desc
            		limit 4) t2
            """

            cbl = pd.read_sql(sql, con=conn).iloc[0, 0]

            if cbl <= 500:
                reduction_energy = cbl * 0.3
            elif cbl <= 1500:
                reduction_energy = cbl * 0.15 + 75
            else:
                reduction_energy = 300

            sql = f"""
SELECT
	DEVICE_ID
	, DEVICE_NAME
	, FREQUENCY
	, WAIT_ENERGY_AVG
	, USE_ENERGY_AVG
    , FLAG_USE_AI
	, STATUS
	, ONOFF
	, ENERGY
	, USE_ENERGY_AVG * (SELECT TIMESTAMPDIFF(MINUTE, START_DATE, END_DATE) FROM AH_DR_REQUEST WHERE REQUEST_DR_NO = '{request_dr_no}') as ENERGY_SUM
FROM
(SELECT
	FR.DEVICE_ID
	, T.DEVICE_NAME
	, FR.FREQUENCY
	, (case when HT.WAIT_ENERGY_AVG is null then 0 else HT.WAIT_ENERGY_AVG end) WAIT_ENERGY_AVG
	, (case when HT.USE_ENERGY_AVG is null then 0 else HT.USE_ENERGY_AVG end) USE_ENERGY_AVG
	, T.STATUS
	, T.ONOFF
	, case 
	when (case when T.STATUS = 1 then HT.USE_ENERGY_AVG else HT.WAIT_ENERGY_AVG end) is null then 0
	else (case when T.STATUS = 1 then HT.USE_ENERGY_AVG else HT.WAIT_ENERGY_AVG end) end as ENERGY
    , case when FLAG_USE_AI = 'Y' then 1 else 0 end FLAG_USE_AI
FROM (
	SELECT 
		DEVICE_ID
		, sum(APPLIANCE_STATUS)
		, case when sum(APPLIANCE_STATUS) is null then 0 else sum(APPLIANCE_STATUS) end FREQUENCY
	FROM
		(SELECT 
			COLLECT_DATE
			, DEVICE_ID
			, max(APPLIANCE_STATUS) APPLIANCE_STATUS
		FROM AH_USE_LOG_BYMINUTE
		WHERE 1=1
		AND GATEWAY_ID = (
			SELECT GATEWAY_ID
			FROM AH_GATEWAY_INSTALL
			WHERE 1=1
			AND HOUSE_NO = '{house_no}'
		)
		AND DAYOFWEEK(COLLECT_DATE) = (SELECT DAYOFWEEK(DATE_FORMAT(START_DATE, '%Y%m%d')) FROM AH_DR_REQUEST WHERE REQUEST_DR_NO = '{request_dr_no}')
		AND COLLECT_TIME >= (SELECT DATE_FORMAT(START_DATE, '%H%i') FROM AH_DR_REQUEST WHERE REQUEST_DR_NO = '{request_dr_no}')
		AND COLLECT_TIME <= (SELECT DATE_FORMAT(END_DATE, '%H%i') FROM AH_DR_REQUEST WHERE REQUEST_DR_NO = '{request_dr_no}')
		GROUP BY
			COLLECT_DATE
			, DEVICE_ID) t
		GROUP BY
			DEVICE_ID
		) FR
	INNER JOIN 
		(SELECT
			DEVICE_ID
			, sum(WAIT_ENERGY)/sum(WAIT_TIME) WAIT_ENERGY_AVG
			, sum(USE_ENERGY)/sum(USE_TIME) USE_ENERGY_AVG
		FROM AH_DEVICE_ENERGY_HISTORY
		WHERE 1=1
		AND GATEWAY_ID = (
			SELECT GATEWAY_ID
			FROM AH_GATEWAY_INSTALL
			WHERE 1=1
			AND HOUSE_NO = '{house_no}')
		GROUP BY
			DEVICE_ID) HT
	ON FR.DEVICE_ID = HT.DEVICE_ID
	INNER JOIN 
		(SELECT 
			gateway_id
			, device_id
			, device_name
		    , sum(onoff) onoff_sum
		    , count(onoff) onoff_count
		    , avg(POWER) power_avg
		    , case when sum(onoff) > 2.5 then 1 else 0 end onoff
			, case when avg(POWER) > 0.5 then 1 else 0 end status -- 조정필요
		FROM aihems_service_db.AH_LOG_SOCKET
		WHERE 1=1
		AND GATEWAY_ID = (SELECT GATEWAY_ID FROM aihems_api_db.AH_GATEWAY_INSTALL WHERE 1=1 AND HOUSE_NO = '{house_no}')
		AND COLLECT_DATE = (SELECT DATE_FORMAT(NOW(), '%Y%m%d') FROM DUAL)
		-- DATE_FORMAT(DATE_ADD((SELECT START_DATE FROM AH_DR_REQUEST WHERE 1=1 AND REQUEST_DR_NO = '{request_dr_no}'), INTERVAL -5 DAY), '%Y%m%d')
		AND COLLECT_TIME >= DATE_FORMAT(DATE_ADD(DATE_ADD(NOW(), INTERVAL 9 HOUR), INTERVAL -5 MINUTE), '%H%i')
		GROUP BY
			gateway_id
			, device_id
			, device_name) T on (FR.DEVICE_ID = T.DEVICE_ID)
    INNER JOIN
		(SELECT
			DEVICE_ID
			, FLAG_USE_AI
		FROM AH_DEVICE) QQ ON FR.DEVICE_ID = QQ.DEVICE_ID
	) FF
ORDER BY
    FLAG_USE_AI asc
	, STATUS desc
	, ONOFF desc
	, FREQUENCY desc
	, ENERGY asc
            """

            status = pd.read_sql(sql, con=conn)
            conn.close()
            status['ENERGY_CUMSUM'] = status.ENERGY_SUM.cumsum()
            status['USE_MAX'] = cbl - reduction_energy
            status['PERMISSION'] = [x < cbl - reduction_energy for x in status.ENERGY_CUMSUM]

            subset = status.loc[:, ['DEVICE_ID', 'PERMISSION']]

            ix = max(status.loc[status.FLAG_USE_AI == 0,].index)
            dr_success = subset.iloc[ix, 1]

            if dr_success:
                recommendation = subset.to_dict('index')

            else:
                recommendation = 0

            return {'flag_success': True,
                    'dr_success': str(dr_success),
                    'cbl': str(cbl),
                    'reduction_energy': str(reduction_energy),
                    'recommendation': recommendation}

        except Exception as e:
            return {'flag_success': False, 'error': str(e)}

class DR_DECISION(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('', type = str)
            parser.add_argument('reque', type = str)

            return(0)

        except Exception as e:
            return {'flag_success': False, 'error': str(e)}




class Make_Model_Elec(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('house_no', type=str)
            args = parser.parse_args()

            house_no = args['house_no']
            today = datetime.datetime.now().strftime('%Y%m%d')

            sql = f"""
SELECT *
FROM AH_USAGE_DAILY_PREDICT
WHERE 1=1
AND HOUSE_NO = {house_no}
AND USE_DATE >= DATE_FORMAT( DATE_ADD( STR_TO_DATE( '{today}', '%Y%m%d'),INTERVAL -28 DAY), '%Y%m%d')
"""

            df = utils.get_table_from_db(sql)

            df.loc[df.use_energy_daily.isnull(), 'use_energy_daily'] = 0

            x, y = utils.split_x_y(df, x_col='use_energy_daily', y_col='use_energy_daily')

            x, y = utils.sliding_window_transform(x, y, step_size=7, lag=0)

            x = x[6:-1]

            y = y[7:]

            """
            random forest
            linear regression
            ridge regression
            lasso regression
            """

            model, param = utils.select_regression_model('linear regression')

            gs = sk.model_selection.GridSearchCV(estimator=model,
                                                 param_grid=param,
                                                 cv=5,
                                                 n_jobs=-1)

            gs.fit(x, y)

            print(gs.best_score_)

            dump(gs, f'./sample_data/joblib/usage_daily/{house_no}.joblib')

            return {'flag_success': True, 'best_score': str(gs.best_score_)}

        except Exception as e:
            return {'flag_success': False, 'error': str(e)}


class Make_Model_Status(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('device_id', type=str)
            args = parser.parse_args()

            lag = 10

            device_id = args['device_id']

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
	AND DEVICE_ID = '{device_id}'
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

            return {'flag_success': True, 'dump_path': str(dump_path), 'score': str(gs.best_score_)}

        except Exception as e:
            return {'flag_success': False, 'error': str(e)}


api.add_resource(PredictElec, '/elec')
api.add_resource(Labeling, '/label')
api.add_resource(DeviceMatching, '/device_matching')
api.add_resource(CBL_INFO, '/cbl_info')
api.add_resource(AISchedule, '/schedule')
api.add_resource(DR_RECOMMEND, '/dr_recommendation')
api.add_resource(Make_Model_Elec, '/make_model_elec')
api.add_resource(Make_Model_Status, '/make_model_status')

# api.add_resource(silvercare_api.SilverCare_Labeling, '/silver_label')


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=5000, debug=True)
    # app.run(host='127.0.0.1', port=5000, debug=True)

