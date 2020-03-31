#-*- coding:utf-8 -*-

from flask import Flask, request, jsonify
from flask_restful import Api
import common.data_load as dl
import common.ai as ai
import common.dr as dr
import common.model_training as mt
import json
import settings
import pandas as pd


app = Flask(__name__)
api = Api(app)


@app.route('/elec/', methods=['GET', 'POST'])
def predict_elec():
    try:
        house_no = request.json['house_no']
        date = request.json['date']
        # print(f'Input: {house_no}, date')

        y = dl.predict_elec(house_no=house_no, date=date)
        y = [str(x) for x in y]
        # print(f'output:{y}')
        response = jsonify({'flag_success': True, 'PREDICT_USE_ENERGY': y})
        response.status_code = 200
        # return jsonify('0')
        return jsonify({'flag_success': True, 'predict_use_energy': y})  # 기존 대문자

    except Exception as e:
        return jsonify({'flag_success': False, 'error': e})


@app.route('/label/', methods=['GET', 'POST'])
def labeling_by_device_for_one_day():
    try:
        device_id = request.json['device_id']
        gateway_id = request.json['gateway_id']
        collect_date = request.json['collect_date']

        y = dl.labeling(device_id=device_id, gateway_id=gateway_id, collect_date=collect_date)
        print(y)
        return jsonify({'flag_success': True, 'predicted_status': y})

    except Exception as e:
        return jsonify({'flag_success': False, 'error': str(e)})


@app.route('/schedule/', methods=['GET', 'POST'])
def ai_schedule():
    try:
        gateway_id = request.json['gateway_id']
        device_id = request.json['device_id']

        weeks = dl.check_weeks(gateway_id=gateway_id, device_id=device_id)

        if weeks >= 4:
            df = ai.get_ai_schedule(device_id=device_id, gateway_id=gateway_id)
            df.columns = ['dayofweek', 'time', 'end', 'duration', 'appliance_status']
            result = df.loc[:, ['dayofweek', 'time', 'appliance_status']].to_dict('index')

        return jsonify({'flag_success': True, 'device_id': device_id, 'result': result})

    except Exception as e:
        return jsonify({'flag_success': False, 'error': str(e)})


@app.route('/cbl_info/', methods=['GET', 'POST'])
def cbl_info():
    try:
        house_no = request.json['house_no']
        request_dr_no = request.json['request_dr_no']

        cbl, reduction_energy = dr.cbl_info(house_no=house_no, request_dr_no=request_dr_no)

        return jsonify({'flag_success': True, 'cbl': cbl, 'reduction_energy': reduction_energy})

    except Exception as e:
        return jsonify({'flag_success': False, 'error': str(e)})


@app.route('/dr_recommendation/', methods=['GET', 'POST'])
def dr_recommendation():
    try:
        house_no = request.json['house_no']
        request_dr_no = request.json['request_dr_no']

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
                		limit 4) t2d
                """

        with settings.open_db_connection() as conn:
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
    	, USE_ENERGY_AVG * (SELECT TIMESTAMPDIFF(MINUTE, START_DATE, END_DATE) FROM AH_DR_REQUEST WHERE REQUEST_DR_NO = '{request_dr_no}') as ENERGY_SUM_USE
        , WAIT_ENERGY_AVG * (SELECT TIMESTAMPDIFF(MINUTE, START_DATE, END_DATE) FROM AH_DR_REQUEST WHERE REQUEST_DR_NO = '{request_dr_no}') as ENERGY_SUM_WAIT
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
    WHERE 1=1 
    AND FLAG_USE_AI != 0
    ORDER BY
        FLAG_USE_AI asc
    	, STATUS desc
    	, ONOFF desc
    	, FREQUENCY desc
    	, ENERGY asc
                """

        with settings.open_db_connection() as conn:
            status = pd.read_sql(sql, con=conn)

        status['ENERGY_SUM'] = [max([x[1][0], x[1][1]]) for x in
                                status.loc[:, ['ENERGY_SUM_WAIT', 'ENERGY_SUM_USE']].iterrows()]
        status['ENERGY_CUMSUM'] = status.ENERGY_SUM.cumsum()
        status['USE_MAX'] = cbl - reduction_energy
        status['PERMISSION'] = [x < cbl - reduction_energy for x in status.ENERGY_CUMSUM]

        subset = status.loc[:, ['DEVICE_ID', 'ENERGY_SUM', 'PERMISSION']]
        subset.ENERGY_SUM = [str(x) for x in subset.ENERGY_SUM]

        dr_success = subset.iloc[0, 2]

        if dr_success:
            recommendation = subset.to_dict('index')
            dr_success = True

        else:
            recommendation = 0
            dr_success = False

        print(subset)

        return jsonify({'flag_success': True,
                'dr_success': dr_success,
                'cbl': str(cbl),
                'reduction_energy': str(reduction_energy),
                'recommendation': recommendation})

    except Exception as e:
        return jsonify({'flag_success': False, 'error': str(e)})


@app.route('/make_model_elec/', methods=['GET', 'POST'])
def make_model_elec():
    try:
        house_no = request.json['house_no']
        score = mt.make_model_elec(house_no=house_no)
        return jsonify({'flag_success': True, 'best_score': str(score)})

    except Exception as e:
        return jsonify({'flag_success': False, 'error': str(e)})


@app.route('/make_model_status/', methods=['GET', 'POST'])
def make_model_status():
    try:
        device_id = request.json['device_id']
        dump_path, best_score = mt.make_model_status(device_id=device_id)
        return jsonify({'flag_success': True, 'dump_path': str(dump_path), 'score': str(best_score)})

    except Exception as e:
        return jsonify({'flag_success': False, 'error': str(e)})


@app.route('/test/', methods=['GET', 'POST'])
def hello():
    name = request.json['name']
    return {'name': name}


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    # app.run(host='127.0.0.1', debug=True)
    # house_no = '20180810000008'
    # date = '20191106'
    # y = dl.predict_elec(house_no=house_no, date=date)
