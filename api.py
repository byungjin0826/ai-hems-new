from flask import Flask
from flask_restful import Resource, Api, reqparse
import utils
import pymysql
from joblib import load
import datetime
import pandas as pd

"""
# 뭔가 일관성이 있으면 좋을듯...
# house_no: 20190325000001
# gateway_id: ep18270236
# 기준일자 추가

1. 실행하면 한 달 치 전력 예측값을 update?
    - 
2. 실행하면 한 달 치 전력 값이 list 형태로 반환되게?
    - 입력 받을 값: gateway_id, month
    - model 7일치...
    - 전달 7일치를 포함하는 데이터를 로드
    - 없는 경우에는 예측 값을 사용하고 있는 경우에는...
3. 입력한 날짜의 데이터만 한 개 나오게?
4. 매일 하루에 한번씩 사용하는 것을 가정
    - 날짜랑 house_no를 인풋으로 받아서
    - 한개의 값만 나오게...
    - 현재일로부터 6일전까지 총 7일간의 데이터로 다음날 것을 예측
    
"""

"""
# AI control
input
    - service
    - command
    - house_no
    - week
output
    - service
    - command
    - flag_success
    - message
    - result
"""

"""
Labeling API
input
    - gateway_id
    - device_id
    - collect_date
output
    - flag
    - appliance_status
"""

"""
DR
input
    - house_no
    - 시간
output
    - flag
    - cbl
    - energy
    - elec_list: dictionary 형식으로, True가 허용, False가 제한
        { 'TV': True
        'Aircon':False}
"""

"""
DR_now
input
    - 
output
    - 
"""

"""
DR_change_list
input
    - elec_change_list: dictionary 형식으로

output
    - 
"""


# todo: 기준일자 추가
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

            elec = [x for x in df.use_energy.values[-7:]]

            model = load(f'./sample_data/{house_no}.joblib')

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

            sql = f"""
            SELECT    *
            FROM      AH_USE_LOG_BYMINUTE
            WHERE      1=1
               AND   GATEWAY_ID = '{gateway_id}'
               AND   DEVICE_ID = '{device_id}'
               AND   CONCAT( COLLECT_DATE, COLLECT_TIME) > DATE_FORMAT( DATE_ADD( STR_TO_DATE( '{collect_date}', '%Y%m%d%H%i'),INTERVAL -20 MINUTE), '%Y%m%d%H%i')
                 AND   CONCAT( COLLECT_DATE, COLLECT_TIME) <= DATE_FORMAT( DATE_ADD( STR_TO_DATE( '{collect_date}', '%Y%m%d%H%i'),INTERVAL 10 MINUTE), '%Y%m%d%H%i')
            ORDER BY COLLECT_DATE, COLLECT_TIME
            """

            df = utils.get_table_from_db(sql)

            model = load(f'./sample_data/joblib/by_device/{device_id}_labeling.joblib')

            y = model.predict([df.energy_diff.values])

            return {'flag_success': True, 'predicted_status': y.item()}

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

            gateway_id = args['gateway_id']
            device_id = args['device_id']

            sql = f"""
            SELECT *
            FROM AH_USE_LOG_BYMINUTE_201904
            WHERE 1=1
            AND GATEWAY_ID = '{gateway_id}'
            AND DEVICE_ID = '{device_id}'
            """

            df = utils.get_table_from_db(sql)
            df = utils.binding_time(df)

            schedule = df.pivot_table(values='appliance_status', index=df.index.time, columns=df.index.dayofweek,
                                      aggfunc='max')

            schedule = schedule.reset_index()

            schedule_unpivoted = schedule.melt(id_vars=['index'], var_name='date', value_name='appliance_status')

            schedule_unpivoted.loc[:,
            'status_change'] = schedule_unpivoted.appliance_status == schedule_unpivoted.appliance_status.shift(1)

            subset = schedule_unpivoted.loc[
                (schedule_unpivoted.status_change == False), ['date', 'index', 'appliance_status']]

            subset.columns = ['dayofweek', 'time', 'appliance_status']

            subset.loc[:, 'dayofweek'] = [str(x) for x in subset.loc[:, 'dayofweek']]

            subset.loc[:, 'time'] = [str(x) for x in subset.loc[:, 'time']]

            subset = subset.reset_index(drop=True)

            result = subset.to_dict('index')

            return {
                'flag_success': True,
                'device_id':device_id,
                'result': result
            }

        except Exception as e:
            return {'flag_success': False, 'error':str(e)}

class CBL(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('house_no', type=str)
            parser.add_argument('dr_time', type=str)
            args = parser.parse_args()

            device_id = args['device_id']
            gateway_id = args['gateway_id']
            collect_date = args['collect_date']

            elec_list = utils.calc_number_of_time_use()

            elec_list
            y = 0
            return {'flag_success': True, 'aim': y, 'elec_list': elec_list}

        except Exception as e:
            return {'flag_success': False, 'error': str(e)}


class Test(Resource):
    def post(self):
        try:
            import pandas as pd
            import numpy as np

            arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]

            tuples = list(zip(*arrays))

            index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])

            s = pd.Series(np.random.randn(8), index=index)

            return {'output': arrays}

        except Exception as e:
            return {'error':str(e)}


api.add_resource(PredictElec, '/elec')
api.add_resource(Labeling, '/label')
api.add_resource(CBL, '/dr')
api.add_resource(AISchedule, '/schedule')
api.add_resource(Test, '/test')

if __name__ == '__main__':
    app.run(host = '127.0.0.1', port=5000, debug=True)