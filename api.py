from flask import Flask
from flask_restful import Resource, Api, reqparse
import utils
import pymysql
from joblib import load
import datetime
import pandas as pd

# todo: 전력 예측 '-값' 나오는 거 모델 해결하기
# todo: DR 개발
# todo:


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

            model = load(f'./sample_data/joblib/usage_daily/{house_no}.joblib')

            x = [3031.2502, 2854.9709, 3115.7338, 4878.9978, 4113.0371, 4156.0059, 5398.0103]

            y = utils.iter_predict(x=x, n_iter=31, model=model)

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


class MakePredictionModel(Resource):
    def post(self):
        try:

            return {'flag_success': True}

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
    # app.run(host = '0.0.0.0', port=5000, debug=True)
    app.run(host = '127.0.0.1', port=5000, debug=True)