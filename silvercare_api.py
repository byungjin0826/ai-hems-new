from flask import Flask
from flask_restful import Resource, Api, reqparse
import utils
from joblib import load, dump

class SilverCare_Labeling(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('gateway_id', type=str)
            parser.add_argument('collect_date', type=str)
            parser.add_argument('collect_time', type=str)
            args = parser.parse_args()

            gateway_id = args['gateway_id']
            collect_date = args['collect_date']
            collect_time = args['collect_time']

            gateway_list = ['ep18270401', 'ep18270403', 'ep18270414', 'ep18270224']

            if gateway_id == 'ep18270192':
                model_name = 'silvercare_model_2'
            elif gateway_id in gateway_list:
                model_name = 'silvercare_model_1'
            else:
                model_name = 'silvercare_model'

            cur_time = collect_date + collect_time

            sql = f"""
                SELECT *
                FROM AH_USE_LOG_BYMINUTE
                WHERE 1=1
                    AND GATEWAY_ID = '{gateway_id}'
                    AND CONCAT( COLLECT_DATE, COLLECT_TIME) > DATE_FORMAT( DATE_ADD( STR_TO_DATE( '{cur_time}', '%Y%m%d%H%i'), INTERVAL -10 MINUTE), '%Y%m%d%H%i')
                ORDER BY COLLECT_DATE, COLLECT_TIME
                LIMIT 0, 10 
            """

            df = utils.get_table_from_db(sql)

            x, y = utils.split_x_y(df, x_col='energy_diff')
            x = [[i for i in x]]

            model = load(f'./sample_data/joblib/{model_name}.joblib')
            y = model.predict(x)
            y = [int(x) for x in y]

            return {'flag_success': True, 'predicted_status': y}
        except Exception as e:
            return {'flag_success': False, 'error': str(e)}


class SilverCare_test(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('gateway_id', type=str)
            parser.add_argument('device_id', type=str)
            parser.add_argument('collect_date', type=str)
            parser.add_argument('collect_time', type=str)
            parser.add_argument('data', type=list)
            args = parser.parse_args()

            gateway_id = args['gateway_id']
            device_id = args['device_id']
            collect_date = args['collect_date']
            collect_time = args['collect_time']
            data = args['data']

            device_list = ['00158D0001A4590E1', '00158D0001A44CC51', '00158D000151B1E71', '00158D0001A4528D1']

            if device_id == '00158D0001A474EC1':
                model_name = 'silvercare_model_2'
            elif device_id in device_list:
                model_name = 'silvercare_model_1'
            else:
                model_name = 'silvercare_model'

            x = [data]

            model = load(f'./sample_data/joblib/{model_name}.joblib')
            y = model.predict(x)
            y = [int(x) for x in y]

            return {'flag_success': True, 'predicted_status': y}
        except Exception as e:
            return {'flag_success': False, 'error': str(e)}

