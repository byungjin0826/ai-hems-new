from flask import Flask, request
# import json
from flask_restful import Resource, Api, reqparse
import utils
from joblib import load, dump

class SilverCare_Labeling(Resource):
    def post(self):
        try:
            json_data = request.get_json(silent=True, cache=False)
            gateway_id = json_data['gateway_id']
            device_id = json_data['device_id']
            collect_date = json_data['collect_date']
            collect_time = json_data['collect_time']
            data = json_data['data']

            device_list = ['00158D0001A4590E1', '00158D0001A44CC51', '00158D000151B1E71', '00158D0001A4528D1']

            if device_id == '00158D0001A474EC1':
                model_name = 'silvercare_model_2'
            elif device_id in device_list:
                model_name = 'silvercare_model_1'
            else:
                model_name = 'silvercare_model'

            x = [data]

            model = load(f'./sample_data/joblib/silvercare/{model_name}.joblib')
            y = model.predict(x)
            y = str(y[0])

            return {'flag_success': True, 'predicted_status': y}
        except Exception as e:
            return {'flag_success': False, 'error': str(e)}



