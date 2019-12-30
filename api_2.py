#-*- coding:utf-8 -*-

from flask import Flask, request, jsonify
from flask_restful import Api
import common.data_load as dl
import common.ai as ai
import common.dr as dr
import common.model_training as mt

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

        return jsonify({'flag_success': True, 'PREDICT_USE_ENERGY': y})
        # return {'flag_success': True, 'predict_use_energy': y}  # 기존 대문자

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

        cbl, reduction_energy = dr.cbl_info(house_no=house_no, request_dr_no=request_dr_no)
        dr_type, duration = dl.get_dr_info(request_dr_no=request_dr_no)

        return 0

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
