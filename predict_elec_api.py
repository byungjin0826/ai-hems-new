from flask import Flask
from flask_restful import Resource, Api, reqparse
import utils
import pymysql
from joblib import load

"""
# 뭔가 일관성이 있으면 좋을듯...
# house_no: 20190325000001
# gateway_id: ep18270236

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

app = Flask(__name__)
api = Api(app)


class PredictElec(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('house_name', type=str)
            # parser.add_argument('month', type=str)
            args = parser.parse_args()

            house_name = args['house_name']
            # month = args['month']
            
            house_no = utils.get_house_no(house_name)
            gateway_id = utils.get_gateway_id(house_name)

            sql = f"""
            SELECT *
            FROM AH_USAGE_DAILY_PREDICT_test
            WHERE 1=1
            AND house_no = '{house_no}'
            """

            df = utils.get_table_from_db(sql)

            elec = [x for x in df.use_energy.values[-7:]]

            model = load(f'./sample_data/{gateway_id}.joblib')

            y = utils.iter_predict(x=elec, n_iter=30, model=model)

            return {'PREDICT_USE_ENERGY': y}

        except Exception as e:
            return {'error': str(e)}


api.add_resource(PredictElec, '/elec')

if __name__ == '__main__':
    app.run(port=5000, debug=True)