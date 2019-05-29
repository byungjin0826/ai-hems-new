from flask import Flask
from flask_restful import Resource, Api, reqparse
import utils
import pymysql
from joblib import load


# device_id
# datetime
"""
gateway_id: ep18270486
device_id: 000D6F000E4B03C61

"""

gateway_id = 'ep18270486'
device_id = '000D6F000E4B03C61'
collect_date = '201905010022'


app = Flask(__name__)
api = Api(app)


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

            return {'flag_success': True, 'predicted_status': y}

        except Exception as e:
            return {'flag_success': False, 'error': str(e)}


api.add_resource(Labeling, '/label')

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=5000, debug=True)