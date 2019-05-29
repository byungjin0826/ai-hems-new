from flask import Flask
from flask_restful import Resource, Api, reqparse
import utils

app = Flask(__name__)
api = Api(app)

class PredictElec(Resource):
    """
    gateway_id를 입력받아 현재 달의....
    """
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('gateway_id', type = str)
            parser.add_argument('month', type = int)
            args = parser.parse_args()

            gateway_id = args['gateway_id']
            month = args['month']

            return {'GatewayID': gateway_id, 'Month': month}

        except Exception as e:
            return {'error': str(e)}


api.add_resource(PredictElec, '/elec')

if __name__ == '__main__':
    app.run(port = 5000, debug = True)
