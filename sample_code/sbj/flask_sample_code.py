from flask import Flask
from flask_restful import Resource, Api
from flask_restful import reqparse

app = Flask(__name__)
api = Api(app)

class CreateUser(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('email', type=str) # parameter 추가
            parser.add_argument('user_name', type=str)
            parser.add_argument('password', type=str)
            args = parser.parse_args()

            _userEmail = args['email']
            _userName = args['user_name']
            _userPassword = args['password']
            return {'Email': args['email'], 'UserName': args['user_name'], 'Password': args['password']}
        except Exception as e:
            return {'error': str(e)}

class Test(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('test', type=str)
            args = parser.parse_args()

            __userTest = args['test']
            return {'Test':args['test']}
        except Exception as e:
            return {'error': str(e)}

api.add_resource(CreateUser, '/user')
api.add_resource(Test, '/test')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)