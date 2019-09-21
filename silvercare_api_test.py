from flask import Flask
from flask_restful import Resource, Api, reqparse
import utils
from joblib import load, dump
import datetime


            # parser = reqparse.RequestParser()
            # parser.add_argument('device_id', type=str)
            # parser.add_argument('gateway_id', type=str)
            # parser.add_argument('collect_date', type=str)
            # parser.add_argument('collect_time', type=str)
            # args = parser.parse_args()

            # device_id = args['device_id']
            # gateway_id = args['gateway_id']
            # collect_date = args['collect_date']
            # collect_time = args['collect_time']

gateway_id = 'ep18270185'
collect_date = '20190810'
collect_time = '0800'
cur_time = collect_date+collect_time
# collect_time = str(int(collect_time)-10).zfill(4)

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
x_ = [[i for i in x]]

model = load('./sample_data/joblib/silvercare_model.joblib')

y_ = model.predict(x_)
print(y_) # 5분 전의 데이터 labeling
y_1 = [int(x) for x in y_]
print(y_1)




