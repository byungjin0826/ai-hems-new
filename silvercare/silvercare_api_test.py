from flask import Flask
from flask_restful import Resource, Api, reqparse
import utils
from joblib import load, dump
import datetime
#
#
#             # parser = reqparse.RequestParser()
#             # parser.add_argument('device_id', type=str)
#             # parser.add_argument('gateway_id', type=str)
#             # parser.add_argument('collect_date', type=str)
#             # parser.add_argument('collect_time', type=str)
#             # args = parser.parse_args()
#
#             # device_id = args['device_id']
#             # gateway_id = args['gateway_id']
#             # collect_date = args['collect_date']
#             # collect_time = args['collect_time']
#
# # gateway_id = 'ep18270185'
# # collect_date = '20190810'
# # collect_time = '0800'
# # cur_time = collect_date+collect_time
# # # collect_time = str(int(collect_time)-10).zfill(4)
# #
# # sql = f"""
# #     SELECT *
# #     FROM AH_USE_LOG_BYMINUTE
# #     WHERE 1=1
# #         AND GATEWAY_ID = '{gateway_id}'
# #         AND CONCAT( COLLECT_DATE, COLLECT_TIME) > DATE_FORMAT( DATE_ADD( STR_TO_DATE( '{cur_time}', '%Y%m%d%H%i'), INTERVAL -10 MINUTE), '%Y%m%d%H%i')
# #     ORDER BY COLLECT_DATE, COLLECT_TIME
# #     LIMIT 0, 10
# # """
# #
# # df = utils.get_table_from_db(sql)
# device_id = '00158D0001A44A651'
# sql = f"""
#     SELECT DEVICE_ID, MODEL_ID
#     FROM AH_DEVICE_MODEL
#     WHERE 1=1
#         AND DEVICE_ID = '{device_id}'
# """
# device_name = utils.get_table_from_db(sql)
# model_name = device_name['model_id'][0]
#
# print(model_name)
#
# # x, y = utils.split_x_y(df, x_col='energy_diff')
# # x_ = [[i for i in x]]
#
# # device_list = ['00158D0001A4590E1', '00158D0001A44CC51', '00158D000151B1E71', '00158D0001A4528D1']
# # if device_id == '00158D0001A474EC1':  # 204호
# #     model_name = 'silvercare_model_2'
# # elif device_id in device_list:
# #     model_name = 'silvercare_model_1'  # 106호,104호, 206호, 111호,
# # else:
# #     model_name = 'silvercare_model'  # 205호
#
# model = load(f'./sample_data/joblib/silvercare/{model_name}_labeling.joblib')
#
# # y_ = model.predict(x_)
# # print(y_) # 5분 전의 데이터 labeling
# # y_1 = [int(x) for x in y_]
# # print(y_1)

# main화면
# input : now_time
# output : house_no, house_name, status

sql = f"""
SELECT HOUSE_NO, HOUSE_NAME
FROM HOUSE_DATA
ORDER BY HOUSE_NO
"""
df = utils.get_table_from_db(sql, db='silver_service_db')
collect_date = '20191006'
cur_time = '1941'
sql = f"""
SELECT *
FROM SC_DEVICE_LOG
WHERE COLLECT_DATE = '{collect_date}'
AND COLLECT_TIME = '{cur_time}'
"""
house_list = []
for i in df.index:
    house_list.append([df['house_no'][i], df['house_name'][i]])



# 상세화면
# input : house_no, now_time
# output : collect_time, avg_status, label_status, status






