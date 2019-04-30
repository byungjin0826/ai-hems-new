import pymysql
import pandas as pd
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sqlalchemy import create_engine
from sklearn.neural_network import multilayer_perceptron

# SELECT *
# FROM ah_device
# WHERE 1=1
# AND gateway_id = 'ep1827-dpiuctns-0236'

def datetime_transform(df):
    datetime = df.collected_date + ' ' + df.collected_time
    df.loc[:, 'datetime'] = pd.to_datetime(datetime, format= '%Y%m%d')
    df_datetime_indexing = df.set_index(['datetime', 'name'])
    return(df_datetime_indexing)

def progressive_level(cumulative_energy):
    if cumulative_energy >= 400:
        progressive_level = 3
    elif cumulative_energy > 200:
        progressive_level = 2
    progressive_level = 1  ####else가 들어가야 하는 것이 아닌지...?(누진구간)
    return(progressive_level)

def transform(df):
    df = df.loc[:, [
        'house_no'
        , 'collected_date' #
        , 'use_energy' #
        , 'predict_use_energy' #
        , 'progressive_level'
        , 'create_date'
        , 'modify_date'
                   ]]
    return(df)

def write_db(df):
    user = 'aihems'
    passwd = '#cslee1234'
    addr = 'aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com'
    port = "3306"
    db_name = 'aihems_api_db'
    engine = create_engine("mysql+mysqldb://"+user+":" + passwd +"@"+addr+":"+port+"/"+db_name,
                           encoding='utf-8')
    # conn = engine.connect()
    df.to_sql('AH_USAGE_DAILY_PREDICT', con=engine, if_exists='append', index=False)
    return(0)


aihems_service_db_connect = pymysql.connect(host='aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com',
                                            port=3306, user='aihems', passwd='#cslee1234', db='aihems_service_db',
                                            charset='utf8')

# df = df.loc[:, [
#     'appliance_no'
#     , 'create_date'
#     , 'wait_energy'
#     , 'wait_minute'
#     , 'wait_power'
#     , 'use_energy'
#     , 'use_minute'
#     , 'use_power'
#                ]]



# 안채
gateway_id = ''
device_address = ''

sql_meter = f"""
SELECT gateway_id,device_address,collected_date, MAX(energy)-MIN(energy) AS use_energy
FROM ah_log_meter_201904
WHERE 1=1
AND gateway_id = {gateway_id}
AND device_address = {device_address}
AND energy != 0
GROUP BY collected_date
"""

df = pd.read_sql(sql_meter, aihems_service_db_connect)

X, Y = df.loc[:, ['collected_date']], df.loc[:, ['use_energy']]

model = linear_model.Ridge()

params = {
    'alpha':[0.5]
}

# Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
#       normalize=False, random_state=None, solver='auto', tol=0.001)
#
gs = GridSearchCV(estimator= model
                  , param_grid=params
                  , scoring='r2'
                  , n_jobs=-1
                  , cv = 5)

gs.fit(X, Y)

Y_pr = gs.predict(X)

print(r2_score(Y, Y_pr))

df['use_energy'] = Y
df['predict_use_energy'] = Y_pr
df['house_no'] = '20190325000001'
df['progressive_level'] = list(map(lambda x:progressive_level(x), df.predict_use_energy))
df['create_date'] = pd.datetime.today()
df['modify_date'] = pd.datetime.today()

df1 = transform(df)
# write_db(df1)