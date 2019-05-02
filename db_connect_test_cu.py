import pymysql
from sqlalchemy import create_engine
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import time
from sklearn.model_selection import GridSearchCV
import numpy as np
from joblib import load, dump
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# sklearn package
def read_db_table(member_name = None, appliance_name = None, start = None, end = None):
    start = pd.to_datetime(start) or pd.to_datetime('2019-01')
    end = pd.to_datetime(end) or pd.to_datetime('2019-03')
    member_name = member_name or '박재훈'
    appliance_name = appliance_name or 'TV'

    def get_gateway_id():
        def get_member_id():
            sql = f"""
                SELECT member_id
                FROM ah_member
                WHERE 1=1
                AND member_name = '{member_name}'
            """
            return pd.read_sql(sql, aihems_service_db_connect).values[0][0]

        member_id = get_member_id()

        sql = f"""
            SELECT gateway_id
            FROM ah_gateway_assign
            WHERE 1=1
            AND member_id = '{member_id}'
        """
        return pd.read_sql(sql, aihems_service_db_connect).values[0][0]

    def get_device_address():
        sql = f"""
                SELECT device_address
                FROM ah_device
                WHERE 1=1
                AND gateway_id = '{gateway_id}'
                AND device_name = '{appliance_name}'
        """
        return pd.read_sql(sql, aihems_service_db_connect).values[0][0]

    def get_months(start, end):
        months = pd.date_range(start, end, freq = 'M')
        months_str = [str(x.year)+str(x.month).rjust(2, '0') for x in months]
        return(months_str)

    # db connection
    aihems_service_db_connect = pymysql.connect(host='aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com',
                         port=3306, user='aihems', passwd='#cslee1234', db='aihems_service_db', charset='utf8')

    # get data
    gateway_id = get_gateway_id()

    device_address = get_device_address()

    months = get_months(start, end)

    # 조회
    table = pd.DataFrame()
    for month in months:
        sql = f"""
                SELECT *
                FROM ah_data_diff_{month}
                WHERE 1=1
                AND gateway_id = '{gateway_id}'
                AND device_address = '{device_address}'
        """
        table = table.append(pd.read_sql(sql, aihems_service_db_connect), ignore_index=True)

    # aihems_service_db_connect.close()
    return(table)

def data_load(member_name, appliance_name, months = None):
    months = months or None
    # encoding = 'euc-kr'
    # df1 = pd.read_csv('./sample_data/csv/aihems/' + appliance_name + '(' + member_name + ')_01.csv',
    #                  encoding=encoding)  # 24일 이전 데이터 x
    # if months != 1:
    #    df2 = pd.read_csv('./sample_data/csv/aihems/' + appliance_name + '(' + member_name + ')_02.csv',
    #                      encoding=encoding)  # 2일부터...
    #    df1 = pd.concat([df1, df2], ignore_index=True)
    # df1 = df1.loc[df1.energy != '\\N', :].copy()
    # df1.columns.values[-1] = 'appliance_status'  # excel에 컬럼값 입력 안됨

    aihems_service_db_connect = pymysql.connect(host='aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com',
                                                port=3306, user='aihems', passwd='#cslee1234', db='aihems_service_db',
                                                charset='utf8')

    cursor = aihems_service_db_connect.cursor()
    query = f"""
                SELECT AD.gateway_id, AD.device_address, T1.member_id
                FROM ah_device AS AD,
                    (SELECT AM.member_id AS member_id, AG.gateway_id AS gateway_id
                    FROM
                    ah_gateway_assign AS AG
                    JOIN
                    ah_member AS AM
                    ON AG.member_id=AM.member_id
                    WHERE AM.member_name='{member_name}') AS T1
                WHERE 1=1
                    AND T1.gateway_id=AD.gateway_id
                    AND AD.device_name='{appliance_name}'
            """
    cursor.execute(query)
    result = cursor.fetchone()
    aihems_service_db_connect.close()
    gateway_id=result[0]
    device_address=result[1]
    member_id = result[2]
    gateway_id = gateway_id.split("-")[0]+gateway_id.split("-")[2]
    aihems_api_db_connect = pymysql.connect(host='aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com',
                                                port=3306, user='aihems', passwd='#cslee1234', db='aihems_api_db',
                                                charset='utf8')

    df1 = pd.DataFrame()

    sql = f"""
                SELECT device_address, collected_date, CONVERT(REPLACE(collected_time,':',''), SIGNED INTEGER) AS collected_time, quality, onoff, energy, energy_diff, appliance_status
                FROM AH_USE_LOG_BYMINUTE_LABLED
                WHERE 1=1
                AND gateway_id = '{gateway_id}'
                AND device_address = '{device_address}'
         """
    df1 = df1.append(pd.read_sql(sql, aihems_api_db_connect), ignore_index=True)
    data = {'member_name' : [member_name],
            'member_id' : [member_id],
            'gateway_id' : [result[0]],
            'device_address' : [device_address]}
    df6 = pd.DataFrame(data)
    df7 = pd.merge(df6, df1, on = 'device_address', how='right')
    return(df7)

def set_data(df, source = None):
    source = source or None

    if source == 'excel':
        df1 = df.loc[df.energy != '\\N', :].copy()  # db에서 load 할 때는 na로 들어옴.
    elif source == 'predict' or source == 'dbwrite':
        df1 = df.dropna()
        df1.loc[:, 'appliance_status'] = 0
    else:
        df1 = df.dropna()
        # df1.loc[:, 'appliance_status'] = 0

    df1.loc[:, 'collected_time'] = [str(x).rjust(4, '0') for x in df1.collected_time]
    if source == 'predict' or source == 'dbwrite':
        df1.loc[:, 'collected_time'] = [x[:2] + ':' + x[2:] for x in df1.collected_time]
    # df1.loc[:, 'collected_time'] = [x[:2] + x[2:] for x in df1.collected_time]
    df1.loc[:, 'collected_date'] = [str(x) for x in df1.collected_date]
    df1.loc[:, 'date_time'] = df1.loc[:, 'collected_date'] + ' ' + df1.loc[:, 'collected_time']
    df1.loc[:, 'date_time'] = pd.to_datetime(df1.loc[:, 'date_time'])

    df1.loc[:, 'smart_plug_onoff'] = df1.loc[:, 'onoff']

    df1.loc[:, 'smart_plug_onoff'] = [int(x) for x in df1.smart_plug_onoff]
    df1.loc[:, 'appliance_status'] = [int(x) for x in df1.appliance_status]
    df1.loc[df1.appliance_status.isna(), 'appliance_status'] = 0


    df1.loc[:, 'date'] = [str(x.date()) for x in df1.date_time]
    df1.loc[:, 'dayofweek'] = [str(x.dayofweek) for x in df1.date_time]  # 0이 월요일

    df1.loc[:, 'time'] = [str(x.time()) for x in df1.date_time]
    df1.loc[:, 'month'] = [str(x.month) for x in df1.date_time]
    df1.loc[:, 'day'] = [str(x.day) for x in df1.date_time]
    df1.loc[:, 'hour'] = [int(x.hour) for x in df1.date_time]
    df1.hour = [[x] for x in df1.hour]
    df1.loc[:, 'minute'] = [[int(x.minute)] for x in df1.date_time]
    df1.minute= [[x] for x in df1.minute]

    df1.loc[:, 'dayofyear'] = [int(x.dayofyear) for x in df1.date_time]
    if source == 'dbwrite':
        df1.loc[:, 'energy'] = [float(x) for x in df1.energy]
    else:
        df1.loc[:, 'energy'] = [round(float(x)*1000) for x in df1.energy]
    df1.loc[:, 'energy_lagged'] = df1.energy.shift(+1)
    df1.loc[:, 'energy_diff'] = df1.energy - df1.energy_lagged
    df1.iloc[-1, 8] = 0
    df1.energy_diff[0] = 0

    # df1.loc[:, 'gateway_id'] = 'ep17470141'
    df1.loc[:, 'end_point'] = 1
    df1.loc[:, 'quality'] = 100
    df1.loc[:, 'create_date'] = pd.datetime.today()
    df1.loc[:, 'gateway_id'] = df1.gateway_id[0]
    # gateway_id = df1.gateway_id[0]
    # df1.gateway_id = gateway_id[:6] + gateway_id[-4:]

    return (df1)

def window_stack(X, stepsize=10):
    X = [0] * (stepsize - 1) + X
    X_transformed = [X[i-stepsize:i] for i in range(len(X)+1) if i > stepsize-1]
    return (X_transformed)

def split_x_y(df):
    X = [x for x in df.energy_diff]
    X_transformed = window_stack(X, stepsize=10)
    # X_transformed = X_transformed.reshape()
    Y = np.array([np.array([x]) for x in df.loc[:, 'appliance_status']])
    Y = Y.ravel()
    return(X_transformed, Y)

def make_prediction_model(member_name = None, appliance_name = None, save = None):
    member_name = member_name or '박재훈'
    appliance_name = appliance_name or 'TV'
    save = save or None

    df = data_load(member_name = member_name, appliance_name = appliance_name)
    # df = set_data(df, source='excel')
    df = set_data(df, source=None)
    X, Y = split_x_y(df)

    model = RandomForestClassifier()
    params = {
        'n_estimators': [10],
        # 'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 11)],
        'max_depth': [None],  # default None
        # 'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
        'min_samples_split': [2],  # default 2
        # 'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1],  # default 1
        # 'min_samples_leaf': [1, 2, 4],  # default 1
        'max_features': ['auto'],
        # 'max_features': ['sqrt', 'auto'],
        'bootstrap': [True],
        # 'bootstrap': [True, False]
    }

    gs = (GridSearchCV(estimator=model,
                       param_grid=params,
                       cv=5,
                       scoring='f1',  # accuracy, balanced_accuracy, average_precision, brier_score_loss,
                       n_jobs=-1))

    gs.fit(X, Y)

    if save == True:
        dump(gs, './filename.joblib')

    return(gs, X, Y)

def write_db(df):
    user = 'aihems'
    passwd = '#cslee1234'
    addr = 'aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com'
    port = "3306"
    db_name = 'aihems_api_db'
    engine = create_engine("mysql+mysqldb://"+user+":" + passwd +"@"+addr+":"+port+"/"+db_name,
                           encoding='utf-8')
    # conn = engine.connect()
    df.to_sql('AH_USE_LOG_BYMINUTE_LABLED', con=engine, if_exists='append', index=False)
    return(0)

def transform_data(df):
    df = df.loc[:, [
          'gateway_id' #
        , 'device_address' #
        , 'end_point'
        , 'collected_date' #
        , 'collected_time' #
        , 'quality'
        , 'onoff' #
        , 'energy' #
        , 'energy_diff' #
        , 'appliance_status' #
        , 'create_date'
                  ]]
    return(df)

member_name = input('사용자 이름: ')
appliance_name = input('가전기기 이름: ')
# start = input('시작일: ')
#end = input('종료일: ')
# df = data_load(member_name=member_name,appliance_name=appliance_name)
# print(df)
df = data_load(member_name=member_name,appliance_name=appliance_name)
start = time.time()
gs, X, Y = make_prediction_model(member_name=member_name, appliance_name=appliance_name)
end = time.time()
print('걸리시간: ', round(end-start, 3), 's', sep = "")
print('정확도: ', round(gs.best_score_, 3) , sep = "")
#
# model_fitted = gs
# model_loaded = load('./')
#
# model_fitted.predict()
# df = read_db_table(member_name= member_name, appliance_name = appliance_name,  start = '2019-03', end = '2019-04') #다원플러그
df = read_db_table(member_name= member_name, appliance_name = appliance_name,  start = '2019-04', end = '2019-05') #이젝스플러그(실증세대)
# df5 = data_load(member_name=member_name, appliance_name=appliance_name)
df1 = set_data(df, source='predict')
# df4 = set_data(df, source='predict')
# X1, Y1 = split_x_y(df1)
X, Y = split_x_y(df1)

Y = gs.predict(X)
df2 = set_data(df, source='dbwrite')
df2.appliance_status = Y
gateway_id = df2.gateway_id[0]
df2.gateway_id = gateway_id[:6] + gateway_id[-4:]

df3 = transform_data(df2) # todo:대기전력도 appliance_status가 1로 표시되는것 수정

# write_db(df2)

#df = data_load(member_name='박재훈', appliance_name='TV')
#df = set_data(df, source='excel')
#df = transform_data(df)
# write_db(df)