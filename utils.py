import pymysql
import numpy as np
import sklearn as sk
import sklearn.ensemble
import sklearn.linear_model
import pandas as pd
from sqlalchemy import create_engine
from joblib import dump, load
import time
import sklearn.metrics
import sklearn.metrics
import datetime

# 변환없이 원본 가져오는 건 get
# 조금이라도 계산하는 건 calc

def labeling_db_to_db(sql,db='aihems_service_db'):
    df = get_table_from_db(sql, db)
    df = df.dropna()
    df.loc[:, 'energy_diff'] = 0
    df.loc[:, 'appliance_status'] = 0
    df.energy_diff = df.energy - df.energy.shift(1)
    df.loc[df.energy_diff.isna(), 'energy_diff'] = 0
    df.loc[:,'end_point'] = 1
    df.loc[:,'quality'] = 100
    df = df.loc[:, cols_dic['ah_use_log_byminute_labeled'][:-1]]
    df.loc[:, 'create_date'] = pd.datetime.today()
    gateway_id = df.gateway_id[0]
    df.gateway_id = gateway_id[:6] + gateway_id[-4:]
    df.columns = ['gateway_id', 'device_id', 'end_point', 'collect_date',
                  'collect_time', 'quality', 'onoff', 'energy',
                  'energy_diff', 'appliance_status', 'create_date']
    return df

def get_appliance_type(device_id):
    sql = f"""
    SELECT appliance_type
    FROM AH_APPLIANCE_HISTORY
    WHERE 1=1
    AND device_id = '{device_id}'
    """
    appliance_type = get_table_from_db(sql)
    return appliance_type.values.item()

def get_appliance_name(appliance_no):
    sql = f"""
    SELECT appliance_name
    FROM AH_APPLIANCE
    WHERE appliance_no = '{appliance_no}' 
    """
    device_name = get_table_from_db(sql)
    return device_name.values.item()

def get_gateway_id(name):
    sql = f"""
    SELECT gateway_id
    FROM AH_GATEWAY
    WHERE gateway_name = '{name}'
    """
    gateway_id = get_table_from_db(sql)
    return gateway_id.values.item()

def get_appliance_no(device_id):
    sql = f"""
    SELECT appliance_no
    FROM AH_APPLIANCE_CONNECT
    WHERE 1=1
    AND device_id = '{device_id}'
    """
    appliance_no = get_table_from_db(sql)
    return appliance_no.values.item()

def get_device_list(gateway_id):
    sql = f"""
    SELECT AH_DEVICE_INSTALL.device_id, device_name, device_type
    FROM AH_DEVICE_INSTALL
    LEFT JOIN AH_DEVICE
    ON AH_DEVICE_INSTALL.device_id = AH_DEVICE.device_id
    WHERE 1=1
    AND gateway_id = '{gateway_id}'
    """
    device_list = get_table_from_db(sql)
    return device_list

def get_device_list_same_type(appliance_type):
    sql = f"""
    SELECT APPLIANCE_NO, APPLIANCE_NAME, DEVICE_ID, GATEWAY_ID
    FROM AH_APPLIANCE_HISTORY
    WHERE APPLIANCE_TYPE = 
            (SELECT APPLIANCE_TYPE
            FROM AH_APPLIANCE_TYPE
            WHERE 1 = 1
            AND APPLIANCE_TYPE = '{appliance_type}')
    """
    device_list = get_table_from_db(sql)
    return device_list

def get_appliance_types():
    sql = """
    SELECT *
    FROM AH_APPLIANCE_TYPE
    """
    df = get_table_from_db(sql)
    return df

def get_raw_data(device_id = None, gateway_id = None, start = None, end = None, month_print = False,
                 sql_print = False, table_name = 'AH_USE_LOG_BYMINUTE'):
    start = start or (datetime.datetime.now() - datetime.timedelta(30)).strftime('%Y%m%d')
    end = end or datetime.datetime.now().strftime('%Y%m%d')
    months = [x.date().strftime('%Y%m') for x in pd.date_range(start, end, freq = 'M')]

    df = pd.DataFrame()
    for month in months:
        sql = f"""
        SELECT *
        FROM {table_name}_{month}
        WHERE 1=1
        AND COLLECT_DATE >= {start}
        AND COLLECT_DATE <= {end}
        """

        if gateway_id != None:
            sql += f"""AND gateway_id = '{gateway_id}'\n"""

        if device_id != None:
            sql += f"""AND device_id = '{device_id}'\n"""

        temp = get_table_from_db(sql)
        df = df.append(temp)
        if month_print:
            print(month)

        if sql_print:
            print(sql)
    return df

def select_device(device_list):
    print(device_list)
    return 0

def get_house_name(gateway_id):
    sql = f"""
    SELECT gateway_name
    FROM AH_GATEWAY
    WHERE gateway_id = '{gateway_id}'
    """
    house_name = get_table_from_db(sql)
    return house_name.values.item()

def get_house_no(house_name):
    sql = f"""
    SELECT house_no
    FROM AH_HOUSE
    WHERE house_name = '{house_name}'
    """
    house_no = get_table_from_db(sql)
    return house_no.values.item()

def get_adr_schedule():
    now = datetime.datetime.now().date().strftime('%y%m%d')
    sql = f"""
    SELECT *
    FROM adr
    WHERE 1=1
    AND start_date >= {now}
    """
    date_time = get_table_from_db(sql)
    return date_time

def load_labeling_model(device_id):
    root_path = './sample_data/joblib/'
    path = root_path + f"""{device_id}_labeling.joblib"""
    model = load(path)
    return model

def search_device_address(member_name, appliance_name):
    member_name = member_name or '박재훈'
    appliance_name = appliance_name or 'TV'
    sql = f"""
    SELECT AD.DEVICE_ID
    FROM	
    	(SELECT AG.GATEWAY_ID AS GATEWAY_ID, AG.GATEWAY_NAME AS GATEWAY_NAME, ADI.DEVICE_ID AS DEVICE_ID
    	FROM 
    		AH_GATEWAY AS AG
    		JOIN AH_DEVICE_INSTALL AS ADI
    		ON AG.GATEWAY_ID = ADI.GATEWAY_ID
    	WHERE AG.GATEWAY_NAME = '{member_name}') T1
    	JOIN AH_DEVICE AS AD
    	ON T1.DEVICE_ID = AD.DEVICE_ID
    WHERE AD.DEVICE_NAME = '{appliance_name}'
    """
    aihems_service_db_connect = pymysql.connect(host='aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com',
                                                port=3306, user='aihems', passwd='#cslee1234', db='aihems_api_db',
                                                charset='utf8')

    cursor = aihems_service_db_connect.cursor()
    cursor.execute(sql)
    result = cursor.fetchone()
    device_address = result[0]

    return device_address

def calc_payment_month(date_time, meter_day):
    print(date_time)
    month = date_time.month - 1
    if date_time.day > meter_day:
        month += 1

    if month == 0:
        month = 12
    return(month)

def calc_appliance_energy_history(device_id): # todo: pivot_table, group by, 또는 sql 함수로 변경
    sql = f"""
    SELECT *
    FROM AH_USE_LOG_BYMINUTE_LABELED
    WHERE 1=1
    AND device_id = '{device_id[:-1]}'
    """
    df = get_table_from_db(sql)
    wait_energy = df.loc[df.appliance_status == 0, 'energy_diff'].sum()
    wait_minute = df.loc[df.appliance_status == 0, 'energy_diff'].count()
    use_energy = df.loc[df.appliance_status == 1, 'energy_diff'].sum()
    use_minute = df.loc[df.appliance_status == 1, 'energy_diff'].count()
    appliance_no = get_appliance_no(device_id)
    energy_history = {'appliance_no':[appliance_no],
                      'wait_energy':[wait_energy],
                      'wait_minute':[wait_minute],
                      'use_energy':[use_energy],
                      'use_minute':[use_minute]}
    energy_history_table = pd.DataFrame(energy_history)
    return energy_history_table

def calc_usage_energy_hourly(gateway_id): # todo: check meter를 이용해서 미터가 있는 경우 meter 데이터를 활용
    # 시간당 데이터 필요 없을 수 있음
    device_list = get_device_list(gateway_id)
    if check_meter(device_list):
        print('True')

    df = get_raw_data(gateway_id='ep17470141', table_name='AH_USE_LOG_BYMINUTE_LABELED')   # table 향후 변경 필요
    df = binding_time(df)
    house_name = get_house_name(gateway_id)
    house_no = get_house_no(house_name)

    df_hourly = df.resample('15min').sum()

    # db에 있는 형식을 맞추기 위한...
    df_hourly = unpacking_time(df_hourly)
    df_hourly.loc[:, 'house_no'] = house_no

    return df_hourly.loc[:, cols_dic['ah_usage_hourly'][:-2]]

def calc_weekly_schedule(device_id, threshold = 0.95): # todo: test 해보기
    df = get_raw_data(device_id = device_id, table_name='AH_USE_LOG_BYMINUTE')
    df = binding_time(df)
    schedule_sum = df.pivot_table(values='appliance_status', index=df.index.time,
                              columns=df.index.dayofweek, aggfunc='sum')
    schedule_count = df.pivot_table(values='appliance_status', index=df.index.time,
                              columns=df.index.dayofweek, aggfunc='count')
    schedule = schedule_sum/schedule_count
    schedule = schedule[schedule>threshold]
    return schedule

def calc_cbl(gateway_id = 'ep17470141', date = '2018-08-24',start = '00:00', end = '00:45'): # todo: 쿼리로 불러오도록 수정
    df = get_raw_data(gateway_id='ep17470141', table_name='AH_USE_LOG_BYMINUTE_201808')   # table 향후 변경 필요
    df = binding_time(df)[:date]

    start_time = datetime.time(int(start[:2]), int(start[-2:]))
    end_time = datetime.time(int(end[:2]), int(end[-2:]))

    df_hourly_per_15min = df.loc[:, 'energy_diff'].resample('15min').sum()

    df_hourly_per_15min_subset = df_hourly_per_15min[start_time:end_time]

    cbl_list = [x for x in df_hourly_per_15min_subset.resample('1d').sum()[-6:-1]]
    cbl_list.sort()
    print(cbl_list)
    return sum(cbl_list[:4])/4 # todo: max 4/5 확인 필요

def calc_number_of_times(device_id): # todo: 정시에 발령되지 않는 상황 고려(15분 단위)
    date = datetime.datetime.now().strftime('%Y%m%d')
    df = get_raw_data(gateway_id='ep17470141')   # table 향후 변경 필요
    df = binding_time(df)[:date]

    start_time = datetime.time(int(start[:2]), int(start[-2:]))
    end_time = datetime.time(int(end[:2]), int(end[-2:]))

    df_hourly_per_15min = df.loc[:, 'energy_diff'].resample('15min').sum()

    df_hourly_per_15min_subset = df_hourly_per_15min[start_time:end_time]

    cbl_list = [x for x in df_hourly_per_15min_subset.resample('1d').sum()[-6:-1]]
    cbl_list.sort()
    print(cbl_list)
    return 0

def check_meter(device_list):
    return len(device_list.loc[device_list.device_type.isin(['meter']), :]) != 0

def excel_to_db(names):
    def load_data(name):
        df = pd.read_csv(f'./sample_data/csv/aihems/{name}.csv', encoding='euc-kr')
        df.columns.values[-1] = 'appliance_status'
        df.energy_diff = df.energy - df.energy.shift(1)
        df.loc[df.energy_diff.isna(), 'energy_diff'] = 0
        df.loc[:, 'end_point'] = 1
        df.loc[:, 'quality'] = 100
        df = df.loc[:, cols_dic['ah_use_log_byminute_labeled'][:-1]]
        return df

    for name in names:
        df = load_data(name)
        write_db(df, table_name='AH_USE_LOG_BYMINUTE_LABELED_copy')
        print(name)
    return df

def progressive_level(cumulative_energy):
    if cumulative_energy >= 400:
        progressive_level = 3
    elif cumulative_energy > 200:
        progressive_level = 2
    progressive_level = 1
    return(progressive_level)

def get_table_from_db(sql, db = 'aihems_api_db'):
    """
    작성된 SQL 문으로 데이터 불러오기
    :return:
    :param sql: sql 문
    :return: python DataFrame
    """
    db = db or 'aihems_api_db'

    aihems_service_db_connect = pymysql.connect(host='aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com',
                                                port=3306, user='aihems', passwd='#cslee1234', db=db,
                                                charset='utf8')
    df = pd.read_sql(sql, aihems_service_db_connect)
    df = df.rename(str.lower, axis='columns')
    return df

def sliding_window_transform(x, y, step_size=10, lag=2):  # todo: 1. X가 여러개의 컬럼일 때도 동작할 수 있도록
    """
    상태 판별 예측을 위한 입력 데이터 변환, 나중에 날짜나 요일, 시간, 날씨 데이터를 추가
    :param x: 분 단위 전력 사용량
    :param step_size: Sliding window 의 사이즈
    :param lag: 숫자만큼 지연
    :return:
    """
    x = [x for x in x]
    y = [x for x in y]
    x = [0] * (step_size - 1) + x
    x_transformed = [x[i - step_size + lag:i + lag] for i in range(len(x) + 1 - lag) if i > step_size - 1]
    y_transformed = y[:-lag]
    return x_transformed, y_transformed  #

def split_x_y(df, x_col = 'energy', y_col = 'appliance_status'):
    """
    학습에 사용할 DataFrame 에서 X와 Y를 분리
    :param df: python DataFrame
    :param x_col: 학습에 사용할 x 변수 선택, 기본값: 전력데이터만 사용
    :param y_col: 가전기기 상태
    :return:
    """
    x_col = x_col or ''
    y_col = y_col or ''

    x = df.loc[:, x_col].values
    if len(x_col) == 1:
        x=x.reshape(-1, 1)

    y = df.loc[:, y_col].values
    return x, y

def set_data_type(df):  # 현재 사용안함
    data_type_list = {
        'energy':float
        , 'collect_date':int
        , 'month':int
        , 'dayinmonth':int
        # , 'day'
    }

    if df.columns in 'collect_date':
        df.loc[:, ]
    df_data_type_setted = 1
    return df_data_type_setted

def transform_collect_date(collect_date): # todo: 날짜를 sin 과 cos 으로 변환
    collect_date = pd.to_datetime(collect_date)
    collect_date_transformed = {
        'month_x':collect_date.month
    }
    return collect_date_transformed

def make_usage_daily_predict_model(gateway_id):
    df = get_raw_data(device_id=gateway_id)
    x, y = split_x_y(df) # x_col, y_col 입력
    estimator, params = select_regression_model('linear regression')
    model = sk.model_selection.GridSearchCV(estimator=estimator,
                                            param_grid=params,
                                            cv = 5,
                                            n_jobs=-1,
                                            scoring='r2')
    model.fit(x, y)
    accuracy = sk.metrics.r2_score(y, model.predict(x))
    print(accuracy)
    return model

def make_prediction_model(member_name=None, appliance_name=None, save=None, model_name = None):
    """
    DB에 입력되어있는 데이터를 토대로 예측 모델 생성 및 저장
    Table은
    :param member_name: 이름, 기본값: 박재훈
    :param appliance_name: 가전기기명, 기본값: TV
    :param save: 저장여부 선택, 기본값: False
    :param model_name: 모델 선택, 모두 영문 소문자로 기입, 기본값: decision tree
    :return: fitting 이 완료된 GridSearchCV 객체
    """
    member_name = member_name or '박재훈'
    appliance_name = appliance_name or '박재훈'
    model_name = model_name or 'random forest'
    table_name = ''

    sql = f"""
    SELECT *
    FROM Table
    WHERE 1=1
    AND gateway_id = {member_name}
    AND device_address = {appliance_name}
    """

    df = get_table_from_db(sql)

    x, y = split_x_y(df)
    start = time.time()

    gs = sk.model_selection.GridSearchCV()
    gs.fit(x, y)
    end = time.time()
    print('학습 소요시간: ', round(end-start, 3), sep = '')

    if save:
        dump('./'+member_name+'_'+appliance_name+'.joblib')
        print('저장되었습니다.')

    return gs

def write_db(df, table_name='AH_USE_LOG_BYMINUTE_LABELED', if_exists = 'append'): # todo: update 기능 구현, 기존에 데이터가 존재하는 경우
    """
    python DataFrame을 Database에 업로드
    :param df: 업로드 하고자 하는 DataFrame
    :param table_name: DB table 명
    :return: 0
    """
    user = 'aihems'
    passwd = '#cslee1234'
    addr = 'aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com'
    port = "3306"
    db_name = 'aihems_api_db'
    engine = create_engine("mysql+mysqldb://" + user + ":" + passwd + "@" + addr + ":" + port + "/" + db_name,
                           encoding='utf-8')
    # conn = engine.connect()

    df.to_sql(table_name, con=engine, if_exists=if_exists, index=False)

    return 0

def binding_time(df): # DB 에서 불러온 데이터를 pandas 의 시계열 데이터로 활용하기 위해 필요
    df.loc[:, 'collect_date'] = [str(x) for x in df.collect_date]
    df.loc[:, 'collect_time'] = [str(x) for x in df.collect_time]
    df.loc[:, 'time'] = pd.to_datetime(df.collect_date + " " + df.collect_time, format='%Y%m%d %H:%M')
    df_time_indexing = df.set_index('time', drop=True)
    return df_time_indexing

def unpacking_time(df_time_indexing): # DB 에 있는 포맷으로 재변환
    df = df_time_indexing.reset_index()
    df.loc[:, 'collect_date'] = [x for x in df.date]
    df.loc[:, 'collect_time'] = [x for x in df.time]
    return df

def select_regression_model(model_name):
    regressions = {
        'random forest': [
            sk.ensemble.RandomForestClassifier(),
            {
                'n_estimator': [10]
                , 'criterion': ['gini']
                , 'max_depth': [None]
                , 'min_samples_split': [2]
                , 'min_samples_leaf': [1]
                , 'min_weight_fraction_leaf': [0.]
                , 'max_features': ["auto"]
                , 'max_leaf_nodes': [None]
                , 'min_impurity_decrease': [0.]
                , 'min_impurity_split': [1e-7]
                , 'bootstrap': [True]
                , 'oob_score': [False]
                , 'n_jobs': [None]
                , 'random_state': [None]
                , 'vervbse': [0]
                , 'warm_start': [False]
                , 'class_weight': [None]
            }
        ],
        'linear regression': [
            sk.linear_model.LinearRegression(),
            {
                'fit_intercept': [True]
                , 'normalize': [False]
                , 'copy_X': [True]
                , 'n_jobs': [None]
            }
        ],
        # 'polynomial regression':[
        #
        # ],
        # 'stepwise regression':[
        #
        # ],
        'ridge regression': [
            sk.linear_model.Ridge(),
            {
                'alpha': []
                , 'fit_intercept': []
                , 'normalize': [False]
                , 'copy_X': [True]
                , 'max_iter': []
                , 'tol': []
                , 'solver': ['auto']
                , 'random_state': [None]
            }
        ],
        'lasso regression': [
            sk.linear_model.Lasso(),
            {
                'alpha': []
                , 'fit_intercept': [True]
                , 'normalize': [False]
                , 'precompute': [False]
                , 'copy_X': [True]
                , 'max_iter': []
                , 'tol': []
                , 'warm_start': []
                , 'positive': []
                , 'random_state': [None]
                , 'selection': ['cyclic']
            }
        ],
        # 'elastic net regression':[
        #
        # ]
    }
    model = regressions[model_name][0]
    params = regressions[model_name][1]
    return model, params

def select_classification_model(model_name): # todo: 다른 모델들 파라미터 정리 필요
    classifications = {
        'logistic regression': [
            sk.linear_model.LogisticRegression(),
            {
                ''
            }
        ],
        # 'naive bayes': [
        #     sk.naive_bayes.GaussianNB(),
        #     {
        #         'var_smoothing':[1e-9]
        #     }
        # ],
        'stochastic gradient descent': [
            sk.linear_model.SGDClassifier(),
            {
                'loss':['hinge']
                , 'penalty':['l2']
                , 'alpha':[0.0001]
                , 'fit_intercept':[True]
                , 'max_iter':[1000]
                , 'tol':[1e-3]
                , 'shuffle':[True]
                # , 'verbose':[]
                # , 'epsilon':[]
                , 'n_jobs':[None]
                , 'random_state':[None]
                # , 'learning_rate':[]
                , 'power_t':[0.5]
                , 'early_stopping':[False]
                , 'validation_fraction':[0.1]
                , 'n_iter_no_change':[5]
                # , 'class_weight':[]
                # , 'warm_start':[]
                # , 'average':[]
                , 'n_iter':[None]
            }
        ],
        'k-nearest neighbours': [

        ],
        'decision tree': [
            sk.tree.DecisionTreeClassifier(),
            {}
        ],
        'random forest': [
            sk.ensemble.RandomForestClassifier(),
            {
                'n_estimators': [10]
                , 'criterion': ['gini']
                , 'max_depth': [None]
                , 'min_samples_split': [2]
                , 'min_samples_leaf': [1]
                , 'min_weight_fraction_leaf': [0.]
                , 'max_features': ['auto']
                , 'max_leaf_nodes': [None]
                , 'min_impurity_decrease': [0.]
                # , 'min_impurity_split': [0]
                , 'bootstrap': [True]
                , 'oob_score': [False]
                , 'n_jobs': [None]
                , 'random_state': [None]
                , 'verbose': [0]
                , 'warm_start': [False]
                , 'class_weight': [None]
            }
        ],
        'support vector machine': [
            sk.svm.SVC(),
            {
                'C':[1.0]
                , 'kernel':['rbf']
                # , 'degree':[3]
                , 'gamma':['auto']
                , 'coef0':[0.0]
                , 'shrinking':[True]
                , 'probability':[False]
                , 'tol':[1e-3]
                , 'cache_size':[]
                , 'class_weight':[]
                , 'verbose':[False]
                , 'max_iter':[-1]
                , 'decision_function_shape':['ovr']
                , 'random_state':[None]
            }
        ]
    }
    model = classifications[model_name][0]
    params = classifications[model_name][1]
    return model, params

def prediction_test(model, device_id):
    df = get_raw_data(device_id=device_id, table_name='AH_USE_LOG_BYMINUTE_LABELED')
    x, y = split_x_y(df, x_col='energy_diff')
    x, y = sliding_window_transform(x, y, lag=10, step_size=30)
    accuracy = sk.metrics.accuracy_score(y, model.predict(x))
    return accuracy

cols_dic = {
    'ah_appliance': [
        'appliance_no'
        , 'appliance_type'
        , 'appliance_name'
        , 'maker'
        , 'model_no'
        , 'purchase_date'
        , 'wait_power'
        , 'use_power'
        , 'flag_use_remocon'
        , 'flag_delete'
        , 'create_date'
        , 'modify_date'
    ],
    'ah_appliance_connect': [
        'appliance_no'
        , 'gateway_id'
        , 'device_address'
        , 'end_point'
        , 'flag_delete'
        , 'create_date'
    ],
    'ah_appliance_energy_history': [
        'appliance_no'
        , 'create_date'
        , 'wait_energy'
        , 'wait_minute'
        , 'wait_power'
        , 'use_energy'
        , 'use_minute'
        , 'use_power'
    ],
    'ah_appliance_history': [
        'appliance_no'
        , 'create_date'
        , 'end_date'
        , 'gateway_id'
        , 'device_address'
        , 'end_point'
        , 'appliance_type'
        , 'appliance_name'
    ],
    'ah_appliance_remocon_config': [
        'appliance_no'
        , 'device_address'
        , 'end_point'
        , 'maker'
        , 'codeset'
        , 'create_date'
        , 'modify_date'
    ],
    'ah_appliance_type': [
        'appliance_type'
        , 'appliance_type_name'
        , 'appliance_type_descript'
        , 'flag_delete'
        , 'create_date'
        , 'modify_date'
    ],
    'ah_device': [
        'device_address'
        , 'end_point'
        , 'device_type'
        , 'device_name'
        , 'control_reason'
        , 'flag_use_ai'
        , 'flag_use_metering'
        , 'flag_delete'
        , 'create_date'
        , 'modify_date'
    ],
    'ah_device_install': [
        'gateway_id'
        , 'device_address'
        , 'end_point'
        , 'flag_delete'
        , 'create_date'
    ],
    'ah_gateway': [
        'gateway_id'
        , 'gateway_name'
        , 'gateway_xmpp_id'
        , 'gateway_mqtt_id'
        , 'flag_delete'
        , 'create_date'
        , 'modify_date'
    ],
    'ah_gateway_install': [
        'house_no'
        , 'gateway_id'
        , 'flag_delete'
        , 'create_date'
    ],
    'ah_house': [
        'house_no'
        , 'house_name'
        , 'house_address'
        , 'house_detail_address'
        , 'meter_day'
        , 'contract_type'
        , 'ai_control_mode'
        , 'flag_active_dr'
        , 'flag_delete'
        , 'create_date'
        , 'modify_date'
    ],
    'ah_house_member': [
        'house_no'
        , 'user_no'
        , 'flag_delete'
        , 'create_date'
    ],
    'ah_usage_daily_predict': [
        'house_no'
        , 'collect_date'
        , 'use_energy'
        , 'predict_use_energy'
        , 'progressive_level'
        , 'create_date'
        , 'modify_date'
    ],
    'ah_usage_hourly': [
        'house_no'
        , 'year'
        , 'month'
        , 'day'
        , 'hour'
        , 'energy_diff' # todo: 수정필요
        , 'create_date'
        , 'modify_date'
    ],
    'ah_usage_monthly': [
        'house_no'
        , 'year'
        , 'month'
        , 'use_energy'
        , 'predict_use_energy'
        , 'create_date'
        , 'modify_date'
    ],
    'ah_usage_monthly_byappliance': [
        'house_no'
        , 'appliance_no'
        , 'year'
        , 'month'
        , 'use_energy'
        , 'create_date'
        , 'modify_date'
    ],
    'ah_usage_monthly_bydevice': [
        'house_no'
        , 'device_address'
        , 'end_point'
        , 'year'
        , 'month'
        , 'use_energy'
        , 'create_date'
        , 'modify_date'
    ],
    'ah_user': [
        'user_no'
        , 'user_name'
        , 'email'
        , 'login_id'
        , 'login_password'
        , 'flag_delete'
        , 'create_date'
        , 'modify_date'
    ],
    'ah_use_log_byminute': [
        'gateway_id'
        , 'device_address'
        , 'end_point'
        , 'collect_date'
        , 'collect_time'
        , 'quality'
        , 'onoff'
        , 'energy'
        , 'energy_diff'
        , 'appliance_status'
        , 'create_date'
    ],
    'ah_use_log_byminute_labeled': [
        'gateway_id'
        , 'device_id'
        , 'collect_date'
        , 'collect_time'
        , 'quality'
        , 'onoff'
        , 'energy'
        , 'energy_diff'
        , 'appliance_status'
        , 'create_date'
    ]
}

# todo: 업데이트 코드 작성

def update(df, table_name):
    write_db(df, table_name='temp', if_exists='replace')
    sql = f"""
    UPDATE final_table AS f
    SET col1 = t.col1
    FROM temp_table AS t
    WHERE f.id = t.id
    """
    return

# todo: 예외처리

# todo: class 구성