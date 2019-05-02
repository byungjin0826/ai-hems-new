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

def datetime_transform(df):
    datetime = df.collected_date + ' ' + df.collected_time
    df.loc[:, 'datetime'] = pd.to_datetime(datetime, format= '%Y%m%d')
    df_datetime_indexing = df.set_index(['datetime', 'name'])
    return df_datetime_indexing


def progressive_level(cumulative_energy):
    if cumulative_energy >= 400:
        progressive_level = 3
    elif cumulative_energy > 200:
        progressive_level = 2
    progressive_level = 1
    return(progressive_level)


def get_table_from_db(sql, db = 'aihems_service_db'):
    """
    작성된 SQL 문으로 데이터 불러오기
    :return:
    :param sql: sql 문
    :return: python DataFrame
    """
    db = db or 'aihems_service_db'

    aihems_service_db_connect = pymysql.connect(host='aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com',
                                                port=3306, user='aihems', passwd='#cslee1234', db=db,
                                                charset='utf8')
    df = pd.read_sql(sql, aihems_service_db_connect)
    return df


# X 값을 변환하여 사용
def sliding_window_transform(x, y, step_size=10, lag=2):  # todo: 1. X가 여러개의 컬럼일 때도 동작할 수 있도록, 2. Lag 부분 추가
    """
    상태 판별 예측을 위한 입력 데이터 변환
    :param x: 분 단위 전력 사용량
    :param step_size: Sliding window 의 사이즈
    :param lag: 숫자만큼 지연
    :return:
    """
    x = [0] * (step_size - 1) + x
    x_transformed = [x[i - step_size + lag:i + lag] for i in range(len(x) + 1 - lag) if i > step_size - 1]
    y_transformed = y[:-lag]
    return x_transformed, y_transformed  #


def set_data_type(df):  # todo: 기존 함수 복사해서 붙여넣기
    data_type_list = {
        'energy':float
        , 'collected_date':int
        , 'month':int
        , 'dayinmonth':int
        # , 'day'
    }

    if df.columns in 'collected_date':
        df.loc[:, ]
    df_data_type_setted = 1
    return df_data_type_setted

def transform_collected_date(collected_date): # todo: 날짜를 sin 과 cos 으로 변환
    collected_date = pd.to_datetime(collected_date)
    collected_date_transformed = {
        'month_x':collected_date.month
    }
    return collected_date_transformed

def split_x_y(df, x_col = 'energy', y_col = 'appliance_status'):  # todo: X와 Y 분리하기, 컬럼이 다수일 때도 가능하도록, 명칭 다시 수정
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


def make_prediction_model(member_name=None, appliance_name=None, save=None, model_name = None):  # todo: params 를 저장되 있는 값으로 불러오기
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

def write_db(df, table_name='AH_USE_LOG_BYMINUTE_LABLED'):
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
    df.to_sql(table_name, con=engine, if_exists='append', index=False)
    return 0


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
        , 'collected_date'
        , 'use_energy'
        , 'predict_use_energy'
        , 'progressive_level'
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
        , 'collected_date'
        , 'collected_time'
        , 'quality'
        , 'onoff'
        , 'energy'
        , 'energy_diff'
        , 'appliance_status'
        , 'create_date'
    ],
    'ah_use_log_byminute_labled': [
        'gateway_id'
        , 'device_address'
        , 'end_point'
        , 'collected_date'
        , 'collected_time'
        , 'quality'
        , 'onoff'
        , 'energy'
        , 'energy_diff'
        , 'appliance_status'
        , 'create_date'
    ]
}

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


classifications = {
    ''
}
