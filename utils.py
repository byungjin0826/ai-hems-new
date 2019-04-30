import pymysql
import numpy as np
import sklearn as sk
import pandas as pd
from sqlalchemy import create_engine
from joblib import dump, load


def get_table_from_db(sql):
    """
    작성된 SQL 문으로 데이터 불러오기
    :return:
    :param sql: sql문
    :return: python DataFrame
    """
    aihems_service_db_connect = pymysql.connect(host='aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com',
                                                port=3306, user='aihems', passwd='#cslee1234', db='aihems_service_db',
                                                charset='utf8')
    df = pd.read_sql(sql, aihems_service_db_connect)
    return df


# X 값을 변환하여 사용
def window_stack(x, step_size=10, lag=2):  # todo: 1. X가 여러개의 컬럼일 때도 동작할 수 있도록, 2. Lag 부분 추가
    """
    상태 판별 예측을 위한 입력 데이터 변환
    :param x: 분 단위 전력 사용량
    :param step_size: Sliding window 의 사이즈
    :param lag: 숫자만큼 지연 
    :return:
    """
    x = [0] * (step_size - 1) + x
    x_transformed = [x[i - step_size + lag:i + lag] for i in range(len(x) + 1) if i > step_size - 1]
    return x_transformed  #


def data_load():  # todo: Excel에서 데이터 불러오기, DB에 저장 완료 시 필요없음
    return 0


def set_data():  # todo: 기존 함수 복사해서 붙여넣기
    return 0


def split_x_y(df, x_col=['elec'], y_col=['status']):  # todo: X와 Y 분리하기, 컬럼이 다수일 때도 가능하도록, 명칭 다시 수정
    """
    DataFrame에서 X와 Y를 분리
    :param df: python DataFrame
    :param x_col:
    :param y_col:
    :return:
    """
    X = df.loc[:, x_col].values
    Y = df.loc[:, y_col].values
    return (X, Y)


def make_prediction_model(member_name=None, appliance_name=None, save=None):  # todo: params를 저장되 있는 값으로 불러오기
    member_name = member_name or '박재훈'
    appliance_name = appliance_name or 'TV'
    save = save or None

    df = data_load(member_name=member_name, appliance_name=appliance_name)
    df = set_data(df, source='excel')
    X, Y = split_x_y(df)

    model = sk.ensemble.RandomForestClassifier()
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

    gs = sk.model_selection.GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    gs.fit(X, Y)

    if save == True:
        dump(gs, './filename.joblib')
    return (gs)


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
    return (0)


cols_dic = dict(ah_appliance=[
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
], ah_appliance_connect=[
    'appliance_no'
    , 'gateway_id'
    , 'device_address'
    , 'end_point'
    , 'flag_delete'
    , 'create_date'
], ah_appliance_energy_history=[
    'appliance_no'
    , 'create_date'
    , 'wait_energy'
    , 'wait_minute'
    , 'wait_power'
    , 'use_energy'
    , 'use_minute'
    , 'use_power'
], ah_appliance_history=[
    'appliance_no'
    , 'create_date'
    , 'end_date'
    , 'gateway_id'
    , 'device_address'
    , 'end_point'
    , 'appliance_type'
    , 'appliance_name'
], ah_appliance_remocon_config=[
    'appliance_no'
    , 'device_address'
    , 'end_point'
    , 'maker'
    , 'codeset'
    , 'create_date'
    , 'modify_date'
], ah_appliance_type=[
    'appliance_type'
    , 'appliance_type_name'
    , 'appliance_type_descript'
    , 'flag_delete'
    , 'create_date'
    , 'modify_date'
], ah_device=[
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
], ah_device_install=[
    'gateway_id'
    , 'device_address'
    , 'end_point'
    , 'flag_delete'
    , 'create_date'
], ah_gateway=[
    'gateway_id'
    , 'gateway_name'
    , 'gateway_xmpp_id'
    , 'gateway_mqtt_id'
    , 'flag_delete'
    , 'create_date'
    , 'modify_date'
], ah_gateway_install=[
    'house_no'
    , 'gateway_id'
    , 'flag_delete'
    , 'create_date'
], ah_house=[
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
], ah_house_member=[
    'house_no'
    , 'user_no'
    , 'flag_delete'
    , 'create_date'
], ah_usage_daily_predict=[
    'house_no'
    , 'collected_date'
    , 'use_energy'
    , 'predict_use_energy'
    , 'progressive_level'
    , 'create_date'
    , 'modify_date'
], ah_usage_monthly=[
    'house_no'
    , 'year'
    , 'month'
    , 'use_energy'
    , 'predict_use_energy'
    , 'create_date'
    , 'modify_date'
], ah_usage_monthly_byappliance=[
    'house_no'
    , 'appliance_no'
    , 'year'
    , 'month'
    , 'use_energy'
    , 'create_date'
    , 'modify_date'
], ah_usage_monthly_bydevice=[
    'house_no'
    , 'device_address'
    , 'end_point'
    , 'year'
    , 'month'
    , 'use_energy'
    , 'create_date'
    , 'modify_date'
], ah_user=[
    'user_no'
    , 'user_name'
    , 'email'
    , 'login_id'
    , 'login_password'
    , 'flag_delete'
    , 'create_date'
    , 'modify_date'
], ah_use_log_byminute=[
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
], ah_use_log_byminute_labled=[
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
])

classifications = {
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

regressions = {''}
