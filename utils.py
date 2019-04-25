import pymysql
import numpy as np
import sklearn as sk
import pandas as pd
from sqlalchemy import create_engine
import joblib

def get_table_from_db(sql):
    aihems_service_db_connect = pymysql.connect(host='aihems-service-db.cnz3sewvscki.ap-northeast-2.rds.amazonaws.com',
                                                port=3306, user='aihems', passwd='#cslee1234', db='aihems_service_db',
                                                charset='utf8')
    df = pd.read_sql(sql, aihems_service_db_connect)
    return(df)

def get_socket_table_from_db():
    return(0)

def window_stack(X, stepsize=10):
    X = [0] * (stepsize - 1) + X
    X_transformed = [X[i-stepsize:i] for i in range(len(X)+1) if i > stepsize-1]
    return (X_transformed)

def make_prediction_model(member_name = None, appliance_name = None, save = None):
    member_name = member_name or '박재훈'
    appliance_name = appliance_name or 'TV'
    save = save or None

    df = data_load(member_name = member_name, appliance_name = appliance_name)
    df = set_data(df, source = 'excel')
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
    return(gs)

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

cols_dic = {
    'ah_appliance':[
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
    'ah_appliance_connect':[
        'appliance_no'
        , 'gateway_id'
        , 'device_address'
        , 'end_point'
        , 'flag_delete'
        , 'create_date'
    ],
    'ah_appliance_energy_history':[
        'appliance_no'
        , 'create_date'
        , 'wait_energy'
        , 'wait_minute'
        , 'wait_power'
        , 'use_energy'
        , 'use_minute'
        , 'use_power'
    ],
    'ah_appliance_history':[
        'appliance_no'
        , 'create_date'
        , 'end_date'
        , 'gateway_id'
        , 'device_address'
        , 'end_point'
        , 'appliance_type'
        , 'appliance_name'
    ],
    'ah_appliance_remocon_config':[
        'appliance_no'
        , 'device_address'
        , 'end_point'
        , 'maker'
        , 'codeset'
        , 'create_date'
        , 'modify_date'
    ],
    'ah_appliance_type':[
        'appliance_type'
        , 'appliance_type_name'
        , 'appliance_type_descript'
        , 'flag_delete'
        , 'create_date'
        , 'modify_date'
    ],
    'ah_device':[
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
    'ah_device_install':[
        'gateway_id'
        , 'device_address'
        , 'end_point'
        , 'flag_delete'
        , 'create_date'
    ],
    'ah_gateway':[
        'gateway_id'
        , 'gateway_name'
        , 'gateway_xmpp_id'
        , 'gateway_mqtt_id'
        , 'flag_delete'
        , 'create_date'
        , 'modify_date'
    ],
    'ah_gateway_install':[
        'house_no'
        , 'gateway_id'
        , 'flag_delete'
        , 'create_date'
    ],
    'ah_house':[
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
    'ah_house_member':[
        'house_no'
        , 'user_no'
        , 'flag_delete'
        , 'create_date'
    ],
    'ah_usage_daily_predict':[
        'house_no'
        , 'collected_date'
        , 'use_energy'
        , 'predict_use_energy'
        , 'progressive_level'
        , 'create_date'
        , 'modify_date'
    ],
    'ah_usage_monthly':[
        'house_no'
        , 'year'
        , 'month'
        , 'use_energy'
        , 'predict_use_energy'
        , 'create_date'
        , 'modify_date'
    ],
    'ah_usage_monthly_byappliance':[
        'house_no'
        , 'appliance_no'
        , 'year'
        , 'month'
        , 'use_energy'
        , 'create_date'
        , 'modify_date'
    ],
    'ah_usage_monthly_bydevice':[
        'house_no'
        , 'device_address'
        , 'end_point'
        , 'year'
        , 'month'
        , 'use_energy'
        , 'create_date'
        , 'modify_date'
    ],
    'ah_user':[
        'user_no'
        , 'user_name'
        , 'email'
        , 'login_id'
        , 'login_password'
        , 'flag_delete'
        , 'create_date'
        , 'modify_date'
    ],
    'ah_use_log_byminute':[
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
    'ah_use_log_byminute_labled':[
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

classifications = {''}

regressions = {''}