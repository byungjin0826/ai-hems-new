import pandas as pd
import settings
from functools import wraps
import sklearn as sk
from joblib import load, dump


def pandas_read_sql(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        return pd.read_sql(f(*args, **kwargs), con=settings.conn)
    return decorated


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


def sliding_window_transform(x, y, step_size=10, lag=2):  # todo: 1. X가 여러개의 컬럼일 때도 동작할 수 있도록
    """
    상태 판별 예측을 위한 입력 데이터 변환
    :param x: 분 단위 전력 사용량
    :param step_size: Sliding window 의 사이즈
    :param lag: 숫자만큼 지연
    :return:
    """
    # todo: lag가 - 값이여도 동작하게 만들기
    x = [x for x in x]
    y = [x for x in y]
    x = [0] * (step_size - 1) + x
    x_transformed = [x[i - step_size + lag:i + lag] for i in range(len(x) + 1 - lag) if i > step_size - 1]
    if lag == 0:
        y_transformed = y
    else:
        y_transformed = y[:-lag]
    return x_transformed, y_transformed  #


def select_regression_model(model_name):
    regressions = {
        'random forest': [
            sk.ensemble.RandomForestRegressor(),
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
                # 'alpha': []
                # , 'fit_intercept': []
                 'normalize': [False]
                , 'copy_X': [True]
                # , 'max_iter': []
                # , 'tol': []
                , 'solver': ['auto']
                , 'random_state': [None]
            }
        ],
        'lasso regression': [
            sk.linear_model.Lasso(),
            {
                # 'alpha': []
                 'fit_intercept': [True]
                , 'normalize': [False]
                , 'precompute': [False]
                , 'copy_X': [True]
                # , 'max_iter': []
                # , 'tol': []
                # , 'warm_start': []
                # , 'positive': []
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


def usage_log(device_id, gateway_id=None, start_date='20191128', end_date=None,
              start_time='0000', end_time='2359', dayofweek=None, raw_data=False,
              sql_print = False, power = False, threshold = 1):
    """

    :param device_id:
    :param gateway_id:
    :param start_date:
    :param end_date:
    :param start_time:
    :param end_time:
    :param dayofweek:
    :param raw_data:
    :param sql_print:
    :return:
    """

    def make_sql():
        sql = f"""
SELECT 
    STR_TO_DATE(CONCAT(COLLECT_DATE, COLLECT_TIME), '%Y%m%d%H%i') DATETIME
    , POWER
    , ENERGY_DIFF
    , ONOFF
    , APPLIANCE_STATUS
FROM AH_USE_LOG_BYMINUTE
WHERE 1=1
AND DEVICE_ID = '{device_id}'"""

        if power:
            sql = f"""
SELECT 
    STR_TO_DATE(CONCAT(COLLECT_DATE, COLLECT_TIME), '%Y%m%d%H%i') DATETIME
    , POWER
    , ENERGY_DIFF
    , ONOFF
    , CASE WHEN POWER >= {threshold} THEN 1 ELSE 0 END APPLIANCE_STATUS
FROM AH_USE_LOG_BYMINUTE
WHERE 1=1
AND DEVICE_ID = '{device_id}'"""

        if gateway_id is not None:
            gateway_id_condition = f"\nAND GATEWAY_ID = '{gateway_id}'"
            sql += gateway_id_condition

        start_date_condition = f"\nAND COLLECT_DATE >= '{start_date}'"
        sql += start_date_condition

        if end_date is not None:
            end_date_condition = f"\nAND COLLECT_DATE <= '{end_date}'"
            sql += end_date_condition

        start_time_condition = f"\nAND COLLECT_TIME >= '{start_time}'"
        sql += start_time_condition

        end_time_condition = f"AND COLLECT_TIME <= '{end_time}'"
        sql += end_time_condition

        if dayofweek is not None:
            dayofweek_condition = f"AND DAYOFWEEK(COLLECT_DATE) = {dayofweek}"
            sql += dayofweek_condition
        return sql

    def transform(df=pd.read_sql(make_sql(), con=settings.conn, index_col='DATETIME')):
        df['APPLIANCE_STATUS_LAG'] = df.APPLIANCE_STATUS.shift(1).fillna(0)
        df['APPLIANCE_STATUS_LAG'] = [int(x) for x in df.APPLIANCE_STATUS_LAG]
        df['APPLIANCE_STATUS'] = df.APPLIANCE_STATUS.fillna(0)
        df['APPLIANCE_STATUS'] = [int(x) for x in df.APPLIANCE_STATUS]

        df_change = df.loc[df.APPLIANCE_STATUS != df.APPLIANCE_STATUS_LAG, :]
        df_change = df_change.reset_index()
        df_change['DATETIME_LAG'] = df_change.DATETIME.shift(1).fillna(0)
        df_change['duration'] = df_change.DATETIME - df_change.DATETIME.shift(1)

        print(df_change)
        df_change['duration'] = [x.days * 1440 + x.seconds / 60 for x in df_change.duration if x != 'NaT']

        history = df_change.loc[:, ['DATETIME_LAG', 'DATETIME', 'duration', 'APPLIANCE_STATUS_LAG']]

        history.columns = ['START', 'END', 'DURATION', 'STATUS']
        history = history.loc[history.STATUS == 1, :].reset_index(drop=True)
        return history

    if sql_print:
        print(make_sql())

    if raw_data:
        result = pd.read_sql(make_sql(), con=settings.conn, index_col='DATETIME')

    else:
        result = transform()

    # result.to_clipboard()
    # settings.conn.close()
    return result


def status(device_id):
    sql = f"""
SELECT *
FROM aihems_service_db.ah_log_socket
WHERE 1=1
AND device_id = '{device_id}'
AND 
"""

    # status = pd.read_sql(sql, con=settings.conn)

    return sql


def status_all_device(gateway_id):
    device_sal = f"""
{gateway_id}
"""
    return device_sal


# 편리하게 찾을 수 있는 기능 구현...
def device_info(device_name=None, gateway_id=None, gateway_name=None, house_name=None,
                house_id=None, energy_info=None):
    def sql():
        device_info_sql = f"""
SELECT *
FROM
    (SELECT
        t01.DEVICE_ID
        , t01.DEVICE_NAME
        , t06.APPLIANCE_NAME
        , t01.DEVICE_TYPE
        , t01.FLAG_USE_AI
        , t02.GATEWAY_ID
        , t03.APPLIANCE_NO
        , t04.HOUSE_NO
        , t05.HOUSE_NAME
    FROM
        AH_DEVICE t01
    INNER JOIN
        AH_DEVICE_INSTALL t02
    ON t01.DEVICE_ID = t02.DEVICE_ID
    INNER JOIN
        AH_APPLIANCE_CONNECT t03
    ON t01.DEVICE_ID = t03.DEVICE_ID
    INNER JOIN
        AH_GATEWAY_INSTALL t04
    ON t03.GATEWAY_ID = t04.GATEWAY_ID
    INNER JOIN
        AH_HOUSE t05
    ON t04.HOUSE_NO = t05.HOUSE_NO
    INNER JOIN
        AH_APPLIANCE t06
    ON t03.APPLIANCE_NO = t06.APPLIANCE_NO
    WHERE 1=1
    AND t06.FLAG_DELETE = 'N'
    AND t01.FLAG_DELETE = 'N'
    AND t03.FLAG_DELETE = 'N') t
WHERE 1=1"""

        if device_name is not None:
            device_name_condition = f"\nAND DEVICE_NAME like '%{device_name}%'"
            device_info_sql += device_name_condition

        if gateway_id is not None:
            gateway_id_condition = f"\nAND GATEWAY_ID = '{gateway_id}'"
            device_info_sql += gateway_id_condition

        if gateway_name is not None:
            gateway_name_condition = f"\nAND GATEWAY_NAME = '{gateway_name}'"
            device_info_sql += gateway_name_condition

        if house_name is not None:
            house_name_condition = f"\nAND HOUSE_NAME like '%{house_name}%'"
            device_info_sql += house_name_condition

        if house_id is not None:
            house_id_condition = f""
            device_info_sql += house_id_condition

        return device_info_sql

    df = pd.read_sql(sql(), con=settings.conn)
    # settings.conn.close()
    return df


def house_info():
    return 0


def label_modify(device_id='000D6F0012577B441', threshold = 1,
                 collect_date_range=('20191101', '20191130'), collect_time_range=('0000', '2359')):
    label0 = f"""
UPDATE AH_USE_LOG_BYMINUTE
SET APPLIANCE_STATUS = 0
WHERE 1=1
AND DEVICE_ID = '{device_id}'
AND COLLECT_DATE >= '{collect_date_range[0]}'
AND COLLECT_DATE <= '{collect_date_range[1]}'
AND COLLECT_TIME >= '{collect_time_range[0]}'
AND COLLECT_TIME <= '{collect_time_range[1]}'
AND POWER < {threshold}
"""
    settings.curs.execute(label0)

    label1 = f"""
UPDATE AH_USE_LOG_BYMINUTE
SET APPLIANCE_STATUS = 1
WHERE 1=1
AND DEVICE_ID = '{device_id}'
AND COLLECT_DATE >= '{collect_date_range[0]}'
AND COLLECT_DATE <= '{collect_date_range[1]}'
AND COLLECT_TIME >= '{collect_time_range[0]}'
AND COLLECT_TIME <= '{collect_time_range[1]}'
AND POWER >= {threshold}
    """
    settings.curs.execute(label1)
    settings.conn.commit()
    settings.conn.close()
    settings.conn.connect()


def sql_select(sql):
    return pd.read_sql(sql, con=settings.conn)


def iter_predict(x, n_iter, model):
    y = []
    for i in range(n_iter):
        y_temp = model.predict([x]).item()
        if y_temp < 0:
            y_temp = y_temp * -1
        y.append(y_temp)
        x_temp = x[1:]
        x_temp.append(y_temp)
        x = x_temp
    return y


def predict_elec(house_no, date):
    sql = f"""
SELECT
    * 
FROM
    AH_USAGE_DAILY_PREDICT
WHERE 1=1
AND HOUSE_NO = '{house_no}'
AND USE_DATE >= DATE_FORMAT( DATE_ADD( STR_TO_DATE( '{date}', '%Y%m%d'),INTERVAL -7 DAY), '%Y%m%d')
AND USE_DATE < '{date}'
ORDER BY
USE_DATE"""

    df = pd.read_sql(sql, con=settings.conn)

    # elec = [x for x in df.use_energy_daily.values[-7:]]
    elec = [x for x in df.USE_ENERGY_DAILY.values[-7:]]

    model = load(f'./sample_data/joblib/usage_daily/{house_no}.joblib') # 여기서 오류 발생.

    y = iter_predict(x=elec, n_iter=31, model=model)
    return y


def labeling(device_id, gateway_id, collect_date):
    def using_rf_model():
        start = collect_date + '0000'
        end = collect_date + '2359'

        sql = f"""
    SELECT    *
    FROM      AH_USE_LOG_BYMINUTE
    WHERE      1=1
       AND   GATEWAY_ID = '{gateway_id}'
       AND   DEVICE_ID = '{device_id}'
       AND   CONCAT( COLLECT_DATE, COLLECT_TIME) >= DATE_FORMAT( DATE_ADD( STR_TO_DATE( '{start}', '%Y%m%d%H%i'),INTERVAL -20 MINUTE), '%Y%m%d%H%i')
         AND   CONCAT( COLLECT_DATE, COLLECT_TIME) <= DATE_FORMAT( DATE_ADD( STR_TO_DATE( '{end}', '%Y%m%d%H%i'),INTERVAL 10 MINUTE), '%Y%m%d%H%i')
    ORDER BY COLLECT_DATE, COLLECT_TIME"""

        df = pd.read_sql(sql, con=settings.conn, index=False)
        print(df.head())
        print('df:', len(df))

        x, y = split_x_y(df, x_col='energy_diff')

        pre = 20
        post = 10
        length = post + pre

        x = [x[i:i + length] for i in range(len(x) - (pre + post))]

        model = load(f'./sample_data/joblib/by_device/{device_id}_labeling.joblib')

        y = model.predict(x)

        y = [int(x) for x in y]
        return y

    sql = f"""
SELECT *
FROM 
"""
    return y


def get_dr_info(request_dr_no):
    sql = f"""
SELECT *
FROM AH_DR_REQUEST
WHERE 1=1
AND REQUEST_DR_NO = '{request_dr_no}'"""

    df = pd.read_sql(sql, con=settings.conn)
    dr_type = df.iloc[0, 1]

    duration = int((df.iloc[0, 3] - df.iloc[0, 2]).seconds/60)
    return dr_type, duration


def test(house_no):
    path = f'./sample_data/joblib/usage_daily/{house_no}.joblib'
    # model = load(path)  # 여기서 오류 발생.
    return path


if __name__ == '__main__':
    # device_id = device_info(device_name='TV', house_name='안채').DEVICE_ID[0]
    # log = usage_log(device_id=device_id, start_date='20191101', power=True, threshold=1)
    # raw = usage_log(device_id=device_id, start_date='20191101', dayofweek=2, raw_data=True)
    # label_modify(device_id=device_id, appliance_status=0, collect_date='20191111', collect_time_range=['2358', '2359'])
    # log = usage_log(device_id=device_id, start_date='20191101', power=False)
    # list = [['20191108', ['2016', '2017']],
    #         ['20191111', ['2358', '2359']],
    #         ['20191114', ['2349', '2359']],
    #         ['20191115', ['0552', '0555']],
    #         ['20191115', ['2357', '2359']],
    #         ['20191116', ['2016', '2017']],
    #         ['20191117', ['2350', '2359']],
    #         ['20191121', ['2245', '2246']],
    #         ['20191121', ['2359', '2359']],
    #         ['20191124', ['1646', '1653']],
    #         ['20191130', ['0000', '0003']],
    #         ['20191130', ['1912', '1921']],
    #         ['20191203', ['0000', '0003']]]
    #
    # for date, time in list:
    #     # print(f'date: {date}, time: {time}')
    #     label_modify(device_id=device_id, appliance_status=0,
    #                  collect_date=date, collect_time_range=time)
#     sql = f"""
# SELECT
#     DEVICE_ID
#     , CASE WHEN APPLIANCE_STATUS = 1 THEN 'ON' ELSE 'OFF' END APPLIANCE_STATUS
#     , SUM(ENERGY_DIFF) ENERGY
#     , COUNT(*) DURATION
# FROM AH_USE_LOG_BYMINUTE
# WHERE 1=1
# AND GATEWAY_ID = 'ep18270236'
# AND COLLECT_DATE = '20191204'
# AND APPLIANCE_STATUS is not NULL
# -- AND COLLECT_DATE
# -- AND
# GROUP BY
#     DEVICE_ID
#     , APPLIANCE_STATUS"""
#
#     df = sql_select(sql)
#     df_pivot = df.pivot_table(columns = 'APPLIANCE_STATUS', index = 'DEVICE_ID', values = 'DURATION')
#     label_modify(device_id='00158D0001A42DA51', collect_date_range=('20191101', '20191209'), threshold=2)
    request_dr_no = '2019111503'
    sql = f"""
    SELECT *
    FROM AH_DR_REQUEST
    WHERE 1=1
    AND REQUEST_DR_NO = '{request_dr_no}'"""

    df = pd.read_sql(sql, con=settings.conn)




