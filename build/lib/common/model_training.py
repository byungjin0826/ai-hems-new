import datetime
import settings
import pandas as pd
import common.data_load as dl
import sklearn as sk
from joblib import dump, load


def make_model_elec(house_no):
    today = datetime.datetime.now().strftime('%Y%m%d')

    sql = f"""
SELECT *
FROM AH_USAGE_DAILY_PREDICT
WHERE 1=1
AND HOUSE_NO = {house_no}
AND USE_DATE >= DATE_FORMAT( DATE_ADD( STR_TO_DATE( '{today}', '%Y%m%d'),INTERVAL -28 DAY), '%Y%m%d')"""

    df = pd.read_sql(sql, con=settings.conn)

    df.loc[df.use_energy_daily.isnull(), 'use_energy_daily'] = 0

    x, y = dl.split_x_y(df, x_col='use_energy_daily', y_col='use_energy_daily')
    x, y = dl.sliding_window_transform(x, y, step_size=7, lag=0)

    x = x[6:-1]
    y = y[7:]

    model, param = dl.select_regression_model('linear regression')

    gs = sk.model_selection.GridSearchCV(estimator=model,
                                         param_grid=param,
                                         cv=5,
                                         n_jobs=-1)
    gs.fit(x, y)
    dump(gs, f'./sample_data/joblib/usage_daily/{house_no}.joblib')
    return gs.best_score_


def make_model_status(device_id, lag=10):
    sql = f"""
SELECT
    GATEWAY_ID
    , DEVICE_ID
    , COLLECT_DATE
    , COLLECT_TIME
    , QUALITY
    , ONOFF
    , ENERGY
    , ENERGY_DIFF
    , case when APPLIANCE_STATUS is null then 0 else APPLIANCE_STATUS end APPLIANCE_STATUS
    , CREATE_DATE
FROM
    AH_USE_LOG_BYMINUTE_LABELED_sbj
WHERE
    1 = 1
    AND DEVICE_ID = '{device_id}'
    AND COLLECT_DATE in (
        SELECT
            t1.COLLECT_DATE
        FROM
            (SELECT
                COLLECT_DATE
                , sum(APPLIANCE_STATUS) APPLIANCE_STATUS_SUM
            FROM 
                AH_USE_LOG_BYMINUTE_LABELED_sbj
            GROUP by
                COLLECT_DATE) t1
        WHERE 1=1
        AND t1.APPLIANCE_STATUS_SUM is not null)"""

    df = pd.read_sql(sql, con=settings.conn)

    x, y = dl.split_x_y(df, x_col='ENERGY_DIFF', y_col='APPLIANCE_STATUS')

    x, y = dl.sliding_window_transform(x, y, lag=lag, step_size=30)

    model, params = dl.select_classification_model('random forest')

    gs = sk.model_selection.GridSearchCV(estimator=model,
                                         param_grid=params,
                                         cv=5,
                                         scoring='accuracy',
                                         n_jobs=-1)

    gs.fit(x, y)

    df = df.iloc[:-lag]
    df.loc[:, 'appliance_status_predicted'] = gs.predict(x)
    dump_path = f'./sample_data/joblib/{device_id}_labeling.joblib'

    dump(gs, dump_path)  # 저장
    return dump_path, gs.best_score_


if __name__ == '__main__':
    print('hello')
