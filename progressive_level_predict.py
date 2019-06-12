from utils import *
from joblib import dump
# progressive_level_predict

# 누진 구간 예측 절차
# 일별 예측...
# 모델 학습
# 기존

# gateway_id: ''
#

def make_model(house_no):
    sql = f"""
    SELECT *
    FROM AH_USAGE_DAILY_PREDICT
    WHERE 1=1
    AND HOUSE_NO = '{house_no}'
    """

    df = get_table_from_db(sql)

    x, y = split_x_y(df, x_col='use_energy_daily', y_col='use_energy_daily')

    x, y = sliding_window_transform(x, y, step_size=7, lag=0)

    x = x[6:-1]
    y = y[7:]

    # x, y = sliding_window_transform(x, y, step_size=7, lag=0) # todo: 과적합 해결

    model, params = select_regression_model('lasso regression')

    gs = sk.model_selection.GridSearchCV(estimator=model,
                                         param_grid=params,
                                         n_jobs=-1,
                                         cv=5)

    gs.fit(x, y)
    dump(gs, f'./sample_data/joblib/usage_daily/{house_no}.joblib')
    return gs, x, y


if __name__ == '__main__':
    gs, x, y = make_model('20190325000001')
    print(gs.best_score_)