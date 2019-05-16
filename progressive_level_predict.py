from utils import *

# progressive_level_predict

# 누진 구간 예측 절차
# 일별 예측...
# 모델 학습
# 기존

# gateway_id: ''
#

def make_model(name):
    gateway_id = get_gateway_id(name)
    device_list = get_device_list(gateway_id)

    df = get_raw_data(gateway_id=gateway_id, start = '20180801')

    df = binding_time(df)

    df_daily_sum = df.loc[:, ['energy_diff']].resample('1d').sum()

    df_daily_sum.loc[:, 'dayofweek'] = [x for x in df_daily_sum.index.dayofweek]

    x, y = split_x_y(df_daily_sum, x_col='energy_diff', y_col='energy_diff')
    x, y = sliding_window_transform(x, y, step_size=8, lag=0)

    x = x[7:]
    x = [x[:-1] for x in x]
    y = y[7:]

    # x = [[x] for x in x]

    model, params = select_regression_model('ridge regression')

    gs = sk.model_selection.GridSearchCV(estimator=model,
                                         param_grid=params,
                                         n_jobs=-1,
                                         cv=5)

    gs.fit(x, y)
    return gs


