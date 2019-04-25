import pandas as pd
from db_connect import data_load
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV

def rand_times(n):
    """Generate n rows of random 24-hour times (seconds past midnight)"""
    rand_seconds = np.random.randint(0, 24*60*60, n)
    return pd.DataFrame(data=dict(seconds=rand_seconds))

def set_data(df, source = None):
    source = source or None

    if source == 'excel':
        df1 = df.loc[df.energy != '\\N', :].copy()  # db에서 load 할 때는 na로 들어옴.

    else:
        df1 = df.dropna()
        df1.loc[:, 'appliance_status'] = 0

    df1.loc[:, 'collected_time'] = [str(x).rjust(4, '0') for x in df1.collected_time]
    df1.loc[:, 'collected_time'] = [x[:2] + ':' + x[2:] for x in df1.collected_time]
    df1.loc[:, 'collected_date'] = [str(x) for x in df1.collected_date]
    df1.loc[:, 'date_time'] = df1.loc[:, 'collected_date'] + ' ' + df1.loc[:, 'collected_time']
    df1.loc[:, 'date_time'] = pd.to_datetime(df1.loc[:, 'date_time'])

    df1.loc[:, 'smart_plug_onoff'] = df1.loc[:, 'onoff']

    df1.loc[:, 'smart_plug_onoff'] = [int(x) for x in df1.smart_plug_onoff]
    df1.loc[df1.appliance_status.isna(), 'appliance_status'] = 0
    df1.loc[:, 'appliance_status'] = [int(x) for x in df1.appliance_status]

    df1.loc[:, 'date'] = [str(x.date()) for x in df1.date_time]
    df1.loc[:, 'dayofweek'] = [str(x.dayofweek) for x in df1.date_time]  # 0이 월요일

    df1.loc[:, 'time'] = [str(x.time()) for x in df1.date_time]
    df1.loc[:, 'month'] = [str(x.month) for x in df1.date_time]
    df1.loc[:, 'day'] = [str(x.day) for x in df1.date_time]
    df1.loc[:, 'hour'] = [int(x.hour) for x in df1.date_time]
    # df1.hour = [[x] for x in df1.hour]
    df1.loc[:, 'minute'] = [[int(x.minute)] for x in df1.date_time]
    df1.loc[:, 'holiday'] = [int(x)>4 for x in df1.dayofweek]
    # df1.minute= [[x] for x in df1.minute]

    df1.loc[:, 'dayofyear'] = [int(x.dayofyear) for x in df1.date_time]

    df1.loc[:, 'energy'] = [round(float(x)*1000) for x in df1.energy]
    df1.loc[:, 'energy_lagged'] = df1.energy.shift(-1)
    df1.loc[:, 'energy_diff'] = df1.energy_lagged - df1.energy
    df1.iloc[-1, 8] = 0
    df1 = df1.iloc[:-1]

    # df1.loc[:, 'gateway_id'] = 'ep17470141'
    df1.loc[:, 'end_point'] = 1
    df1.loc[:, 'quality'] = 100
    df1.loc[:, 'create_date'] = pd.datetime.today()

    return (df1)

target = 1

df = data_load(member_name='박재훈', appliance_name='TV')

df = set_data(df, source='excel')

if target == 0:
    df.loc[:, 'appliance_status'] = [1 if x == 0 else 0 for x in df.appliance_status]

# df.collected_time = [str(x).rjust(4, '0') for x in df.collected_time]

df['sin_time'] = [np.sin(x) for x in (df.hour * 60 + df.minute)/1440]
df['cos_time'] = [np.cos(x) for x in (df.hour * 60 + df.minute)/1440]

df_subset = df.loc[:, ['holiday', 'dayofweek', 'sin_time', 'cos_time', 'appliance_status']]

# df_subset = df.loc[df.appliance_status == 1, ['holiday', 'dayofweek', 'sin_time', 'cos_time', 'appliance_status']]

X_train = df_subset.loc[df_subset.appliance_status == 1, ['holiday', 'dayofweek', 'sin_time', 'cos_time']].values

Y_train = df_subset.loc[df_subset.appliance_status == 1, ['appliance_status']].values

model = OneClassSVM()

params = {
    'kernel':['rbf'] # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    , 'gamma':['scale']
    # , 'nu': np.linspace(start = 0.0001, stop = 1, endpoint= False, num = 11) # (0, 1]
    , 'nu': [0.5]
    , 'max_iter':[-1]
}

gs = (GridSearchCV(estimator=model,
                   param_grid=params,
                   cv=5,
                   scoring = 'f1', # accuracy, balanced_accuracy, average_precision, brier_score_loss,
                   n_jobs = -1))

gs.fit(X_train, Y_train)

X_test = df.loc[:, ['holiday', 'dayofweek', 'sin_time', 'cos_time']].values

Y_test = df.loc[:, ['appliance_status']].values

Y_pr = gs.predict(X_test)

Y_pr1 = [x if x==1 else 0 for x in Y_pr]

print(accuracy_score(Y_test, Y_pr1))

df.loc[:, 'predict'] = Y_pr1

df2 = df.loc[:, ['appliance_status', 'predict']]

len(df2.loc[(df2.appliance_status == 1) & (df2.predict == 0), :])
len(df2.loc[(df2.appliance_status == 0) & (df2.predict == 1), :])