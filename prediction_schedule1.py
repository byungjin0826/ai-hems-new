# 하루 치를 한번에
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier # partial fit : 중간에 업데이트를 할 수 있음.
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer
from joblib import load, dump

def data_load(name, device_name):
    df1 = pd.read_csv('./sample_data/csv/aihems/' + device_name + '(' + name + ')_01.csv',
                      encoding=encoding)  # 24일 이전 데이터 x
    df2 = pd.read_csv('./sample_data/csv/aihems/' + device_name + '(' + name + ')_02.csv',
                      encoding=encoding)  # 2일부터...
    df = pd.concat([df1, df2], ignore_index=True)

    df.columns.values[-1] = 'appliance_onoff'  # excel에 컬럼값 입력 안됨
    return(df)

def set_data(df):
    df1 = df.loc[df.energy != '\\N', :].copy()
    df1.loc[:, 'collected_time'] = [str(x).rjust(4, '0') for x in df1.collected_time]
    df1.loc[:, 'collected_time'] = [x[:2] + ':' + x[2:] for x in df1.collected_time]
    df1.loc[:, 'collected_date'] = [str(x) for x in df1.collected_date]
    df1.loc[:, 'date_time'] = df1.loc[:, 'collected_date'] + ' ' + df1.loc[:, 'collected_time']
    df1.loc[:, 'date_time'] = pd.to_datetime(df1.loc[:, 'date_time'])

    df1.loc[:, 'smart_plug_onoff'] = df1.loc[:, 'onoff']

    df1.loc[:, 'smart_plug_onoff'] = [int(x) for x in df1.smart_plug_onoff]
    df1.loc[df1.appliance_onoff.isna(), 'appliance_onoff'] = 0
    df1.loc[:, 'appliance_onoff'] = [int(x) for x in df1.appliance_onoff]

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

    df1.loc[:, 'energy'] = [float(x) for x in df1.energy]
    df1.loc[:, 'energy_lagged'] = df1.energy.shift(-1)
    df1.loc[:, 'energy_diff'] = df1.energy_lagged - df1.energy
    df1.iloc[-1, 8] = 0

    subset = df1.loc[:, ['date_time', 'date', 'dayofweek', 'daysinmonth', 'time',
                         'month', 'day', 'hour', 'minute',
                         'energy', 'energy_diff', 'smart_plug_onoff', 'appliance_onoff']].copy()
    # sebset
    return (subset)

def get_dummy(df, col_list):
    lb = LabelBinarizer()
    df_add_dummies = df.copy()
    for i in col_list:
        lb.fit(df[i])
        df_add_dummies[i+'_lb'] = [[y for y in x] for x in lb.transform(df_add_dummies[i])]
    return(df_add_dummies)

def X_and_Y(df_add_dummies, col_list):
    X = df_add_dummies.loc[:, col_list].values
    X_flatten = [sum(x, []) for x in X]
    # flat_list = [item for sublist in l for item in sublist]
    Y = np.array([np.array([x]) for x in df_add_dummies.loc[:, 'appliance_onoff']])
    Y = Y.ravel()
    return(X_flatten, Y)

def reverse_pivot_table(pivotted_df):
    pivotted_df.columns = [str(x) for x in pivotted_df.columns]
    pivotted_df.index.name = 'time'
    pivotted_df.columns.name = 'dayofweek'
    pivotted_df = pivotted_df.reset_index()
    reversed_pivot_table = pd.melt(pivotted_df, id_vars='time', value_vars=list(pivotted_df.columns[1:]),
                                   var_name='dayofweek', value_name='appliance_onoff_predicted')
    return (reversed_pivot_table)

def merge_table(df, reversed_pivot_table):
    # reversed_pivot_table.loc[: 'day_of_week'] = [str(x) for x in reversed_pivot_table]
    df = pd.merge(df, reversed_pivot_table, how='left', on=['time', 'dayofweek'])
    return (df)

def grid_search(model_name):
    model_list = {'Logistic Regression': [LogisticRegression(), # todo: 변수들 조정 필요한 것 설명 달기
                                          {
                                              'estimator__penalty': ['l2'],
                                              # 'estimator__dual': [False],
                                              # 'estimator__tol': [1e-4],

                                              # 'estimator__C': [1.0],
                                              'estimator__C': [0.001, 0.01, 0.1, 1, 10, 100],

                                              # 'estimator__fit_intercept':[True],
                                              # 'estimator__intercept_scaling':[1.],
                                              'estimator__class_weight':[None, 'balanced'],

                                              'estimator__random_state':[None],

                                              'estimator__solver':['liblinear', 'sag', 'saga'],

                                              'estimator__max_iter':[100],

                                              'estimator__multi_class':['ovr'],

                                              'estimator__verbose':[0],

                                              'estimator__warm_start': [False],

                                              'estimator__n_jobs':[None]
                                          }],
                  'Decision Tree': [DecisionTreeClassifier(),
                                    {
                                         'estimator__creterion': ['gini'],


                                         'estimator__splitter': ['best'],

                                         'estimator__max_depth':[None],

                                         'estimator__min_samples_split':[2],

                                         'estimator__min_samples_leaf':[1],

                                         'estimator__min_weight_fraction_leaf':[0.],

                                         'estimator__max_features':[None],

                                         'estimator__random_state':[None],

                                         'estimator__max_leaf_nodes':[None],

                                         'estimator__min_impurity_decrease':[0.],

                                         'estimator__min_impurity_split':[1e-7],

                                         'estimator__class_weight':[None],

                                         'estimator__presort':[False]
                                    }],
                  'Random Forest': [RandomForestClassifier(),
                                    {
                                        'n_estimators': [10],
                                        # 'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 1)],

                                        # 'criterion':['gini'],

                                        'max_depth': [None], #default None
                                        # 'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],

                                        'min_samples_split':[2], # default 2
                                        # 'min_samples_split': [2, 5, 10]

                                        'min_samples_leaf':[1], # default 1
                                        # 'min_samples_leaf': [1, 2, 4],  # default 1

                                        # 'min_weight_fraction_leaf':[0.],

                                        'max_features':['auto'],
                                        # 'max_features': ['sqrt', 'auto'],

                                        # 'max_leaf_nodes':[None],

                                        # 'min_impurity_decrease':[0.],

                                        # 'min_impurity_split':[1e-7],

                                        'bootstrap':[True],
                                        # 'bootstrap': [True, False]

                                        # 'oob_score':[False],

                                        # 'n_jobs':[None],

                                        # 'random_state':[None],

                                        # 'verbose':[0],

                                        # 'warm_start':[False],

                                        # 'class_weight':[None]
                                    }],
                  'Gaussian Naive Bayes': [GaussianNB(),
                                           {
                                            'priors': [None],
                                            'var_smoothing': [1e-9]
                                           }],
                  'MLP Classifier': [MLPClassifier(),
                                     {
                                          'estimator__hidden_layer_sizes':[(100,)],
                                          'estimator__activation':['relu'],
                                          'estimator__solver':['adma'],
                                          'estimator__alpha':[0.0001],
                                          'estimator__batch_size':['auto'],
                                          'estimator__learning_rate':['constant'],
                                          'estimator__learning_rate_init':[0.001],
                                          'estimator__power_t':[0.5],
                                          'estimator__max_iter':[200],
                                          'estimator__shuffle':[True],
                                          'estimator__random_state':[None],
                                          'estimator__tol':[1e-4],
                                          'estimator__verbose':[False],
                                          'estimator__warm_start':[False],
                                          'estimator__momentum':[0.9],
                                          'estimator__nesterovs_momentum':[True],
                                          'estimator__early_stopping':[False],
                                          'estimator__validation_fraction':[0.1],
                                          'estimator__beta_1':[0.9],
                                          'estimator__beta_2':[0.999],
                                          'estimator__epsilon': [1e-8],
                                          'estimator__n_iter_no_change': [10]
                                     }],
                  'k-Neighbors Classifier':[KNeighborsClassifier(),
                                            {
                                                 'estimator__n_neighbors': [5],
                                                 'estimator__weights': ['uniform'],
                                                 'estimator__algorithm':['auto'],
                                                 'estimator__leaf_size':[30],
                                                 'estimator__p':[2],
                                                 'estimator__metric':['minkowski'],
                                                 'estimator__metric_params':[None],
                                                 'estimator__n_jobs':[None]
                                            }],
                  'SGD Classifier': [SGDClassifier(),
                                     {
                                          'estimator__loss': ['hinge'],
                                          'estimator__penalty':['none'],
                                          'estimator__alpha':[0.0001],
                                          'estimator__l1_ratio':[0.15],
                                          'estimator__fit_intercept':[True],
                                          'estimator__max_iter':[5],
                                          'estimator__tol':[None],
                                          'estimator__shuffle':[True],
                                          # 'estimator__verbose':[],
                                          # 'estimator__epsilon':[],
                                          'estimator__n_jobs':[None],
                                          'estimator__random_state':[None],
                                          'estimator__learning_rate':['optimal'],
                                          'estimator__eta0':['optimal'],
                                          'estimator__power_t':[0.5],
                                          'estimator__early_stopping':[False],
                                          'estimator__validation_fraction':[0.1],
                                          'estimator__n_iter_no_change':[5],
                                          'estimator__class_weight':[None],
                                          # 'estimator__warm_start':[],
                                          # 'estimator__average':[],
                                          'estimator__n_iter':[None]
                                      }],
                  'SVM':[SVC(),
                         {
                             'estimator__C': [1.0],
                             'estimator__kernel':['rbf'],
                             'estimator__degree':[3],
                             'estimator__gamma':['auto'],
                             'estimator__coef0':[0.0],
                             'estimator__shrinking':[True],
                             'estimator__probability':[False],
                             'estimator__tol':[1e-3],
                             # 'estimator__cache_size':[],
                             # 'estimator__class_weight':[],
                             'estimator__verbose':[False],
                             'estimator__max_iter':[-1],
                             'estimator__decision_function_shape':['ovr'],
                             'estimator__random_state':[None]
                         }]
                  # 'one class SVM', []
    }
    grid_model = model_list[model_name][0]
    params = model_list[model_name][1]
    return(grid_model, params)

total_start = time.time()
encoding = 'euc-kr'

# name = input('사용자: ')
# device_name = input('기기명: ')
name = '박재훈'
device_name = 'TV'

# df = set_data(data_load(name, device_name))

df = data_load(name, device_name)

df = set_data(df)

df_add_dummies = get_dummy(df, ['dayofweek', 'day'])

X, Y = X_and_Y(df_add_dummies, ['hour', 'minute', 'dayofweek_lb', 'day_lb'])

model, params = grid_search('Random Forest')

gs = (GridSearchCV(estimator=model,
                   param_grid=params,
                   cv=2,
                   scoring = 'f1_micro', # accuracy, balanced_accuracy, average_precision, brier_score_loss,
                   n_jobs = -1))

fitting_start = time.time()
gs = gs.fit(X, Y)
fitting_end = time.time()

# Grid search CV 추가

testX = X
prediction_start = time.time()
testY = gs.predict(testX)
prediction_end = time.time()

# schedule_table = pd.DataFrame(testY.T, index=df.time.unique())
#
# reversed_pivot_table = reverse_pivot_table(schedule_table)
# merged_df = merge_table(df, reversed_pivot_table)

total_end = time.time()

print('가전기기명: ', device_name)
print('데이터 개수: ', len(df))
print('데이터 기간: ', df.date_time[0].date(), '~', df.date_time[len(df)].date())
print('Grid Search 결과: ', gs.best_score_)
print('Grid Search 결과: ', gs.best_estimator_)
print('=======================================================================================')
print('총 걸린 시간: ', round(total_end - total_start, 3))
print('학습에 걸린 시간: ', round(fitting_end - fitting_start, 3))
# print('예측에 걸린 시간: ', round(prediction_end - prediction_start, 3))
print('=======================================================================================')
# print('base line: ', round(len(merged_df.loc[merged_df.appliance_onoff==0, :])/len(merged_df), 3)) # 한 쪽으로 찍었을 때
# print('accuracy: ', round(accuracy_score(merged_df.appliance_onoff, merged_df.appliance_onoff_predicted), 3)) # 정확성
# print('precision: ', round(precision_score(merged_df.appliance_onoff, merged_df.appliance_onoff_predicted), 3)) # 정밀도
# print('recall: ', round(recall_score(merged_df.appliance_onoff, merged_df.appliance_onoff_predicted), 3)) # 재현율
# print('f1_score: ', round(f1_score(merged_df.appliance_onoff, merged_df.appliance_onoff_predicted), 3)) # 1에 가까울수록 좋음


from sklearn import svm
from sklearn import datasets
clf = svm.SVC(gamma='scale')
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)




dump(clf, 'filename.joblib')
import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(X[0:1])

