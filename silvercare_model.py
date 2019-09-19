from utils import *

df = pd.read_csv('D:/rowdata/204.csv')
# df['ENERGY_DIFF'] = df['ENERGY_DIFF'] * 1000
lag=4
x, y = split_x_y(df, x_col='ENERGY_DIFF', y_col='APPLIANCE_STATUS')
x, y = sliding_window_transform(x, y, lag=lag, step_size=10)

model, params = select_classification_model('random forest')

gs = sk.model_selection.GridSearchCV(estimator=model,
                                     param_grid=params,
                                     cv=5,
                                     scoring='accuracy',
                                     n_jobs=-1)

gs.fit(x, y)
df = df.iloc[:-lag]
print(round(gs.best_score_*100, 2), '%', sep= '')

df.loc[:, 'appliance_status_predicted'] = gs.predict(x)

dump(gs, './sample_data/joblib/silvercare_model_2.joblib')

