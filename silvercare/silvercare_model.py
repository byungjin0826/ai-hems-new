from utils import *


# 104호 : 104, 106, 206, 111
# 204호 : 204
# 205호 : 그 외
#

df = pd.read_csv('D:/rowdata/205.csv')
# df['ENERGY_DIFF'] = df['ENERGY_DIFF'] * 1000
lag=10
x, y = split_x_y(df, x_col='ENERGY_DIFF', y_col='APPLIANCE_STATUS')
x, y = sliding_window_transform(x, y, lag=lag, step_size=31)

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

dump(gs, './sample_data/joblib/205_labeling.joblib')

