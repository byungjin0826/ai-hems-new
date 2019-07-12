# house_no: '20190325000002'
import utils
import sklearn as sk
import sklearn.model_selection
from joblib import dump
import pandas as pd

house_no = '20190325000001'

sql = f"""
SELECT   * 
FROM AH_USAGE_DAILY_PREDICT
WHERE HOUSE_NO = '{house_no}'
AND USE_DATE != '20190417'
AND USE_DATE < '20190616'
ORDER BY USE_DATE
"""

df = utils.get_table_from_db(sql)

x, y = utils.split_x_y(df, x_col = 'use_energy_daily', y_col='use_energy_daily')

x, y = utils.sliding_window_transform(x, y, step_size= 7, lag = 0)

x = x[6:-1]

y = y[7:]


"""
random forest
linear regression
ridge regression
lasso regression
"""


model, param = utils.select_regression_model('linear regression')

gs = sk.model_selection.GridSearchCV(estimator=model,
                                     param_grid=param,
                                     cv=5,
                                     n_jobs=-1)

gs.fit(x, y)

print(gs.best_score_)


predicted_y = gs.predict(x)
comparison = pd.DataFrame({'y':y, 'pr_y':predicted_y})


dump(gs, f'./{house_no}.joblib')