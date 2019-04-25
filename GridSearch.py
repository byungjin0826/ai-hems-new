from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe_svr = Pipeline([('scl', StandardScaler()),
        ('reg', MultiOutputRegressor(SVR()))])

grid_param_svr = {
    'reg__estimator__C': [0.1,1,10]
}

gs_svr = (GridSearchCV(estimator=pipe_svr,
                      param_grid=grid_param_svr,
                      cv=2,
                      scoring = 'neg_mean_squared_error',
                      n_jobs = -1))

gs_svr = gs_svr.fit(X_train, y_train)
gs_svr.best_estimator_

Pipeline(steps=[('scl', StandardScaler(copy=True, with_mean=True, with_std=True)),
('reg', MultiOutputRegressor(estimator=SVR(C=10, cache_size=200,
 coef0=0.0, degree=3, epsilon=0.1, gamma='auto', kernel='rbf', max_iter=-1,
 shrinking=True, tol=0.001, verbose=False), n_jobs=1))])