from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import numpy as np
from sklearn.pipeline import Pipeline
import pandas as pd

X, y1 = make_classification(n_samples=10, n_features=100, n_informative=30, n_classes=3, random_state=1)

y2 = shuffle(y1, random_state=1)
y3 = shuffle(y1, random_state=2)

Y = np.vstack((y1, y2, y3)).T
n_samples, n_features = X.shape # 10,100
n_outputs = Y.shape[1] # 3
n_classes = 3
forest = RandomForestClassifier(n_estimators=100, random_state=1)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
multi_target_forest.fit(X, Y).predict(X)

pipe_svr = Pipeline([('svc', MultiOutputClassifier(svm.SVC()))])
#
# iris = datasets.load_iris()
# parameters = {'svc__kernel': ('linear', 'rbf'), 'svc__C': [1, 10]}
# svc = svm.SVC(gamma="scale")
# multi_target_forest = MultiOutputClassifier(svc, n_jobs=-1)
parameters = {'svc__kernel': ('linear', 'rbf'), 'svc__C': [1, 10]}
clf = GridSearchCV(estimator=pipe_svr, param_grid=pipe_svr, cv=5)
clf.fit(X, Y)

gs_svr = (GridSearchCV(estimator=pipe_svr,
                      param_grid=parameters,
                      cv=2,
                      scoring = 'neg_mean_squared_error',
                      n_jobs = -1))

sorted(clf.cv_results_.keys())

