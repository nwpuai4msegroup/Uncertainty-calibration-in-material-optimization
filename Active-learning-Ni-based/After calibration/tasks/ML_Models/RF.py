import numpy as np
from lolopy.learners import RandomForestRegressor as lolo_RF
from sklearn.ensemble import RandomForestRegressor as sk_RF
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

class RF:
    def __init__(self):
        self.rf = None
        self.sc = None
        self.X_train = None
        self.y_train = None
        self.module = 'lolopy'  # 'lolopy'
        self.scale = True
        self.correction = False

    def fit(self, X_train, y_train):
        # Scale features
        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(X_train)
        # self.X_train = X_train
        self.y_train = y_train
        if self.module == 'sklearn':
            regressor = sk_RF()
            param_grid = {
                'n_estimators': [10, 20, 30, 50, 60, 100, 200, 300, 400, 500],
                'max_depth': [5, 6, 7, 8, 9, 10],
            }
            regressor = GridSearchCV(regressor, param_grid=param_grid, cv=5).fit(self.X_train, self.y_train.ravel())
            # print(regressor.best_params_)
            # print(regressor.best_score_)
            self.n_models = regressor.best_params_['n_estimators']
            self.rf = sk_RF(**regressor.best_params_).fit(self.X_train, self.y_train)
        if self.module == 'lolopy':
            # regressor = lolo_RF()
            self.rf = lolo_RF().fit(self.X_train, self.y_train)

    def predict(self, X):
        X = self.sc.transform(X)
        if self.module == 'sklearn':
            pred = self.rf.predict(X)
            est_preds = np.empty((len(X), len(self.rf.estimators_)))
            # loop over each tree estimator in forest and use it to predict
            for ind, est in enumerate(self.rf.estimators_):
                est_preds[:, ind] = est.predict(X)
            std = np.std(est_preds, axis=1)
        else:
            pred, std = self.rf.predict(X, return_std=True)
        return pred, std
