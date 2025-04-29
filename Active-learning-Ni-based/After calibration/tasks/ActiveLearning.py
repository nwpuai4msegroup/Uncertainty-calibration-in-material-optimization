import math
import random
import time
import numpy as np
import pandas as pd
import uncertainty_toolbox as uct
from scipy.stats import norm, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from tqdm.auto import tqdm
from config import valid_dict, sl_config
from ML_Models import RF, NN, SVR, XGBOOST

valid_dict = {

    'models': [],
    'utility_functions': ['EI', 'MU', 'Random', 'MV', 'UCB', 'KG'],
    'model_metrics': ['mae', 'r2', 'mse', 'mape', 'rmse'],
    'uq_metrics': ['scc', 'mis_area', 'ence', 'nll', 'cv'],
    'al_metrics': ['n_found', 'oc'],
    'recalibraiton': False
}

class AL:
    def __init__(self, dataset, n_test, n_init, n_batch, utility_functions, n_trials, regressors,
                 target_feature, n_target, random_state, model_metrics, uq_metrics, al_metrics, file_name,
                 recalibration=False, save_results=True, n_iter=None):
        for key, value in sl_config.items():
            setattr(self, key, value)
        self.data = dataset
        # 最后一列取对数
        self.data.iloc[:, -1] = np.log(self.data.iloc[:, -1])
        random.seed(random_state)
        self.random_seeds = []
        for i in range(n_trials):
            self.random_seeds += [random.random()]

        # initial n_test
        if (type(n_test) == int) & (n_test < len(dataset)):
            self.n_test = n_test
        elif (type(n_test) == float) & (0 < n_test < 1):
            self.n_test = int(n_test * len(self.data))
        else:
            raise ValueError("Input wrong value of n_test!")
        self.n_al_data = len(self.data) - self.n_test
        # sampling test data
        random.seed(123)
        self.test_list = random.sample(list(self.data.index), self.n_test)

        # n_target
        if (type(n_target) == int) & (n_target < self.n_al_data):
            self.n_target = n_target
        elif (type(n_target) == float) & (0 < n_target < 1):
            self.n_target = int(n_target * self.n_al_data)
        else:
            raise ValueError("Input wrong value of n_target!")
        # n_init
        if (type(n_init) == int) & (n_init < self.n_al_data):
            self.n_init = n_init
        elif (type(n_init) == float) & (0 < n_init < 1):
            self.n_init = int(n_init * len(self.data))
        else:
            raise ValueError("Input wrong value of n_init!")
        # check n_test, n_target, n_init
        if self.n_init + self.n_target >= self.n_al_data:
            raise ValueError("The number of n_init and n_target out of range!")
        # n_iter
        if isinstance(n_iter, int) & (n_iter <= self.n_al_data - self.n_init):
            self.n_iter = n_iter
            self.find_the_best = False
        elif n_iter is None:
            self.find_the_best = True
        else:
            raise Warning("The number of n_iter out of range, and AL will search all data!")

        self.n_batch = n_batch
        self.n_trials = n_trials

        self.target_feature = target_feature
        self.X_features = [i for i in list(self.data.columns) if i not in [target_feature]]
        self.X = self.data[self.X_features].values
        self.y = self.data[self.target_feature].values

        if recalibration:
            self.uq_correction = 'recalibration'
        else:
            self.uq_correction = 'None'
        self.regressors = regressors
        self.utility_functions = utility_functions

        for k in ['model_metrics', 'uq_metrics', 'al_metrics']:
            if len([i for i in locals()[k] if i not in valid_dict[k]]):
                raise ValueError('{} is invalid {}! '.format(
                    str([i for i in locals()[k] if i not in valid_dict[k]]), k))
        self.model_metrics = model_metrics
        self.uq_metrics = uq_metrics
        self.al_metrics = al_metrics
        self.results = []
        self.save_results = save_results
        self.file_name = file_name

    def data_split(self, trials):
        """
        self.al_data is a sorted Dataframe
        :param trials:
        :return:
        """
        # sort data by values of target feature and select first n_target samples as target_data
        random.seed(self.random_seeds[trials])
        self.al_data = self.data.loc[[i for i in list(self.data.index) if i not in self.test_list], :].sort_values(
            by=self.target_feature, ascending=False)
        self.target_list = list(self.al_data.iloc[:self.n_target, :].index)
        self.target_value = self.al_data.at[self.target_list[-1], self.target_feature]
        # sample initial data from residual data randomly
        # self.init_list = random.sample(list(self.al_data.iloc[self.n_target:, :].index), self.n_init)

        # 直接选择排序后剩余数据中最小的 n_init 个样本
        self.init_list = list(self.al_data.iloc[self.n_target:, :].index[-self.n_init:])

        # split data as measured and unmeasured input to modeling which will change dynamically in active learning
        self.measured_list = self.init_list
        self.unmeasured_list = [i for i in list(self.al_data.index) if i not in self.init_list]

    def train_model(self, model):
        """
        :param model: callable
        :return: prediction and std estimate of all samples(numpy.ndarray)
        """
        regressor = globals()[model]()
        regressor.fit(X_train=self.X[self.measured_list], y_train=self.y[self.measured_list],
                      X_test=self.X[self.test_list], y_test=self.y[self.test_list])
        y_pred, y_std = regressor.predict(self.X)
        return y_pred, y_std

    def EI_scoring(self, y_pred, y_std):
        best_so_far = self.data.loc[self.measured_list, self.target_feature].max()
        cost = (y_pred - best_so_far) / y_std
        EI = y_std * cost * norm.cdf(cost, 0, 1) + y_std * norm.pdf(cost, 0, 1)
        return EI


    # def KG_scoring(self, y_pred, y_std):
    #     max_pre = y_pred.max()
    #     max_mean = self.data.loc[self.measured_list, self.target_feature].max()
    #     cost = -1 * abs((y_pred - max(max_pre, max_mean)) / y_std)
    #     KG = y_std * cost * norm.cdf(cost, 0, 1) + y_std * norm.pdf(cost, 0, 1)
    #     return KG

    def calculate_metrics(self, y_pred, y_std, result_dict):
        for i in self.model_metrics:
            if i == 'mae':
                result_dict[i] = mean_absolute_error(y_pred[self.test_list], self.y[self.test_list])
            if i == 'r2':
                result_dict[i] = r2_score(y_pred[self.test_list], self.y[self.test_list])
            if i == 'mse':
                result_dict[i] = mean_squared_error(y_pred[self.test_list], self.y[self.test_list])
            if i == 'rmse':
                result_dict[i] = np.sqrt(mean_squared_error(y_pred[self.test_list], self.y[self.test_list]))
            if i == 'mape':
                result_dict[i] = mean_absolute_percentage_error(y_pred[self.test_list], self.y[self.test_list])
        for j in self.uq_metrics:
            if j == 'scc':
                result_dict[j] = \
                    spearmanr(abs(y_pred[self.test_list] - self.y[self.test_list]), y_std[self.test_list])[0]
            if j == 'mis_area':
                result_dict[j] = uct.metrics.miscalibration_area(y_pred[self.test_list],
                                                                 y_std[self.test_list],
                                                                 self.y[self.test_list])
            if j == 'cv':
                mean = np.mean(y_std[self.test_list])
                std = np.std(y_std[self.test_list])
                result_dict[j] = std / mean
            if j == 'ence':
                std_list = pd.Series(y_std[self.test_list]).sort_values(ascending=False).index
                std_list = [self.test_list[i] for i in std_list]
                n_bin = 10
                len_bin = math.ceil(len(std_list) / n_bin)
                rmse = []
                rmv = []
                for i in range(0, len(std_list), len_bin):
                    list_i = list(std_list)[i:i + len_bin]
                    rmv.append(y_std[list_i].mean() ** 0.5)
                    rmse.append(mean_squared_error(self.y[std_list][i:i + n_bin], y_pred[std_list][i:i + n_bin]) ** 0.5)
                ence = np.abs(np.array(rmv) - np.array(rmse)) / np.array(rmv)
                result_dict[j] = ence.mean()
            if j == 'nll':
                pass
        for k in self.al_metrics:
            if k == 'n_found':
                result_dict['n_found'] = np.sum(self.y[self.measured_list] == self.target_value)
            if k == 'oc':
                train_max_value = np.max(self.y[self.measured_list])
                result_dict['oc'] = self.target_value - train_max_value
        return result_dict

    def select_candidate(self, unmeasured_pred, unmeasured_std, utility_function):
        """
        Select the list of candidates and add them to measured samples
        :param unmeasured_pred: list, prediction of samples
        :param unmeasured_std: list, standard devation of samples (uncertainty estimate)
        :param utility_function: str, type of utility function
        :return:
        """
        if utility_function == 'Random':
            self.candidates_list = random.sample([i for i in self.unmeasured_list], self.n_batch)
        elif utility_function == 'MU':
            if (np.array(unmeasured_std) == 0).all():
                raise Warning('y_std are all 0 value, MU will not work normally!')
            candidates = pd.Series(unmeasured_std).sort_values(ascending=False).index[:self.n_batch]
            self.candidates_list = [self.unmeasured_list[i] for i in candidates]
        elif utility_function == 'EV':
            candidates = pd.Series(unmeasured_pred).sort_values(ascending=False).index[:self.n_batch]
            self.candidates_list = [self.unmeasured_list[i] for i in candidates]
        elif utility_function == 'EI':
            if (np.array(unmeasured_std) == 0).any():
                raise ValueError('y_std contains 0 value, EI is invalid!')
            EI = self.EI_scoring(unmeasured_pred, unmeasured_std)
            candidates = pd.Series(EI).sort_values(ascending=False).index[:self.n_batch]
            self.candidates_list = [self.unmeasured_list[i] for i in candidates]
        elif utility_function == 'harmonic_mean':
            candidates = pd.Series((unmeasured_pred * unmeasured_std)/(unmeasured_pred + unmeasured_std)).sort_values(ascending=False).index[:self.n_batch]
            self.candidates_list = [self.unmeasured_list[i] for i in candidates]
        elif utility_function == 'UCB':
            candidates = pd.Series(unmeasured_pred + unmeasured_std).sort_values(ascending=False).index[:self.n_batch]
            self.candidates_list = [self.unmeasured_list[i] for i in candidates]
        elif utility_function == 'KG':
            pass
        else:
            raise ValueError('{} acquisition function is invalid'.format(utility_function))
        self.measured_list += self.candidates_list
        self.unmeasured_list = [i for i in self.unmeasured_list if i not in self.candidates_list]

    def single_trial(self, model, utility_function, trials):
        self.data_split(trials)
        print('target_value:', self.target_value)
        desc = 'Regressor:{} Strategy:{} Trials:{}'.format(model, utility_function, trials)

        if self.find_the_best:
            i = 0
            while True:
                y_pred, y_std = self.train_model(model)

                # 校准: 使用 measured 和 test 数据进行校准
                if self.uq_correction == 'recalibration':
                    recalibrator = uct.recalibration.get_std_recalibrator(
                        np.concatenate([y_pred[self.measured_list], y_pred[self.test_list]]),
                        np.concatenate([y_std[self.measured_list], y_std[self.test_list]]),
                        np.concatenate([self.y[self.measured_list], self.y[self.test_list]])
                    )
                    y_std = recalibrator(y_std)

                self.select_candidate(y_pred[self.unmeasured_list], y_std[self.unmeasured_list], utility_function)
                result = {
                    'regressor': model,
                    'utility_function': utility_function,
                    'trial': trials,
                    'iteration': i,
                    'candidates_value': list(self.y[self.candidates_list]),
                    'target_value': self.target_value,
                    'uq_correction': self.uq_correction,
                }
                result = self.calculate_metrics(y_pred, y_std, result)
                self.results.append(result)
                i += 1

                if max(self.y[self.candidates_list]) >= self.target_value:
                    break
        else:
            for i in tqdm(range(self.n_iter), desc=desc, colour='green', leave=True):
                y_pred, y_std = self.train_model(model)

                # 校准: 使用 measured 和 test 数据进行校准
                if self.uq_correction == 'recalibration':
                    recalibrator = uct.recalibration.get_std_recalibrator(
                        np.concatenate([y_pred[self.measured_list], y_pred[self.test_list]]),
                        np.concatenate([y_std[self.measured_list], y_std[self.test_list]]),
                        np.concatenate([self.y[self.measured_list], self.y[self.test_list]])
                    )
                    y_std = recalibrator(y_std)

                self.select_candidate(y_pred[self.unmeasured_list], y_std[self.unmeasured_list], utility_function)
                result = {
                    'regressor': model,
                    'utility_function': utility_function,
                    'trial': trials,
                    'iteration': i,
                    'candidates_value': list(self.y[self.candidates_list]),
                    'target_value': self.target_value,
                    'uq_correction': self.uq_correction,
                }
                result = self.calculate_metrics(y_pred, y_std, result)
                self.results.append(result)

    def run_trials(self):
        self.start_workflow_time = time.time()
        self.results = []
        for model in self.regressors:
            for strategy in self.utility_functions:
                if isinstance(self.n_iter, int):
                    iter_list = range(self.n_trials)
                else:
                    iter_list = tqdm(range(self.n_trials), desc='find_the_target', colour='green', leave=True)
                for i in iter_list:
                    self.single_trial(model, strategy, i)
                results = pd.DataFrame(self.results)
                if self.save_results:
                    results.to_csv(
                        path_or_buf='../results/' + self.file_name)
        self.total_runtime = round((time.time() - self.start_workflow_time) / 60, 2)
        print('Total time: {} min'.format(self.total_runtime))
