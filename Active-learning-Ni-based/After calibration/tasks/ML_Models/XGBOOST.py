import random
import numpy as np
from sklearn.utils import resample
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

class XGBOOST:
    def __init__(self, random_seed=123):
        self.correction = False
        self.random_seed = random_seed
        random.seed(self.random_seed)  # 设置全局随机种子
        # 生成固定的随机种子列表，长度为 model_num
        self.random_seed_list = [random.randint(0, 10000) for _ in range(200)]  # 假设200次重采样

    def fit(self, X_train, y_train, X_test, y_test, model_num=200):
        """
        Training using predefined hyperparameters with resampling
        :param X_train: training features
        :param y_train: training labels
        :param X_test: testing features
        :param y_test: testing labels
        :param model_num: number of resampling iterations
        :return:
        """
        self.n_models = model_num
        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(X_train)
        self.X_test = self.sc.transform(X_test)
        self.y_train = y_train.ravel()

        # 定义超参数搜索范围
        n_estimators_range = [100, 200, 500]
        max_depth_range = [3, 5, 7]
        learning_rate_range = [0.01, 0.05, 0.1]
        subsample_range = [0.7, 0.8, 0.9]
        colsample_bytree_range = [0.7, 0.8, 0.9]

        best_score = -float('inf')
        best_params = {}

        # 遍历所有超参数组合
        for n_estimators in n_estimators_range:
            for max_depth in max_depth_range:
                for learning_rate in learning_rate_range:
                    for subsample in subsample_range:
                        for colsample_bytree in colsample_bytree_range:

                            # 设置 XGBoost 模型的参数
                            self.regressor = xgb.XGBRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                learning_rate=learning_rate,
                                subsample=subsample,
                                colsample_bytree=colsample_bytree,
                                n_jobs=-1,
                                tree_method='hist',
                                random_state=self.random_seed,  # 固定模型训练的随机性
                                device='cuda'  # 使用 GPU 训练
                            )

                            # 在训练集上训练模型
                            self.regressor.fit(self.X_train, self.y_train)

                            # 在测试集上进行验证并获取 R^2 分数
                            test_score = self.regressor.score(self.X_test, y_test)

                            # 保存最佳的超参数组合
                            if test_score > best_score:
                                best_score = test_score
                                best_params = {
                                    'n_estimators': n_estimators,
                                    'max_depth': max_depth,
                                    'learning_rate': learning_rate,
                                    'subsample': subsample,
                                    'colsample_bytree': colsample_bytree
                                }

        # 使用最佳超参数重新训练模型
        self.regressor = xgb.XGBRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            n_jobs=-1,
            tree_method='hist',
            random_state=self.random_seed,  # 固定模型训练的随机性
            device='cuda'  # 使用 GPU 训练
        )

    def predict(self, x_test, retstd=True):
        """
        Make predictions using resampling technique to compute mean and standard deviation
        :param x_test: testing features
        :param retstd: whether to return standard deviation
        :return: mean predictions and standard deviation
        """
        x_pred = self.sc.transform(x_test)
        pred_list = []

        for i in range(self.n_models):
            # 对训练集进行重采样，使用固定的随机种子列表中的种子
            xtrain_resampled, ytrain_resampled = resample(
                self.X_train, self.y_train, random_state=self.random_seed_list[i]
            )

            # 使用重采样的数据训练模型
            self.regressor.fit(xtrain_resampled, ytrain_resampled.ravel())

            # 记录对测试集的预测结果
            pred_list.append(self.regressor.predict(x_pred))

        # 转换为 NumPy 数组
        pred_list = np.array(pred_list)

        # 计算每个样本的预测均值和标准差
        pred_mean = pred_list.mean(axis=0)
        pred_std = pred_list.std(axis=0)

        if not retstd:
            return pred_mean

        # 如果启用了修正系数
        if self.correction:
            pred_std = (self.n_models / (self.n_models - 1) * pred_std ** 2) ** 0.5

        return pred_mean, pred_std
