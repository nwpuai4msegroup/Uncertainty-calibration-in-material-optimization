import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR as svr
from sklearn.utils import resample
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

class SVR:
    def __init__(self, random_seed=123):
        self.correction = False
        self.random_seed = random_seed
        random.seed(self.random_seed)  # 设置全局随机种子
        # 生成固定的随机种子列表，长度为 model_num
        self.random_seed_list = [random.randint(0, 10000) for _ in range(200)]  # 假设200次重采样

    def fit(self, X_train, y_train, X_test, y_test, model_num=200):
        """
        Training parameters use GridSearchCV after normalization, return baggingregressor by bootstrap
        :param X_train:
        :param y_train:
        :param model_num:
        :return:
        """
        self.n_models = model_num
        self.sc = StandardScaler()

        # 标准化训练集和测试集
        self.X_train = self.sc.fit_transform(X_train)
        self.X_test = self.sc.transform(X_test)
        self.y_train = y_train.ravel()

        # 定义较大的超参数范围
        C_range = [0.5, 1, 5, 10, 100]  # C值的范围
        kernel_list = ['linear', 'rbf', 'poly']  # 核函数类型

        best_score = -float('inf')
        best_params = {}

        # 遍历所有的超参数组合
        for C in C_range:
            for kernel in kernel_list:

                # 设置SVR模型的参数
                self.regressor = svr(C=C, kernel=kernel)

                # 在训练集上训练模型
                self.regressor.fit(self.X_train, self.y_train)

                # 在测试集上进行验证并获取 R^2 分数
                test_score = self.regressor.score(self.X_test, y_test)

                # 保存最佳的超参数组合
                if test_score > best_score:
                    best_score = test_score
                    best_params = {'C': C, 'kernel': kernel}

        # 使用最佳超参数训练最终模型
        self.regressor = svr(C=best_params['C'], kernel=best_params['kernel'])

    def predict(self, x_test, retstd=True):
            """
            Make predictions using resampling technique to compute mean and standard deviation
            :param x_train: training features
            :param y_train: training labels
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
