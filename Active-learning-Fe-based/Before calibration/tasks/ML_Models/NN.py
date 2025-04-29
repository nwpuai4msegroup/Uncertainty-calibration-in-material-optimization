import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetRegressor
import torch
import torch.nn.functional as func

# 定义神经网络结构
class NNRegressor(torch.nn.Module):
    def __init__(self, dim_x):
        super(NNRegressor, self).__init__()
        self.input = torch.nn.Linear(dim_x, 100)
        self.dense1 = torch.nn.Linear(100, 200)
        self.dense2 = torch.nn.Linear(200, 200)
        self.dropout = torch.nn.Dropout(0.5)
        self.output = torch.nn.Linear(200, 1)

    def forward(self, x):
        x = self.input(x)
        x = func.relu(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        output = func.relu(x)
        output = self.output(output)
        return output

class NN:
    def __init__(self, random_seed=123):
        self.nn = []
        self.correction = False
        self.random_seed = random_seed
        random.seed(self.random_seed)  # 设置全局随机种子
        torch.manual_seed(self.random_seed)  # 设置 PyTorch 的随机种子
        # 生成固定的随机种子列表，用于每次模型的训练
        self.random_seed_list = [random.randint(0, 10000) for _ in range(200)]

        # 判断是否有可用的GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 内部定义的学习率范围
        self.lr_range = [0.01, 0.1, 0.2]

    def fit(self, X_train, y_train, X_test, y_test, model_num=200):
        torch.set_default_tensor_type(torch.FloatTensor)
        self.n_models = model_num
        self.sc_x = StandardScaler()
        self.sc_y = StandardScaler()
        self.X_train = self.sc_x.fit_transform(X_train)
        self.y_train = self.sc_y.fit_transform(y_train.reshape(-1, 1))
        self.X_test = self.sc_x.transform(X_test)
        self.y_test = self.sc_y.transform(y_test.reshape(-1, 1))
        dim_x = len(X_train[0])

        # 将训练数据转换为Tensor并迁移到GPU（如果有GPU可用）
        self.X_train_tensor = torch.tensor(self.X_train, dtype=torch.float).to(self.device)
        self.y_train_tensor = torch.tensor(self.y_train, dtype=torch.float).to(self.device)
        self.X_test_tensor = torch.tensor(self.X_test, dtype=torch.float).to(self.device)
        self.y_test_tensor = torch.tensor(self.y_test, dtype=torch.float).to(self.device)

        best_lr = None
        best_score = float('inf')  # 初始化为正无穷大，因为要找最小的损失

        # 遍历内部定义的学习率范围，寻找最佳学习率
        for lr in self.lr_range:
            # 初始化神经网络
            self.net = NeuralNetRegressor(
                NNRegressor(dim_x=dim_x),
                lr=lr,
                max_epochs=800,
                iterator_train__shuffle=True,
                warm_start=True,
                verbose=False,
                device=self.device  # 设置设备
            )

            # 训练并评估模型
            self.net.fit(self.X_train_tensor, self.y_train_tensor)
            test_preds = self.net.predict(self.X_test_tensor)
            test_loss = ((test_preds - self.y_test_tensor.cpu().numpy()) ** 2).mean()  # 计算测试集均方误差

            # 更新最佳学习率
            if test_loss < best_score:
                best_score = test_loss
                best_lr = lr

        # 使用最佳学习率训练最终模型
        self.net = NeuralNetRegressor(
            NNRegressor(dim_x=dim_x),
            lr=best_lr,
            max_epochs=800,
            iterator_train__shuffle=True,
            warm_start=True,
            verbose=False,
            device=self.device  # 设置设备
        )

    def predict(self, x_test):
        # 将测试数据转换为Tensor并迁移到GPU（如果有GPU可用）
        x_test = self.sc_x.transform(x_test)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float).to(self.device)

        preds = []

        for i in range(self.n_models):
            # 设置每个模型的随机种子，保证每次训练时的随机性固定
            random_seed = self.random_seed_list[i]
            torch.manual_seed(random_seed)  # 固定 PyTorch 随机种子
            self.net.initialize()  # 重新初始化神经网络
            self.net.fit(self.X_train_tensor, self.y_train_tensor)  # 训练模型
            pred = self.net.predict(x_test_tensor)
            preds.append(self.sc_y.inverse_transform(pred))

        # 计算均值和标准差
        preds = np.hstack(preds)
        std = preds.std(axis=1)
        mean = preds.mean(axis=1)

        if self.correction:
            std = (self.n_models / (self.n_models - 1) * std ** 2) ** 0.5

        return mean, std
