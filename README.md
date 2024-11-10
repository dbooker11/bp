# bp
Numpy implementation of BP neural network
# BP神经网络分类器
# 简介
 这个Python脚本实现了一个基于反向传播算法的神经网络分类器。该网络使用Sigmoid激活函数和交叉熵损失函数进行训练，并使用L2正则化来防止过拟合。该网络可以用于二分类问题。
# 依赖
numpy
matplotlib
scikit-learn
# 前置工作
导入必要的库：
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
```
定义相关函数
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred)) / m
```
定义BP神经网络类：
```python
class BPNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, reg_lambda, learning_rate=0.1):
    def forward(self, X):
    def backward(self, X, y, output):
    def train(self, X, y, epochs=10000):
    def predict(self, X):
```
生成数据集并分割：
```python
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
y = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
创建并训练神经网络：
```python
nn = BPNeuralNetwork(input_size=2, hidden_size=10, output_size=1, learning_rate=0.1, reg_lambda=0)
nn.train(X_train, y_train, epochs=10000)
绘制决策边界：
```python
def plot_decision_boundary(model, X, y):
    #绘图代码
plot_decision_boundary(nn, X_test, y_test)
```
# 注意事项
确保安装了所有依赖库。
可以调整网络参数，如隐藏层大小、学习率和正则化参数，以获得更好的性能。
该网络适用于二分类问题，对于多分类问题需要进行修改。
