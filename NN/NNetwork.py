import numpy as np

def tanh(x):
    return np.tanh(x)
def tanh_deriv(x):
    return 1.0-np.tanh(x)*np.tanh(x)
def logistic(x):
    return 1(1+np.exp(-x))
def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))


class NeuralNetWork:
    def __init__(self, layers, activaion='tanh'):
        #layers 一个intlist int表示有多少个神经元 算输入层
        if activaion == 'logistic':
            self.activaion = logistic
            self.activaion_deriv = logistic_derivative
        elif activaion == 'tanh':
            self.activaion = tanh
            self.activaion_deriv = tanh_deriv

        self.weights = []
        for i in range(1, len(layers)-1):
            # 每一层与前一层的联系
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            # 当前层与下一层的联系
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)

    def fit(self, x, y, learning_rate=0.2, epochs=10000):
        #确定X至少是2维数组
        X = np.atleast_2d(x)
        #初始化矩阵。全是1.行数与X相同，列数比X多1，因为要对bias赋值
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        #将X的值付给除了最后一列的temp中
        temp[:, 0:-1] = X
        #给x多加了一列 bias初始全部付1
        X = temp
        #将y转换成array
        y = np.array(y)

        #每次循环从X中随机抽取一行对神经网络进行更新
        for k in range(epochs):
            i = np.random.randint((X.shape[0]))
            #将X的第i行提取出来,以它对神经网络进行跟新
            a = [X[i]]
            #对每一层进行正向更新
            for l in range(len(self.weights)):
                a.append(self.activaion(np.dot(a[l],self.weights[l])))
            #取最后一层的值进行计算
            error = y[i] - a[-1]
            deltas = [error*self.activaion_deriv(a[-1])]
            #开始向回计算
            # Staring backprobagation
            for l in range(len(a) - 2, 0, -1):  # we need to begin at the second to last layer
                # Compute the updated error (i,e, deltas) for each node going from top layer to input layer
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activaion_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activaion(np.dot(a, self.weights[l]))
        return a

