import math
import numpy as np
import time
#两个工具函数
#a,b是预设的互为相反数的参数。rand函数生成。从a到b的随机值
def rand(a,b):
    return (b-a)*np.random.random()+a
#创建一个指定大小的矩阵
def make_matrix(m,n,fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill]*n)
    return mat
#sigmod函数
def sigmod(x):
    return 1.0/(1.0+np.exp(-x))
def sigmod_derivate(x):
    return x*(1-x)

class BPNeuralNetwork:
    def __init__(self):
        self.input_n = 0
        self.hidden1_n = 0
        self.hidden2_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden1_cells = []
        self.hidden2_cells = []
        self.output_cells = []
        self.first_weights = []
        self.second_weights = []
        self.third_weights = []
        self.first_correction = []
        self.second_correction = []
        self.third_correction = []
    #初始化神经网络.三个参数分别为输入层神经元个数，隐藏层，和输出层

    def initNN(self,ni,nh1,nh2,no):
        #增加一个偏置神经元
        self.input_n = ni+1
        self.hidden1_n = nh1
        self.hidden2_n = nh2
        self.output_n = no
        #初始化神经元
        self.input_cells = [1.0]*self.input_n
        self.hidden1_cells = [1.0]*self.hidden1_n
        self.hidden2_cells = [1.0] * self.hidden2_n
        self.output_cells = [1.0]*self.output_n
        #初始化全职 weight
        self.first_weights = make_matrix(self.input_n,self.hidden1_n)
        self.second_weights = make_matrix(self.hidden1_n,self.hidden2_n)
        self.third_weights = make_matrix(self.hidden2_n, self.output_n)
        #随机设置weight的初值
        for i in range(self.input_n):
            for h1 in range(self.hidden1_n):
                self.first_weights[i][h1]=rand(-0.2,0.2)
        for h1 in range(self.hidden1_n):
            for h2 in range(self.hidden2_n):
                self.second_weights[h1][h2]=rand(-1.0,1.0)
        for h2 in range(self.hidden2_n):
            for o in range(self.output_n):
                self.third_weights[h2][o] = rand(-2.0,2.0)
        #初始化矫正矩阵
        self.first_correction = make_matrix(self.input_n,self.hidden1_n)
        self.second_correction = make_matrix(self.hidden1_n, self.hidden2_n)
        self.third_correction = make_matrix(self.hidden2_n,self.output_n)





    #进行一次前馈并返回输出
    def predict(self,inputs):
        #激活输入层，将训练样本的值赋给输入层
        for i in range(self.input_n-1):
            self.input_cells[i] = inputs[i]
        #激活隐藏1层
        for h1 in range(self.hidden1_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i]*self.first_weights[i][h1]
            self.hidden1_cells[h1] = sigmod(total)
        # 激活隐藏2层
        for h2 in range(self.hidden2_n):
            total = 0.0
            for h1 in range(self.hidden1_n):
                total += self.hidden1_cells[h1] * self.second_weights[h1][h2]
            self.hidden2_cells[h2] = sigmod(total)
        #激活输出层
        for o in range(self.output_n):
            total = 0.0
            for h2 in range(self.hidden2_n):
                total += self.hidden2_cells[h2]*self.third_weights[h2][o]
            self.output_cells[o] = sigmod(total)
        return  self.output_cells

    #定义反向传播和更新权值的过程，返回最终预测误差
    #训练样例，样例标签，学习率，矫正率
    def backprobagate(self,case,label,learn,correct):
        #前馈
        self.predict(case)
        #获得输出层的误差
        output_deltas = [0.0]*self.output_n
        for o in range(self.output_n):
            error = label[o] -self.output_cells[o]
            output_deltas[o] = sigmod_derivate(self.output_cells[o])*error
        #隐藏层的错误初始化，并更新
        hidden2_deltas = [0.0]*self.hidden2_n
        for h2 in range(self.hidden2_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.third_weights[h2][o]
            hidden2_deltas[h2] = sigmod_derivate(self.hidden2_cells[h2])*error

        hidden1_deltas = [0.0] * self.hidden1_n
        for h1 in range(self.hidden1_n):
            error = 0.0
            for h2 in range(self.hidden2_n):
                error += hidden2_deltas[h2] * self.second_weights[h1][h2]
            hidden1_deltas[h1] = sigmod_derivate(self.hidden1_cells[h1]) * error

        #更新反向第一层的权值
        for h2 in range(self.hidden2_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden2_cells[h2]
                self.third_weights[h2][o] += learn*change+correct*self.third_correction[h2][o]
                self.third_correction[h2][o] = change
        #更新反向第二层
        for h1 in range(self.hidden1_n):
            for h2 in range(self.hidden2_n):
                change = hidden2_deltas[h2]*self.hidden1_cells[h1]
                self.second_weights[h1][h2] += learn*change+correct*self.second_correction[h1][h2]
                self.second_correction[h1][h2] = change
        for i in range(self.input_n):
            for h1 in range(self.hidden1_n):
                change = hidden1_deltas[h1] * self.input_cells[i]
                self.first_weights[i][h1] += learn * change + correct * self.first_correction[i][h1]
                self.first_correction[i][h1] = change
        #获取全局的误差
        error = 0.0
        for o in range(len(label)):
            error += len(label)*(label[o]-self.output_cells[o])**2
        return error

    def train(self,cases,labels,limit=50000,learn=0.05,correct=0.1):
        starttraintime = time.time()
        for i in range(limit):
            error = 0.0
            for casenum in range(len(cases)):
                label = labels[casenum]
                case = cases[casenum]
                error += self.backprobagate(case,label,learn,correct)
            if i%100 ==0:
                print('迭代',i,'次',end=' ')
                print('当前用时：',time.time()-starttraintime)
            if i%500 ==0:
                print('当前第'+str(i)+'次误差是%f'%error)
    def test(self):
        cases = [
            [0,0],
            [0,1],
            [1,0],
            [1,1]
        ]
        labels = [[0],[1],[1],[0]]
        self.initNN(2,3,3,1)
        #迭代收敛相对较难 需要增加迭代次数
        self.train(cases,labels,100000,0.1,0.1)
        for case in cases:
            print(self.predict(case))
    
    def testData(self,train,label):
        self.initNN(5,7,3,1)
        self.train(train,label,2000,0.05,0.1)

if __name__ == '__main__':
    theBPNN = BPNeuralNetwork()
    theBPNN.test()