# -*- coding:utf-8 -*-
__author__ = 'lenovo'

import numpy as np
import matplotlib.pyplot as plt


def loadDataSet():
    dataMat=[]
    labelMat=[]
    _file = open('testSet.txt')
    for line in _file.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat


def sigmod(inX):
    return 1.0/(1+np.exp(-inX))

def classifyVector(inX,weights):
    prob = sigmod(sum(inX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0

def gradAscent(dataMatIn,classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat  =np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights= np.ones((n,1))
    for K in range(maxCycles):
        h = sigmod(dataMatrix*weights)
        error = (labelMat-h)
        weights = weights + alpha*dataMatrix.T*error
    return weights

def GradAscentRandom(dataMatrix,classLabels):
    m,n = np.shape(dataMatrix)
    dataMatrix = np.array(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmod(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + (alpha * error) * dataMatrix[i]
    return weights

def plotBestFit(wei):
    #变成一个矩阵
    weights = wei
    dataMat,labelMat = loadDataSet()
    dataArr  = np.array(dataMat)
    print(dataArr)
    n = np.shape(dataArr)[0]
    xcord1 = [];ycord1=[]
    xcord2 = [];ycord2= []
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = np.arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingList = []
    trainingLabel = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for  i in range(21):
            lineArr.append(float(currLine[i]))
        trainingList.append(lineArr)
        trainingLabel.append(float(currLine[21]))
    trainWeights = GradAscentRandom(np.array(trainingList),trainingLabel)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec +=1.0
        currLine = line.strip().split('\t')
        if int(classifyVector(np.array(lineArr),trainWeights))!= int(currLine[-1]):
            errorCount+=1
    errorRate = float(errorCount)/numTestVec
    print('错误率：%f' %errorRate)
    return errorRate

def multiTest(numtests = 10):
    numTests = numtests
    errorSum = 0.0
    for K in range(numTests):
        errorSum += colicTest()
    errorRate = errorSum/float(numTests)
    print('%d 次迭代之后的错误率是%f'%(numTests,errorRate))

if __name__ == '__main__':
    # dataArr,kabelMat = loadDataSet()
    # weights = GradAscentRandom(dataArr,kabelMat)
    # plotBestFit(weights)
    multiTest(20)