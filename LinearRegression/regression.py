# -*- coding:utf-8 -*-
__author__ = 'lenovo'

from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(filename):
    #求出前面的参数的列数
    numFeat = len(open(filename).readline().split('\t')) -1
    dataMat = []
    labelMat = []
    fr = open(filename)
    #迭代文件的每一行
    for line in fr.readlines():
        lineArr=[]
        curLine = line.strip().split('\t')
        #迭代参数 加入每行的列表 再将每行列表加入参数总列表
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        #加入目标列表
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0 :
        print('这个矩阵不能求逆')
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

if __name__ == '__main__':
    xArr,yArr = loadDataSet('./ex0.txt')
    print(xArr[0:2])
    ws = standRegres(xArr,yArr)
    print(ws)
    xMat = mat(xArr);yMat = mat(yArr)
    yHat = xMat *ws

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
    xCopy = xMat.__copy__()
    xCopy.sort(0)
    yHat = xCopy*ws
    ax.plot(xCopy[:,1],yHat)
    plt.show()