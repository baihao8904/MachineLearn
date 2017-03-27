# -*- coding:utf-8 -*-
__author__ = 'lenovo'

import numpy as np
import LinearRegression.regression as regression
import matplotlib.pyplot as plt

def lwlr(testpoint,xArr,yArr,k=1.0):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testpoint - xMat[j,:]
        weights[j,j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print ('erroe')
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testpoint * ws

def lwlrtest(testArr,xArr,yArr,k=1.0):
    m= np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i]= lwlr(testArr[i],xArr,yArr,k)
    return yHat

if __name__ == '__main__':
    xArr,yArr = regression.loadDataSet('./ex0.txt')
    print(yArr[0])
    print(lwlr(xArr[0],xArr,yArr,1.0))
    print(lwlr(xArr[0],xArr,yArr,0.001))

    yHat = lwlrtest(xArr,xArr,yArr,0.01)

    xMat = np.mat(xArr)
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0],np.mat(yArr).T.flatten().A[0],s=2,c='red')
    plt.show()