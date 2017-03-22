#encoding:utf-8
import numpy as np
import operator


#读已标记数据
def createDataSet():
    dataset = open('./dataset.txt')
    labels = []
    datas = []
    for line in dataset.readlines():
        lineData = []
        lineData.append(line.split(" ")[0])
        lineData.append(line.split(" ")[1])
        labels.append(line.split(" ")[-1].strip())
        datas.append(lineData)
    group = np.array(datas)
    return group,labels

def KNN(inX,dataSet,labels,k):
    #返回“数组”的行数，如果shape[1]返回的则是数组的列数
    dataSetSize = dataSet.shape[0]
    #两个“数组”相减，得到新的数组
    diffMat = np.tile(inX,(dataSetSize,1))- dataSet
    #求平方
    sqDiffMat = diffMat **2
    #求和，返回的是一维数组
    sqDistances = sqDiffMat.sum(axis=1)
    #开方，即测试点到其余各个点的距离
    distances = sqDistances **0.5
    #排序，返回值是原数组从小到大排序的下标值.前几个即为距离最小的点
    sortedDistIndicies = np.argsort(distances)
    classCount = {}
    for i in range(k):
        #返回距离最近的k个点所对应的标签值
        voteIlabel = labels[sortedDistIndicies[i]]
        #存放到字典中 每多一个相同分类 加1
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    #排序 classCount.items() 输出键值对 key代表排序的关键字 True代表降序
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    #返回距离最小的点对应的标签
    for item in sortedClassCount:
        print("分类标签:",item[0],'投票个数：',item[1])
    return sortedClassCount[0][0]

if __name__ == '__main__':
    dataset,labels = createDataSet()
    dataset = dataset.astype("float64")
    print(KNN([8,5],dataset,labels,3))
