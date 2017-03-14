# -*- coding：utf-8 -*-
import NN.threeNN as NN
import pickle

def createDataset(path='./dataset.txt'):
    dataset = open(path,'r')
    datas = []
    labels = []
    for line in dataset.readlines():
        lineData = []
        lineData.append(float(line.split(' ')[0]))
        lineData.append(float(line.split(' ')[1]))
        labels.append([int(line.split(' ')[-1].strip())])
        datas.append(lineData)
    return datas,labels
    
if __name__ == "__main__":
    traindata,trainlabel = createDataset()
    testdata,testlabel = createDataset('./test.txt')
    thrBPTree = NN.BPNeuralNetwork()
    thrBPTree.initNN(2,3,3,1)
    thrBPTree.train(traindata,trainlabel,50000,0.05,0.1)
    with open("NNmodel.pkl",'wb') as f_:
        pickle.dump(thrBPTree,f_)
    print('模型已保存')
    for i in range(len(testdata)):
        print('预测值：',thrBPTree.predict(testdata[i]),"实际值：",testlabel[i])
        