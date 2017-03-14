import pickle

model = pickle.load(open('NNmodel.pkl','rb'))

def createDataset(path='./test.txt'):
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


datas,labels = createDataset()
print(datas)
for i in range(len(datas)):
    print(model.predict(datas[i]),labels[i])