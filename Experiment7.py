from GetScore7 import *
from IG import *
from sklearn.model_selection import train_test_split
import copy
from sklearn import tree
from sklearn import naive_bayes



class Experiment7():
    '''DCS-OVO'''

    def __init__(self, X_train, X_test, y_train, y_test,  sortNum,model_example):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.sortNum = sortNum
        self.model_example = model_example
        self.result = 0
        self.kappa = np.zeros(shape=[sortNum, sortNum])
        self.kappaResult = 0
        self.iter()

    def getResult(self):
        return self.result,self.kappaResult

    def getKappa(self):
        Pe=0
        Pa=0
        for i in range(self.sortNum):
            Pe+=(self.kappa.sum(axis=0)[i]*self.kappa.sum(axis=1)[i])
            Pa+=self.kappa[i][i]
        Pe=Pe/(self.kappa.sum()**2)
        Pa=Pa/self.kappa.sum()
        self.kappaResult =(Pa-Pe)/(1-Pe)

    def iter(self):

        '''训练'''
        X_dict = {}  # 存放每个label的样本
        y_dict = {}  # 存放每个label的样本
        for sort in range(self.sortNum):
            X_dict[sort] = []
            y_dict[sort] = []
            for num in range(len(self.y_train)):
                if (self.y_train[num] == sort):
                    X_dict[sort].append(self.X_train[num])
                    y_dict[sort].append(self.y_train[num])
        temp_index = 0
        model = {}
        for i in range(self.sortNum):
            for j in range(self.sortNum):
                if (j > i):
                    model[temp_index] = {}
                    X_new = X_dict[i] + X_dict[j]
                    y_new = y_dict[i] + y_dict[j]
                    X_new = np.array(X_new)
                    temp_model = copy.deepcopy(self.model_example)
                    temp_model.fit(X_new, y_new)
                    model[temp_index] = copy.deepcopy(temp_model)
                    temp_index+=1

        '''测试'''
        pred_prec = 0
        for test_sort in range(len(self.y_test)):
            result = GetScore7(self.X_test[test_sort], model, self.sortNum,self.X_train,self.y_train).getResult()  # 对每条测试集进行测试
            # print('y', test_sort, '的预测结束了，预测结果为', result, ',实际结果为', self.y_test[test_sort])
            if (result == self.y_test[test_sort]):
                pred_prec += 1
            self.kappa[result][self.y_test[test_sort]] += 1
        self.getKappa()
        self.result = float(pred_prec) / len(self.y_test)
