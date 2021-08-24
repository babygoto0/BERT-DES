import itertools
import random
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from Experiment6 import *
from Experiment7 import *

class DES_DRCW_OVOBPAP():
    def __init__(self,hidden_layer_sizes=(6,10),activation='relu',solver='sgd',alpha=0.01):
        self.k = 7
        self.fenceng = 1.0
        self.num = 40
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver =solver
        self.alpha = alpha
        # self.learning_rate = learning_rate
        # self.learning_rate_init = learning_rate_init
        self.totalClassCountDict = {}
        self.dataDictByClass = {}

    def fit(self,X,Y):
        self.X_train = np.copy(X)
        self.Y_train = np.copy(Y)
        for label in Y:
            if label not in self.totalClassCountDict:
                self.totalClassCountDict[label] = 1
            else:
                self.totalClassCountDict[label] += 1
        for label in self.totalClassCountDict:
            self.dataDictByClass[label] = []
        for i in range(len(Y)):
            self.dataDictByClass[Y[i]].append(X[i])
        self.cla = sorted(self.totalClassCountDict.keys())
        self.twoLabelEnsembel = list(itertools.combinations(self.cla, 2))
        self.classifier = []
        for i in range(len(self.twoLabelEnsembel)):
            traindata = np.vstack((self.dataDictByClass[self.twoLabelEnsembel[i][0]], self.dataDictByClass[self.twoLabelEnsembel[i][1]]))
            labelY = np.append(np.array([self.twoLabelEnsembel[i][0]] * self.totalClassCountDict[self.twoLabelEnsembel[i][0]]),
                               np.array([self.twoLabelEnsembel[i][1]] * self.totalClassCountDict[self.twoLabelEnsembel[i][1]]))
            models = self.trainmodel(traindata,labelY)
            self.classifier.append(models)


    def predict(self,predata):
        y = []
        for x in predata:
            matrix = np.zeros((len(self.cla), len(self.cla)))
            wmatrix = np.zeros((len(self.cla), len(self.cla)))
            for i in range(len(self.twoLabelEnsembel)):
                traindata = np.vstack((self.dataDictByClass[self.twoLabelEnsembel[i][0]],
                                       self.dataDictByClass[self.twoLabelEnsembel[i][1]]))
                labelY = np.append(
                    np.array([self.twoLabelEnsembel[i][0]] * self.totalClassCountDict[self.twoLabelEnsembel[i][0]]),
                    np.array([self.twoLabelEnsembel[i][1]] * self.totalClassCountDict[self.twoLabelEnsembel[i][1]]))
                matrix[self.twoLabelEnsembel[i][0]][self.twoLabelEnsembel[i][1]] = self.selectclassifierAndPredict(traindata, labelY, self.classifier[i], [x], self.twoLabelEnsembel[i][0])
                matrix[self.twoLabelEnsembel[i][1]][self.twoLabelEnsembel[i][0]] = 1 - matrix[self.twoLabelEnsembel[i][0]][self.twoLabelEnsembel[i][1]]

                wmatrix[self.twoLabelEnsembel[i][0]][self.twoLabelEnsembel[i][1]] = self.getW(
                    np.array(self.dataDictByClass[self.twoLabelEnsembel[i][0]]),
                    np.array(self.dataDictByClass[self.twoLabelEnsembel[i][1]]), [x])
                wmatrix[self.twoLabelEnsembel[i][1]][self.twoLabelEnsembel[i][0]] = self.getW(
                    np.array(self.dataDictByClass[self.twoLabelEnsembel[i][1]]),
                    np.array(self.dataDictByClass[self.twoLabelEnsembel[i][0]]), [x])

            maxindex = np.argmax(np.sum(matrix*wmatrix, axis=1))
            y.append(maxindex)

        y = np.array(y)
        return y

    def getdistance(self,x_train,testdata):
        dis = np.zeros(x_train.shape[0])
        for i in range(x_train.shape[0]):
            dist = np.linalg.norm(x_train[i] - testdata)
            dis[i] = dist
        dis.sort()
        distance = 0
        for i in range(5):
            distance = distance + dis[i]
        average = distance / 5
        return average

    def getW(self,x_train,x_trainother,testdata):
        d_train = self.getdistance(x_train,testdata)
        d_trainother = self.getdistance(x_trainother,testdata)

        W = (d_trainother*d_trainother)/(d_train*d_train + d_trainother*d_trainother)
        return W


    def trainmodel(self,x_train,y_train):
        classifi = []
        for i in range(self.num):
            train_x = []
            train_y = []
            #   有放回随机抽样
            for j in range(int(x_train.shape[0]* self.fenceng )):
                n = random.randint(0,x_train.shape[0]-1)
                train_x.append(x_train[n])
                train_y.append(y_train[n])
            train_x = np.array(train_x)
            train_y = np.array(train_y)
            # model = DecisionTreeClassifier(criterion='gini',min_samples_leaf=10,max_depth=3)
            model = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes,activation=self.activation,solver=self.solver,alpha=self.alpha,max_iter=2000)
            model.fit(train_x, train_y)
            classifi.append(model)
        return classifi

    def selectclassifierAndPredict(self, data, label, classifier, testdata, classNum):
        dis = []
        for i in range(data.shape[0]):
            distance = np.linalg.norm(data[i] - testdata[0])
            dis.append(distance)
        dis = np.array(dis)
        disxiabiao = dis.argsort()
        label1 = []
        data1 = []
        for m in range(self.k):
            label1.append(label[disxiabiao[m]])
            data1.append(data[disxiabiao[m]])
        label1 = np.array(label1)
        data1 = np.array(data1)
        selectedcla = []
        for i in range(self.num):
            model = classifier[i]
            acc = model.score(data1, label1)
            if acc > 0.5:
                selectedcla.append(model)

        if len(selectedcla) == 0:
            accuracy = 0.0
            for k in range(self.num):
                mode = classifier[k]
                ac = mode.predict_proba(testdata)[0][0]
                accuracy += ac
            result = accuracy / self.num
            return result
        else:
            accuracy1 = 0.0
            for n in range(len(selectedcla)):
                modell = selectedcla[n]
                ac1 = modell.predict_proba(testdata)[0][0]
                accuracy1 += ac1
            result = accuracy1/len(selectedcla)
            return result

    def get_params(self, deep=False):
        params = {'hidden_layer_sizes': self.hidden_layer_sizes, 'activation':self.activation, 'solver': self.solver, 'alpha':self.alpha}
        return params

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# data = np.loadtxt('1027bert_max_seq_len_512')
# label = np.loadtxt('D:\\fqjpypro\\scaledata\\Dennis+Schwartz\\label.3class.Dennis+Schwartz',delimiter='\n',dtype=int)
# # label = np.loadtxt('D:\\fqjpypro\\scaledata\\Dennis+Schwartz\\label.4class.Dennis+Schwartz',delimiter='\n',dtype=int)
#
# print('1027 3')

# data = np.loadtxt('1307bert_max_seq_len_512')
# label = np.loadtxt('D:\\fqjpypro\\scaledata\\James+Berardinelli\\label.3class.James+Berardinelli',delimiter='\n',dtype=int)
# # label = np.loadtxt('D:\\fqjpypro\\scaledata\\James+Berardinelli\\label.4class.James+Berardinelli',delimiter='\n',dtype=int)
#
# print('1307 3')
#
data = np.loadtxt('902bert_max_seq_len_512')
label = np.loadtxt('D:\\fqjpypro\\scaledata\\Scott+Renshaw\\label.3class.Scott+Renshaw',delimiter='\n',dtype=int)
# label = np.loadtxt('D:\\fqjpypro\\scaledata\\Scott+Renshaw\\label.4class.Scott+Renshaw',delimiter='\n',dtype=int)

print('902 3')
#
# data = np.loadtxt('1770bert_max_seq_len_512')
# label = np.loadtxt('D:\\fqjpypro\\scaledata\\Steve+Rhodes\\label.3class.Steve+Rhodes',delimiter='\n',dtype=int)
# # label = np.loadtxt('D:\\fqjpypro\\scaledata\\Steve+Rhodes\\label.4class.Steve+Rhodes',delimiter='\n',dtype=int)
# print('1770 3')




model_example7 =  MLPClassifier(hidden_layer_sizes=(6,10),activation='relu',solver='sgd',alpha=0.01,max_iter=2000)
model_example6 = {}
for i in range(40):
    model_example6[i] = MLPClassifier(hidden_layer_sizes=(6,10),activation='relu',solver='sgd',alpha=0.01,max_iter=2000)

from sklearn.model_selection import StratifiedKFold,train_test_split

scores1 = []
kappas1 = []

scores2 = []
kappas2 = []

scores3 = []
kappas3 = []

for i in range(3):
    score1 = []
    kappa1 = []

    score2 = []
    kappa2 = []

    score3 = []
    kappa3 = []
    kf = StratifiedKFold(n_splits=10,shuffle=True)
    # for train,test in kf.split(data_tfidf,label):
    for train,test in kf.split(data,label):
        X_train,X_test = data[train],data[test]
        y_train,y_test = label[train],label[test]

        model1 = DES_DRCW_OVOBPAP()
        model1.fit(X_train,y_train)
        pre1 = model1.predict(X_test)
        acc1= accuracy_score(pre1,y_test)
        score1.append(acc1)
        ka1 = cohen_kappa_score(pre1,y_test)
        kappa1.append(ka1)

        acc2, ka2 = Experiment7(X_train, X_test, y_train, y_test, 3,
                             model_example7).getResult()

        acc3, ka3 = Experiment6(X_train, X_test, y_train, y_test, 3,
                             model_example6).getResult()

        score2.append(acc2)
        kappa2.append(ka2)

        score3.append(acc3)
        kappa3.append(ka3)
        print('-----')
        print(acc1)
        print(ka1)
        print(acc2)
        print(ka2)
        print(acc3)
        print(ka3)
        print('-----')

    scores1.append(np.mean(score1))
    kappas1.append(np.mean(kappa1))
    scores2.append(np.mean(score2))
    kappas2.append(np.mean(kappa2))
    scores3.append(np.mean(score3))
    kappas3.append(np.mean(kappa3))
    print('--------------------')
    print(np.mean(score1))
    print(np.mean(kappa1))
    print(np.mean(score2))
    print(np.mean(kappa2))
    print(np.mean(score3))
    print(np.mean(kappa3))
    print('--------------------')


print("111" + str(np.mean(scores1)))
print("111"+str(np.mean(kappas1)))
print("222" + str(np.mean(scores2)))
print("222"+str(np.mean(kappas2)))
print("333" + str(np.mean(scores3)))
print("333"+str(np.mean(kappas3)))


