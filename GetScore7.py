import numpy as np
from sklearn import neighbors


class GetScore7:
    '''生成得分矩阵并返回正确率'''
    def __init__(self, X, model, n,X_train,y_train):
        self.X = X
        self.model = model
        self.result = 0
        self.n = n
        self.k = 5
        self.score = np.zeros(shape=[n, n])
        self.X_train = X_train
        self.y_train = y_train
        self.existSort=[]
        self.iter()

    def getResult(self):
        return self.result

    def getScore(self, X):
        temp = 0
        for i in range(self.n):
            for j in range(self.n):
                if (j > i):
                    self.score[i][j] = self.model[temp].predict_proba([X])[0][0]
                    self.score[j][i] = 1 - self.score[i][j]
                    temp += 1

    def delScore(self,X):
        model=neighbors.KNeighborsClassifier(n_neighbors=3*self.n , p=2)
        model.fit(self.X_train,self.y_train)
        kIndex = model.kneighbors([X], return_distance=False)[0]
        y_sort=[]
        for a in kIndex:
            y_sort.append(self.y_train[a])
        # y_sort=set(y_sort)
        if len(set(y_sort))==1:
            model = neighbors.KNeighborsClassifier(n_neighbors=2*3 * self.n, p=2)
            model.fit(self.X_train, self.y_train)
            kIndex = model.kneighbors([X], return_distance=False)[0]
            y_sort = []
            for b in kIndex:
                y_sort.append(self.y_train[b])
            if len(set(y_sort)) == 1:
                return
        zero=np.zeros(shape=[1, self.n])
        for i in range(self.n):
            if i not in y_sort:
                self.score[[i],:]=zero
                self.score[:,[i]]=zero.T

    def iter(self):
        self.getScore(self.X)
        # print(self.score)
        self.delScore(self.X)
        # print(self.score)
        scoreSum = np.sum(self.score, axis=1)
        self.result = np.argmax(scoreSum)

