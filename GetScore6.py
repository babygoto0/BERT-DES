import numpy as np
from sklearn.cluster import KMeans
from sklearn import neighbors


class GetScore6:
    '''获得得分矩阵'''

    def __init__(self, X, model, n,X_dict, y_dict,num):
        self.X = X
        self.model = model
        self.result = 0
        self.n = n
        self.score = np.zeros(shape=[n, n])
        self.X_dict = X_dict
        self.y_dict = y_dict
        self.result_dict={}
        self.d = {}
        self.num=num
        self.k1 = 5
        self.k2=9
        self.iter()


    def getResult(self):
        return self.result

    def getW(self, i, j):
        w = self.d[j] ** 2 / (self.d[i] ** 2 + self.d[j] ** 2+0.00000001)
        if w==0:
            w=1
        return w


    def getKD(self):
        for i in range(self.n):
            if len(self.y_dict[i])<self.k1:
                knn_model = neighbors.KNeighborsClassifier(p=2, n_neighbors=len(self.y_dict[i]))
            else:
                knn_model = neighbors.KNeighborsClassifier(p=2, n_neighbors=self.k1)
            knn_model.fit(self.X_dict[i], self.y_dict[i])
            dis = knn_model.kneighbors([self.X])[0][0]
            d = sum(dis)
            # for a in range(len(dis)):
            #     d += dis[a]
            if len(self.y_dict[i]) < self.k1:
                self.d[i] = d / len(self.y_dict[i])
            else:
                self.d[i] = d / self.k1

    def getScore(self, X):
        temp = 0
        for i in range(self.n):
            for j in range(self.n):
                if (j > i):
                    X_temp=[]
                    y_temp=[]
                    self.result_dict[i] = 0
                    self.result_dict[j] = 0
                    if len(self.y_dict[i])<self.k2:
                        model1 = neighbors.KNeighborsClassifier(n_neighbors=len(self.X_dict[i]), p=2)
                    else:
                        model1 = neighbors.KNeighborsClassifier(n_neighbors=self.k2,  p=2)
                    if len(self.y_dict[j])<self.k2:
                        model2 = neighbors.KNeighborsClassifier(n_neighbors=len(self.X_dict[j]),  p=2)
                    else:
                        model2 = neighbors.KNeighborsClassifier(n_neighbors=self.k2,  p=2)
                    model1.fit(self.X_dict[i],self.y_dict[i])
                    model2.fit(self.X_dict[j],self.y_dict[j])
                    kIndex1=model1.kneighbors([X], return_distance=False)[0]
                    kIndex2=model2.kneighbors([X], return_distance=False)[0]
                    for l in kIndex1:
                        X_temp.append(self.X_dict[i][l])
                        y_temp.append(self.y_dict[i][l])
                    for n in kIndex2:
                        X_temp.append(self.X_dict[j][n])
                        y_temp.append(self.y_dict[j][n])
                    X_temp = np.array(X_temp)
                    model_score=[]
                    for m in range(self.num):
                        model_score.append(self.model[temp][m].score(X_temp, y_temp))
                    result = self.model[temp][model_score.index(max(model_score))].predict(X=[X])
                    self.result_dict[result[0]] += 1
                    self.score[i][j] = self.result_dict[i]
                    self.score[j][i] = self.result_dict[j]
                    self.score[i][j] = self.score[i][j] * self.getW(i, j)
                    self.score[j][i] = self.score[j][i] * self.getW(j, i)
                    temp += 1

    def iter(self):
        self.getKD()
        self.getScore(self.X)
        scoreSum = np.sum(self.score, axis=1)
        self.result = np.argmax(scoreSum)