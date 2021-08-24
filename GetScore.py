import numpy as np
from sklearn.cluster import KMeans
from sklearn import neighbors


class GetScore:
    '''获得得分矩阵'''

    def __init__(self, X, model, n, X_dict, y_dict):
        self.X = X
        self.model = model
        self.result = 0
        self.n = n
        self.score = np.zeros(shape=[n, n])
        self.X_dict = X_dict
        self.y_dict = y_dict
        self.d1 = {}
        self.d2 = {}
        self.iter()

    def getResult(self):
        return self.result

    def getW(self, i, j):
        w = self.d1[i] ** 2 / (self.d1[i] ** 2 + self.d1[j] ** 2) * self.d2[i] ** 2 / (
                    self.d2[i] ** 2 + self.d2[j] ** 2)
        return w

    def getScore(self, X):
        temp = 0
        for i in range(self.n):
            for j in range(self.n):
                if (j > i):
                    self.score[i][j] = self.model[temp].predict_proba([X])[0][0]
                    self.score[j][i] = 1 - self.score[i][j]
                    # self.score[i][j]=self.score[i][j]*self.getW(i,j)
                    # self.score[j][i]=self.score[j][i]*self.getW(j,i)
                    temp += 1

    def getCentreD(self):
        for i in range(self.n):
            kmeans_model = KMeans(n_clusters=1)
            kmeans_model.fit_transform(self.X_dict[i], self.y_dict[i])
            center = kmeans_model.cluster_centers_
            d = self.getDistance(self.X, center[0], i)
            self.d1[i] = d


    def getKD(self):
        for i in range(self.n):
            knn_model = neighbors.KNeighborsClassifier(p=2, n_neighbors=5)
            knn_model.fit(self.X_dict[i], self.y_dict[i])
            dis = knn_model.kneighbors([self.X])[0][0]
            d = 0
            for a in range(len(dis)):
                d += dis[a]
            self.d2[i] = d / 5

    def getDistance(self, x1, x2, i):
        dis = 0
        for i in range(len(x2)):
            dis += (x1[i] - x2[i]) ** 2
        return dis ** 0.5

    def iter(self):
        # self.getCentreD()
        # self.getKD()
        self.getScore(self.X)
        # print(self.score)
        scoreSum = np.sum(self.score, axis=1)
        self.result = np.argmax(scoreSum)
