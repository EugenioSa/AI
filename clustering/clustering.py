import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import random
import pandas as pd



colors = 10*["g","r","c","b","k"]

class KMeans:
    def __init__(self, k=3, tolerance=.0001, maxIters = 1000):
        self.k = k
        self.tolerance = tolerance
        self.maxIters = maxIters

    def fit(self, data):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.maxIters):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for feat in data:
                dist = [np.linalg.norm(feat - self.centroids[centroid]) for centroid in self.centroids]
                classification = dist.index(min(dist))
                self.classifications[classification].append(feat)
            
            prevCentroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
            
            optim = True

            for i in self.centroids:
                firstCent = prevCentroids[i]
                currCent = self.centroids[i]
                if (np.sum((currCent - firstCent) / firstCent * 100.0) > self.tolerance):
                    print(np.sum((currCent - firstCent) / firstCent * 100.0))
                    optim = False
            if optim:
                break
    def predict(self, data):
        dist = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = dist.index(min(dist))
        return classification


df=pd.read_csv('Iris.csv', sep=',')
print(df)
aux = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].to_numpy()
np.random.shuffle(aux)
X = aux[0:100]
X_test = aux[100:150]
print(X)
print(X_test)

clustering = KMeans()
clustering.fit(X)

for centroid in clustering.centroids:
    plt.scatter(clustering.centroids[centroid][0], clustering.centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)

for classification in clustering.classifications:
    color = colors[classification]
    for featureset in clustering.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)


for unknown in X_test:
    classification = clustering.predict(unknown)
    print(classification)
    plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=150, linewidths=5)

plt.show()
