#By Arman Vossoughi

#IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as distance
from sklearn.datasets import make_blobs

#Goal of this mini-project
#To practice implementing K-Means from scratch, in order to better understand the algorithm

class KMeans:
    # Constructer to initialize how many centers we have
    def __init__(self, k: int):
        self.k = k
        self.data = None
        self.labels = None
        self.clusters_centers = None
        self.cost = None

    # Initializes the data and initial cluster centers for K-Means
    def fit(self, X: np.ndarray):
        self.data = X
        minCost = float('inf')

        #run initialization 10 times, until we get lowest cost function initialziation
        for i in range(10):
            firstCenter = np.random.randint(0, X.shape[0], size = 1)[0]
            clustering = np.array([X[firstCenter]])

            #we want to get distances from this point to all other one
            #generate k clusters
            for _ in range(self.k -1):
                cluster_distances = np.min(distance.cdist(X, clustering), axis=1)            #calculate the points from the cluster centers
                probabilites = cluster_distances / np.sum(cluster_distances)                 #assign probablities porportional to d(x)^2 
                num = np.random.choice(X.shape[0], size = 1, p = probabilites)               #select the index for the new centroid
                clustering = np.vstack([clustering, X[num]])                                 #update the centroid list

            #if cost of this initialization is lower than current assignment, update the old one
            cost = np.sum(np.min(distance.cdist(self.data, clustering), axis= 1))
            if cost < minCost:
                self.clusters_centers = clustering
                minCost = cost

        return 
    
    #Predict method does clustering assignments 
    def predict(self) -> np.ndarray:
        while True:
            centroids = []
            distances = distance.cdist(self.data, self.clusters_centers)
            self.labels = np.argmin(distances, axis=1)
            
            #get new centroids
            for label in range(self.k):
                cluster = self.data[self.labels == label]
                newCentroid = np.mean(cluster, axis = 0)
                centroids.append(newCentroid)

            #update old centroids to new ones
            centroids = np.array(centroids)
            oldCentroids = self.clusters_centers
            self.clusters_centers = centroids

            #checks for convergence
            if np.linalg.norm(oldCentroids - self.clusters_centers) < 1e-4:
                break
        
        #compute the final cost of this and store it
        self.cost = np.sum(np.min(distance.cdist(self.data, self.clusters_centers), axis= 1))
        return self.labels

    #fits and predicts in one functon similar to how sklearn has fit_predict and fit transform
    def fit_predict(self, X: np.array) -> np.ndarray:    
        self.fit(X)
        self.predict()
        return self.labels

#Plot the result from the KMeans algorithm after it converges
def plot_clustering(Kmean: KMeans):
    plt.scatter(Kmean.data[:, 0], Kmean.data[:, 1], c=Kmean.labels, cmap='viridis', alpha=0.6)
    plt.scatter(Kmean.clusters_centers[:, 0], Kmean.clusters_centers[:, 1], color='red', marker='x', s=100, label="Centroids")
    plt.title("K-Means Clustering")
    plt.legend()
    plt.show() 

#Plot the elbow method
def elbow_method_visualization(X: np.ndarray):
    kValues = []
    costs = []
    for i in range(1, 10):
        kValues.append(i)
        Kmean = KMeans(i)
        Kmean.fit_predict(X)
        costs.append(Kmean.cost)

    plt.plot(kValues, costs)
    plt.ylabel("Cost")
    plt.xlabel("K")
    plt.title("Elbow Method")
    plt.show()
#Main method
def main():
    X, _ = make_blobs(n_samples = 400, centers = 3, n_features=2, random_state= 42)

    #run kmeans on multiple K's to find best K 
    elbow_method_visualization(X)

    #By elbow method we see 3 centroids is the optimal
    Kmean = KMeans(3)
    Kmean.fit_predict(X)
    plot_clustering(Kmean)

if __name__ == "__main__":
    main()

  
