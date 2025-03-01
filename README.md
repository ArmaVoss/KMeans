# KMeans

## Goal
- To implement KMeans from scratch to better understand the algorithm

## Libraries
- Matplotlib
- Scipy (cdist)
- Numpy

## Implementation
- Implemented a optimized KMeans++ initialization that initializes clusters 10 times similar to how it's done in sklearn.
  - Used furthest first distance to create centroids
- Implemented KMeans, otherwise known as Lloyd's algorithm after centroids were initialized
- Ran KMeans on multiple K values to find optimal number of clusters
  - Achieved through visualization, Elbow Method, of the cost of each KMeans iteration on different K's
 
## Results
- Found that K = 3 was the optimal K value for this synthetic data set
- Found optimal labeling for this dataset
