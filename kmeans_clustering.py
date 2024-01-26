# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate some random data for demonstration
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# Create a KMeans instance with the desired number of clusters (k)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)

# Fit the KMeans model to the data
kmeans.fit(X)

# Get the cluster centers and labels
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Plot the original data points and the cluster centers
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors='k', marker='o', label='Data points')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.legend()
plt.show()
