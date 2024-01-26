# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.datasets import make_blobs

# Generate some random data for demonstration
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# Perform hierarchical clustering using linkage function
# The linkage function computes the distance between clusters
linked = linkage(X, method='ward')

# Plot the dendrogram
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
