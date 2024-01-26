import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset (or any other dataset you want to use)
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the data
X_standardized = StandardScaler().fit_transform(X)

# Apply PCA to reduce the data to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_standardized)

# Plot the results
plt.figure(figsize=(8, 6))

for i in range(len(iris.target_names)):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=iris.target_names[i])

plt.title('PCA: Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
