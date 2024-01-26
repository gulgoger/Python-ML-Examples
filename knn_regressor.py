# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Generate a sample dataset
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN regressor with k=3 (you can choose any value for k)
knn_regressor = KNeighborsRegressor(n_neighbors=3)

# Train the regressor on the training data
knn_regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn_regressor.predict(X_test)

# Evaluate the performance using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Plot the results
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='black', label='True values')
plt.scatter(X_test, y_pred, color='blue', label='Predicted values')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('KNN Regression Example')
plt.show()

