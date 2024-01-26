# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Regressor model
tree_reg_model = DecisionTreeRegressor(max_depth=3)  # You can adjust the depth as needed

# Train the model on the training data
tree_reg_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = tree_reg_model.predict(X_test)

# Evaluate the performance using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Visualize the results
X_range = np.linspace(0, 2, 100).reshape(-1, 1)
y_range_pred = tree_reg_model.predict(X_range)

plt.scatter(X_test, y_test, color='black', label='True values')
plt.plot(X_range, y_range_pred, color='blue', linewidth=3, label='Predicted values')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('Decision Tree Regression Example')
plt.show()
