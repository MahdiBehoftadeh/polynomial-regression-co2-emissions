import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

# Load the dataset
data = pd.read_csv('car_co2_emissions_data.csv')

# Example features and target
features = data[['ENGINE_SIZE', 'CYLINDERS']]
target = data['CO2_EMISSIONS']

# Generate polynomial features
polynomial_features = PolynomialFeatures(degree=2, include_bias=False)
features_poly = polynomial_features.fit_transform(features)

# Split the data
features_train, features_test, target_train, target_test = train_test_split(features_poly, target, test_size=0.2, random_state=random.randint(0, 42))

# Initialize and train the neural network model with epochs
neural_network_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=random.randint(0, 42))
neural_network_model.fit(features_train, target_train)

# Predictions for both training and test data
target_train_pred = neural_network_model.predict(features_train)
target_test_pred = neural_network_model.predict(features_test)

# Evaluate the model
train_mse = mean_squared_error(target_train, target_train_pred)
test_mse = mean_squared_error(target_test, target_test_pred)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_mae = mean_absolute_error(target_train, target_train_pred)
test_mae = mean_absolute_error(target_test, target_test_pred)
train_r2 = r2_score(target_train, target_train_pred)
test_r2 = r2_score(target_test, target_test_pred)

print(f"Mean Squared Error (Train): {train_mse:.2f}")
print(f"Mean Squared Error (Test): {test_mse:.2f}")
print(f"Root Mean Squared Error (Train): {train_rmse:.2f}")
print(f"Root Mean Squared Error (Test): {test_rmse:.2f}")
print(f"Mean Absolute Error (Train): {train_mae:.2f}")
print(f"Mean Absolute Error (Test): {test_mae:.2f}")
print(f"R-squared (Train): {train_r2:.2f}")
print(f"R-squared (Test): {test_r2:.2f}")

# Convert original features to a NumPy array for indexing
features_np = features.to_numpy()

# Scatter plot for training data
plt.scatter(features_np[:len(features_train), 0], target_train, color='blue', label='Train Data')
# Scatter plot for test data
plt.scatter(features_np[len(features_train):, 0], target_test, color='red', label='Test Data')

# Prediction line (for visualization, we need to sort the data)
engine_size_range = np.linspace(min(features_np[:, 0]), max(features_np[:, 0]), 100).reshape(-1, 1)
engine_size_range_poly = polynomial_features.transform(np.hstack([engine_size_range, np.full_like(engine_size_range, features_np[:, 1].mean())]))
emission_range_pred = neural_network_model.predict(engine_size_range_poly)

plt.plot(engine_size_range, emission_range_pred, color='green', label='Prediction Line')

# Labels and legend
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')
plt.title('Polynomial Regression - Training and Test Data')
plt.legend()

plt.show()

# Continuous input loop asking user for data needed
while True:
    input_engine_size = float(input("Enter your engine size: "))
    input_cylinders = float(input("Enter the number of cylinders: "))

    # Prepare the input data as a 2D array
    user_input_data = np.array([[input_engine_size, input_cylinders]])

    # Transform the input data into polynomial features
    user_input_data_poly = polynomial_features.transform(user_input_data)

    predicted_emission = neural_network_model.predict(user_input_data_poly)
    print(f"Your CO2 emission will be: {predicted_emission[0]:.2f} grams/km")