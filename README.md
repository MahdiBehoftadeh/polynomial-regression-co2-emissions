# Car CO2 Emission Prediction - by Mahdi Behoftadeh

This project uses polynomial regression to predict CO2 emissions from car features. The code includes data loading, polynomial feature generation, model training, evaluation, and a user input loop for real-time predictions.

### Prerequisites

Make sure you have the following Python libraries installed:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

You can install them using pip:

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Data

The dataset (`car_co2_emissions_data.csv`) in the same directory has the following columns:
- `ENGINE_SIZE`: Size of the car engine
- `CYLINDERS`: Number of cylinders in the car engine
- `CO2_EMISSIONS`: CO2 emissions of the car

### How It Works

1. **Load Data**: The script reads the dataset from `car_co2_emissions_data.csv`.
2. **Feature and Target Setup**: It selects `ENGINE_SIZE` and `CYLINDERS` as features and `CO2_EMISSIONS` as the target variable.
3. **Polynomial Features**: Generates polynomial features to capture interactions between the original features.
4. **Split Data**: Splits the data into training and testing sets.
5. **Train Model**: Trains a model to predict CO2 emissions.
6. **Evaluate Model**: Calculates and prints various performance metrics including Mean Squared Error, Root Mean Squared Error, Mean Absolute Error, and R-squared.
7. **Plot Data**: Creates a scatter plot for training and test data along with a prediction line. (Helps us understand the accuracy of the model better)
8. **User Input**: Continuously prompts the user for engine size and number of cylinders, then predicts and displays the CO2 emissions.

### Running the Script

Run the script using Python:

```bash
python polynomial_regression_co2_emissions.py
```

Follow the prompts to input engine size and the number of cylinders for real-time CO2 emission predictions.
