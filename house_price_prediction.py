# House Price Prediction using Linear Regression

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
data = pd.read_csv("train.csv")

# Display first 5 rows
print("First 5 Rows of Dataset")
print(data.head())

# Select Features (Input)
X = data[["GrLivArea", "BedroomAbvGr", "FullBath"]]

# Select Target (Output)
y = data["SalePrice"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Linear Regression Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Predict House Prices
y_pred = model.predict(X_test)

# Compare Actual and Predicted Prices
comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nFirst 10 Predictions")
print(comparison.head(10))

# Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("------------------------")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)

# Display Model Coefficients
print("\nModel Coefficients")
print("------------------------")
print("Square Footage Coefficient:", model.coef_[0])
print("Bedrooms Coefficient:", model.coef_[1])
print("Bathrooms Coefficient:", model.coef_[2])
print("Intercept:", model.intercept_)

# Plot Actual vs Predicted Prices
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color="blue")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linewidth=2
)

plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.grid(True)

plt.show()

# Predict Price of a New House
new_house = pd.DataFrame({
    "GrLivArea": [2500],
    "BedroomAbvGr": [4],
    "FullBath": [3]
})

predicted_price = model.predict(new_house)

print("\nPrediction for New House")
print("------------------------")
print("Square Footage :", 2500)
print("Bedrooms       :", 4)
print("Bathrooms      :", 3)
print("Predicted House Price : ${:,.2f}".format(predicted_price[0]))