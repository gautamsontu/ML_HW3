import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define the lift function that expands the input vector into a higher-dimensional space
def lift(x):
    d = len(x)
    # Initialize the lifted vector with the original elements and interaction terms
    x_lifted = list(x)
    # Compute all possible coordinate combinations of the form xi * xj for i >= j
    for i in range(d):
        for j in range(i, d):
            x_lifted.append(x[i] * x[j])
    return np.array(x_lifted)

# Load the dataset (replace 'path_to_X.npy' and 'path_to_y.npy' with the actual file paths)
X = np.load(r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\HW3\X_N_1000_d_40_sig_0_01.npy')
y = np.load(r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\HW3\y_N_1000_d_40_sig_0_01.npy')

# Apply the lifting to each row of the original dataset X
X_lifted = np.apply_along_axis(lift, 1, X)

# Split the lifted dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_lifted, y, test_size=0.3, random_state=42)

# Initialize lists to store RMSE values
train_rmse = []
test_rmse = []
fraction_of_training_data = []

# Train models on increasing fractions of the training data
fractions = np.arange(0.1, 1.1, 0.1)
for fr in fractions:
    # Determine the number of samples for the current fraction
    num_samples = int(fr * X_train.shape[0])
    X_train_fr = X_train[:num_samples]
    y_train_fr = y_train[:num_samples]

    # Train the model
    model = LinearRegression()
    model.fit(X_train_fr, y_train_fr)

    # Predict on the training and test sets
    y_train_pred = model.predict(X_train_fr)
    y_test_pred = model.predict(X_test)

    # Calculate RMSE and store it
    train_rmse.append(np.sqrt(mean_squared_error(y_train_fr, y_train_pred)))
    test_rmse.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    fraction_of_training_data.append(fr * 100)

# Train the final model on the entire training set
final_model = LinearRegression()
final_model.fit(X_train, y_train)

# Extract the coefficients and intercept of the final model
final_coefficients = final_model.coef_
final_intercept = final_model.intercept_

# Output the parameters of the final trained model
print(f"Final Model Coefficients: {final_coefficients}")
print(f"Final Model Intercept: {final_intercept}")

# Plotting the RMSE values
plt.figure(figsize=(10, 6))
plt.plot(fraction_of_training_data, train_rmse, label='Training RMSE', marker='o')
plt.plot(fraction_of_training_data, test_rmse, label='Test RMSE', marker='s')
plt.xlabel('Percentage of Training Data Used')
plt.ylabel('RMSE')
plt.title('Training and Test RMSE vs. Training Data Used with Lift')
plt.legend()
plt.grid(True)
plt.show()
