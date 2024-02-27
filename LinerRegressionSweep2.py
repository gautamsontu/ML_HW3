import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# Attempting to load the dataset again to verify it's loaded correctly
try:
    # Load the dataset
    X = np.load(r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\HW3\X_N_1000_d_40_sig_0_01.npy')
    y = np.load(r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\HW3\y_N_1000_d_40_sig_0_01.npy')

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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

        # Indicating success to proceed with plotting
    success = True
except Exception as e:
    success = False
    error_message = str(e)

success, error_message if not success else print("No errors, ready to plot.")

# Plotting the RMSE values
plt.figure(figsize=(10, 6))
plt.plot(fraction_of_training_data, train_rmse, label='Training RMSE')
plt.plot(fraction_of_training_data, test_rmse, label='Test RMSE')
plt.xlabel('Percentage of Training Data Used')
plt.ylabel('RMSE')
plt.title('Training and Test RMSE vs. Training Data Used')
plt.legend()
plt.grid(True)
plt.show()