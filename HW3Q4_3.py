import numpy as np
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Define the lift function that expands the input vector into a higher-dimensional space
def lift(X):
    d = X.shape[1]
    # Create an empty list to store the new features
    new_features = []
    # Compute all possible combinations of features including squares and interactions
    for i in range(d):
        for j in range(i, d):
            new_features.append(X[:, i] * X[:, j])
    # Concatenate the new features with the original features
    X_lifted = np.hstack((X, np.column_stack(new_features)))
    return X_lifted

# Load the dataset
# Assuming the dataset is in the current directory; adjust the path as needed.
X = np.load(r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\HW3\X_N_1000_d_40_sig_0_01.npy')
y = np.load(r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\HW3\y_N_1000_d_40_sig_0_01.npy')

# Lift the dataset before splitting
X_lifted = lift(X)

# Split the lifted dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_lifted, y, test_size=0.30, random_state=42)

# Prepare KFold cross-validation
cv = KFold(n_splits=5, random_state=42, shuffle=True)

# Prepare a range of Lasso regularization parameters
alphas = np.logspace(-10, 10, 21)
mean_rmse = []
std_rmse = []

# Perform cross-validation for each alpha
for alpha in alphas:
    model = Lasso(alpha=alpha, max_iter=10000)
    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
    mean_rmse.append(-np.mean(scores))
    std_rmse.append(np.std(scores))

# Find the optimal alpha
optimal_alpha = alphas[np.argmin(mean_rmse)]

# Retrain a Lasso model with the optimal alpha on the entire training set
model_optimal = Lasso(alpha=optimal_alpha, max_iter=10000)
model_optimal.fit(X_train, y_train)

# Compute the train and test RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, model_optimal.predict(X_train)))
test_rmse = np.sqrt(mean_squared_error(y_test, model_optimal.predict(X_test)))

# Report the train and test RMSE for the model with the optimal alpha
print(f"Train RMSE with optimal alpha {optimal_alpha}: {train_rmse}")
print(f"Test RMSE with optimal alpha {optimal_alpha}: {test_rmse}")

# Report the parameters of the final model that have an absolute value larger than 1e-3
params = np.hstack((model_optimal.intercept_, model_optimal.coef_))
significant_params = params[np.abs(params) > 1e-3]
print("Parameters with absolute value larger than 1e-3:", significant_params)

# Plot the cross-validation mean RMSE as a function of alpha
plt.errorbar(alphas, mean_rmse, yerr=std_rmse, fmt='-o', capsize=5)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Mean RMSE')
plt.title('Cross-validation Mean RMSE vs Alpha')
plt.grid(True)
plt.show()
