import numpy as np
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def lift(X):
    d = X.shape[1]
    new_features = []
    for i in range(d):
        for j in range(i, d):
            new_features.append(X[:, i] * X[:, j])
    X_lifted = np.hstack((X, np.column_stack(new_features)))
    return X_lifted

X = np.load(r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\HW3\X_N_1000_d_40_sig_0_01.npy')
y = np.load(r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\HW3\y_N_1000_d_40_sig_0_01.npy')

X_lifted = lift(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_lifted, y, test_size=0.30, random_state=42)

cv = KFold(n_splits=5, random_state=42, shuffle=True)

alphas = np.logspace(-10, 10, 21)
mean_rmse = []
std_rmse = []

for alpha in alphas:
    model = Lasso(alpha=alpha, max_iter=10000)
    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
    mean_rmse.append(-np.mean(scores))
    std_rmse.append(np.std(scores))

optimal_alpha = alphas[np.argmin(mean_rmse)]

model_optimal = Lasso(alpha=optimal_alpha, max_iter=10000)
model_optimal.fit(X_train, y_train)

train_rmse = np.sqrt(mean_squared_error(y_train, model_optimal.predict(X_train)))
test_rmse = np.sqrt(mean_squared_error(y_test, model_optimal.predict(X_test)))

print(f"Train RMSE with optimal alpha {optimal_alpha}: {train_rmse}")
print(f"Test RMSE with optimal alpha {optimal_alpha}: {test_rmse}")

params = np.hstack((model_optimal.intercept_, model_optimal.coef_))
significant_params = params[np.abs(params) > 1e-3]
print("Parameters with absolute value larger than 1e-3:", significant_params)

plt.errorbar(alphas, mean_rmse, yerr=std_rmse, fmt='-o', capsize=5)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Mean RMSE')
plt.title('Cross-validation Mean RMSE vs Alpha')
plt.grid(True)
plt.show()
