import numpy as np
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from data_generator import postfix

def lift(X):
    # Lifts the dataset by including interaction terms (products of features)
    X_lifted = np.hstack([X] + [np.prod(X[:, i:], axis=1).reshape(-1, 1) for i in range(X.shape[1])])
    return X_lifted

N = 1000
sigma = 0.01
d = 40

psfx = postfix(N, d, sigma)

X = np.load(r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\HW3\X_N_1000_d_40_sig_0_01.npy')
y = np.load(r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\HW3\X_N_1000_d_40_sig_0_01.npy')

X_lifted = lift(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_lifted, y, test_size=0.30, random_state=42)

cv = KFold(n_splits=5, random_state=42, shuffle=True)

alphas = np.logspace(-10, 10, 21)
mean_rmse = []
std_rmse = []

for alpha in alphas:
    model = Lasso(alpha=alpha)
    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
    mean_rmse.append(-np.mean(scores))
    std_rmse.append(np.std(scores))

optimal_alpha = alphas[np.argmin(mean_rmse)]

model_optimal = Lasso(alpha=optimal_alpha)
model_optimal.fit(X_train, y_train)

rmse_train = np.sqrt(mean_squared_error(y_train, model_optimal.predict(X_train)))
rmse_test = np.sqrt(mean_squared_error(y_test, model_optimal.predict(X_test)))

print(f"Optimal α: {optimal_alpha}")
print(f"Train RMSE: {rmse_train}, Test RMSE: {rmse_test}")

plt.figure(figsize=(10, 6))
plt.errorbar(alphas, mean_rmse, yerr=std_rmse, fmt='-o')
plt.xscale('log')
plt.xlabel('α (Lasso Regularization Parameter)')
plt.ylabel('Mean RMSE (with error bars)')
plt.title('Cross-Validation Mean RMSE vs. α')
plt.grid(True)
plt.show()
