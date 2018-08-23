import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Instantiate models
svr = SVR()
lr = LinearRegression()
knr = KNeighborsRegressor()
dtr = DecisionTreeRegressor()
mlpr = MLPRegressor()
gbr = GradientBoostingRegressor()

# Extract data
df_compounds = pd.read_csv('compound_data_IC50.csv')

# Define feature, target arrays
X, y = df_compounds.iloc[:, 0:9].values, df_compounds.iloc[:, 9].values

# Split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=21)

# Fit models to training set
svr.fit(X_train, y_train)
lr.fit(X_train, y_train)
knr.fit(X_train, y_train)
dtr.fit(X_train, y_train)
mlpr.fit(X_train, y_train)
gbr.fit(X_train, y_train)

# Predict IC50, calculate MSE and r2 for each model
mse_svr, r2_svr = mean_squared_error(y_test, svr.predict(X_test)), r2_score(y_test, svr.predict(X_test))
mse_lr, r2_lr = mean_squared_error(y_test, lr.predict(X_test)), r2_score(y_test, lr.predict(X_test))
mse_knr, r2_knr = mean_squared_error(y_test, knr.predict(X_test)), r2_score(y_test, knr.predict(X_test))
mse_dtr, r2_dtr = mean_squared_error(y_test, dtr.predict(X_test)), r2_score(y_test, dtr.predict(X_test))
mse_mlpr, r2_mlpr = mean_squared_error(y_test, mlpr.predict(X_test)), r2_score(y_test, mlpr.predict(X_test))
mse_gbr, r2_gbr = mean_squared_error(y_test, gbr.predict(X_test)), r2_score(y_test, gbr.predict(X_test))

print('Model\t MSE\t r2\t')
print('SVR\t {0:.4f}\t {1:.4f}\t'.format(mse_svr, r2_svr))
print('LR\t {0:.4f}\t {1:.4f}\t'.format(mse_lr, r2_lr))
print('KNR\t {0:.4f}\t {1:.4f}\t'.format(mse_knr, r2_knr))
print('DTR\t {0:.4f}\t {1:.4f}\t'.format(mse_dtr, r2_dtr))
print('MLPR\t {0:.4f}\t {1:.4f}\t'.format(mse_mlpr, r2_mlpr))
print('GBR\t {0:.4f}\t {1:.4f}\t'.format(mse_gbr, r2_gbr))
