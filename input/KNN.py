import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Extract data
df_compounds = pd.read_csv('compound_data_names.csv')

# Define pipeline steps
steps = [('scaler', MaxAbsScaler()), ('knn', KNeighborsClassifier())]

# Instantiate pipeline
pipeline = Pipeline(steps)

# Define hyperparameter space
parameters = {'knn__n_neighbors' : np.arange(1,20) }

# Instantiate grid search
cv = GridSearchCV(pipeline, param_grid=parameters)

# Define feature, target arrays
X, y = df_compounds.iloc[:, 0:9].values, df_compounds.iloc[:, 9].values

# Split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=21)

# Fit classifiers to training set
knn_unscaled = KNeighborsClassifier()
knn_unscaled.fit(X_train, y_train)
knn_scaled = cv.fit(X_train, y_train)

# Predict the label values
y_pred = knn_scaled.predict(X_test)

# Compute accuracies
acc_unscaled = knn_unscaled.score(X_test, y_test)
acc_scaled = knn_scaled.best_score_

print('\nK-Nearest Neighbors')
print('---------------------------------------------------------------')
print('Classification Report: \n', classification_report(y_test, y_pred))
print('Tuned Model Parameters: {}\n'.format(knn_scaled.best_params_))
print('Accuracy w/o scaling:       %0.9f' % acc_unscaled)
print('Accuracy with scaling:      %0.9f' % acc_scaled)
print('Improvement due to scaling: %0.9f\n' % (acc_scaled - acc_unscaled))
