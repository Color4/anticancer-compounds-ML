import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Extract data
df_compounds = pd.read_csv('compound_data_names.csv')

# Define pipeline steps
steps = [('scaler', StandardScaler()), ('lr', LogisticRegression())]

# Instantiate pipeline
pipeline = Pipeline(steps)

# Define hyperparameter space
parameters = {'lr__solver' : ['newton-cg', 'lbfgs', 'sag', 'saga']}

# Instantiate grid search
cv = GridSearchCV(pipeline, param_grid=parameters)

# Define feature, target arrays
X, y = df_compounds.iloc[:, 0:9].values, df_compounds.iloc[:, 9].values

# Split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=21)

# Fit classifiers to training set
lr_unscaled = LogisticRegression()
lr_unscaled.fit(X_train, y_train)
lr_scaled = cv.fit(X_train, y_train)

# Predict the label values
y_pred = lr_scaled.predict(X_test)

# Compute accuracies
acc_unscaled = lr_unscaled.score(X_test, y_test)
acc_scaled = lr_scaled.best_score_

print('\nLogistic Regression')
print('---------------------------------------------------------------')
print('Classification Report: \n', classification_report(y_test, y_pred))
print('Tuned Model Parameters: {}\n'.format(lr_scaled.best_params_))
print('Accuracy w/o scaling:       %0.9f' % acc_unscaled)
print('Accuracy with scaling:      %0.9f' % acc_scaled)
print('Improvement due to scaling: %0.9f\n' % (acc_scaled - acc_unscaled))
