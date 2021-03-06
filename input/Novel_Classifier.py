import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Import data
df = pd.read_csv('compound_data_IC50.csv')

# Add column of custom category labels
def category(p):
	if p['IC50'] <= 0:
		return 'Non-inhibitor'
	elif p['IC50'] > 0 and p['IC50'] <= 2:
		return 'Weak inhibitor'
	elif p['IC50'] > 2 and p['IC50'] <= 4:
		return 'Moderate inhibitor'
	elif p['IC50'] > 4 and p['IC50'] <= 6:
		return 'Potent inhibitor'
	else:
		return 'Very Potent inhibitor'

df['Potency'] = df.apply(category, axis=1)

# Unit test for report screenshot
#print(df.groupby(['Potency']).size())

# Define feature, target arrays
X, y = df.iloc[:, 0:9].values, df.iloc[:, 10].values

# Split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=21)

# Define pipeline steps
steps = [('scaler', StandardScaler()), ('SVM', SVC())]

# Instantiate pipeline
pipeline = Pipeline(steps)

# Define hyperparameter space
parameters = {'SVM__C' : [1,10,100], 'SVM__gamma' : [0.1, 0.01]}

# Instantiate grid search and fit to training set
cv = GridSearchCV(pipeline, param_grid=parameters)
clf = cv.fit(X_train, y_train)
clf_unscaled = SVC().fit(X_train, y_train)

# Predict the label values
y_pred = clf.predict(X_test)

# Compute accuracy
acc_scaled = clf.score(X_test, y_test)
acc_unscaled = clf_unscaled.score(X_test, y_test)

print('\nNovel Classifier Report')
print('---------------------------------------------------------------')
print('Algorithm: Support Vector Machine\n', classification_report(y_test, y_pred))
print('Tuned Model Parameters: {}\n'.format(cv.best_params_))
print('Accuracy w/o scaling:       %0.9f' % acc_unscaled)
print('Accuracy with scaling:      %0.9f' % acc_scaled)
print('Improvement due to scaling: %0.9f\n' % (acc_scaled - acc_unscaled))
