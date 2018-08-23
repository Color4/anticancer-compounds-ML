import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


df_compounds = pd.read_csv('compound_data_names.csv')

### Extract and scale data
X, y = df_compounds.iloc[:, 0:9].values, df_compounds.iloc[:, 9].values
sc = StandardScaler()
X_std = sc.fit_transform(X)

### Construct covariance matrix
cov_mat = np.cov(X_std.T)
print('\nCovariance Matrix \n%s' % cov_mat)

### Compute eigenvalues and eigenvectors
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)

### Calculate cumulative sum of explained variances
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(1,10), var_exp, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1,10), cum_var_exp, where='mid', label='cumulative explained variance')
plt.title('Explained Variance')
plt.xlabel('principal components')
plt.ylabel('ratio')
plt.legend(loc='best')
plt.show()

### Decompose covariance matrix into eigenpairs
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)

### Create a 2D projection matrix
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
				eigen_pairs[1][1][:, np.newaxis]))
print('\nMatrix W:\n', w)

### Reduce Dimensionality via PCA
X_pca = X_std.dot(w)
colors = ['blue', 'green', 'yellowgreen', 'yellow', 'orange', 'red']
markers= ['o','o','o','o','o','o']
plt.style.use('ggplot')
for l, c, m in zip(np.unique(y), colors, markers):
	plt.scatter(X_pca[y == l, 0],
				X_pca[y == l, 1],
				c=c, label=l, marker=m, alpha=0.4)
plt.title('PCA of Unknown Anticancer Compounds')
plt.xlabel('first principal component')
plt.ylabel('second principal component')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()
