import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

k = 2
csv = pd.read_csv('compound_data.csv')
selection = csv.iloc[:,2:]
names = csv.iloc[:,1]
data = selection.as_matrix()
kmeans = KMeans(n_clusters = k).fit(data)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
col = ['red', 'orange', 'yellow', 'yellowgreen', 'green', 'blue']

### Plot clusters
for i in range(k):
	ds = data[np.where(labels==i)]
	plt.plot(ds[:,0],ds[:,1], 'o', color = col[i], alpha=0.4)
	plt.plot(centroids[i,0], centroids[i,1], 'o', alpha=0.5, color = 'black')
	plt.title('K-means Clustering of Anticancer Compounds')
	plt.xlabel('IC50')
	plt.ylabel('Measured Concentration')
plt.show()
