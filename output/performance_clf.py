import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

clf_data = pd.read_csv('classifier_performance.csv', index_col=False)
algo = clf_data.iloc[:,0] # first column
acc = clf_data.iloc[:,1] # second column

fig, ax = plt.subplots()

colors = ['midnightblue','darkblue','royalblue','cornflowerblue','deepskyblue','cyan']

clf_bars = ax.bar(range(len(algo)), acc, color=colors, tick_label=algo)

ax.set_ylabel('Accuracy')
ax.set_ylim(0.4, 0.75)
ax.set_xlabel('Algorithm')
ax.set_title('Classifier Performance')
ax.grid(linestyle='dotted')
plt.show()
