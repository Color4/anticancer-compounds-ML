import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

clf_data = pd.read_csv('regressor_performance.csv')
algo = clf_data.iloc[:,0] # first column is model names
acc = clf_data.iloc[:,2] # third column is R^2

fig, ax = plt.subplots()

colors = ['sienna','darkgoldenrod','goldenrod','gold','yellow','palegoldenrod']

clf_bars = ax.bar(range(len(algo)), acc, color=colors, tick_label=algo)

ax.set_ylabel(r'$R^2$')
ax.set_ylim(0.65, 0.95)
ax.set_xlabel('Algorithm')
ax.set_title('Regressor Performance')
ax.grid(linestyle='dotted')
plt.show()
