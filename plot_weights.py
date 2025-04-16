import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from seaborn.external.kde import gaussian_kde

from src.commons import shared_variables as shared

folder_path = shared.output_folder / 'Encodings_results'
file_path = os.path.join(folder_path, "aggregated_results_weights_new.csv")
df = pd.read_csv(file_path, delimiter=',')
#df['dataset'] = df['Dataset'].apply(lambda x: str(x).split('_')[0])
grouped = df.groupby(['weight'])
total_counts = grouped.size().reset_index(name='total_count')
total_counts['total_count']=total_counts['total_count']/(8*3*12)
fig, ax = plt.subplots(figsize=(5, 5))
#sns.lineplot(data=total_counts, x='weight',y='total_count', marker='o')
#sns.barplot(data=total_counts, x='weight', y='total_count', color='skyblue')
#sns.histplot(data=grouped, x='weight', stat='probability',bins=5, color='skyblue')
#sns.kdeplot(data=total_counts, x='weight', y='total_count', color='red')
sns.histplot(data=total_counts, x='weight',bins=5, weights='total_count',color='skyblue')
ax.set_xticks([0.5, 0.6,0.7,0.8,0.9])
#ax.set_yticks([0.4, 0.45, 0.5, 0.55, 0.6])
plt.xlabel('weight (w)')
plt.ylabel(' ')
#ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.tight_layout()
#plt.show()
title = "weights hist_plot"
plt.savefig(os.path.join(folder_path, f'{title}.pdf'))
