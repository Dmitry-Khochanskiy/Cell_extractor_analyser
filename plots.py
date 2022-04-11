# Descriptive statistics for DF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set_theme()
from cellobj import CellObj, add_measured_value, save_objects_as_pickle
import pickle


def pair_grid(df, columns_for_analysis):
# https://seaborn.pydata.org/tutorial/distributions.html
# pair grid
    g = sns.PairGrid(df[columns_for_analysis])
    g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot, fill=True)
    g.map_diag(sns.histplot, kde=True)
    plt.show()

def heat_map(df, columns_for_analysis):
# heat map on normalizied image
    df_norm = df / df.sum()
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    ax = sns.heatmap(df_norm[columns_for_analysis])
    plt.show()

def correlation_heat_map(df, columns_for_analysis):
#https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e
# Correlation heatmap
    plt.figure(figsize=(16, 6))
    heatmap = sns.heatmap(df[columns_for_analysis].corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12)
    plt.show()
