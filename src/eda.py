import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def show_univariate_analysis(df, col_name, bins=30):
    print(col_name)
    print(df[col_name].describe())

    #histogram
    plt.figure(figsize=(4, 3))
    sns.histplot(df[col_name], bins=bins)
    plt.title(f"{col_name} Histogram")
    plt.show()

    #kdeplot
    plt.figure(figsize=(4, 3))
    sns.kdeplot(df[col_name])
    plt.title(f"{col_name} KDE Plot")
    plt.show()

    #boxplot
    plt.figure(figsize=(4, 3))
    sns.boxplot(df[col_name])
    plt.title(f"{col_name} Boxplot")
    plt.show()