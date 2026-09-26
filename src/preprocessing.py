## Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

## Importing dataset
data = pd.read_csv("data/diabetes.csv")

## Inspecting the dataset
def inspect_data(data):
    print(data.head()) # prints out first rows of dataset
    print(data.info()) # prints summary of dataset
    print(data.isnull.sum()) # checks if there's any missing values
    
## Statistical summary of dataset
def stat_summary(data):
    print(data.describe()) # prints out statistical summary of dataset
    
    # graph visualisation of outliers and data spread
    fig,axs=plt.subplots(len(data.columns),1,figsize=(7,18), dpi=95)
    for i, col in enumerate(data.columns):
        axs[i].boxplot(data[col], vert=False) # plots boxplot for each column
        axs[i].set_ylabel(col) # sets y-axis label to column name
    plt.tight_layout()
    plt.show()

## removing outliers via IQR methods
def remove_outliers(data):
    pass
    
# inspect_data(data)
#stat_summary(data)