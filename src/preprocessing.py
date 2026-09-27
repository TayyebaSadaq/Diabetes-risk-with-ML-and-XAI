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
    print(data.nunique()) # prints out number of unique values in each column - helps to distinguish what columns will need outlier removal
    
    ## graph visualisation of outliers and data spread
    fig,axs=plt.subplots(len(data.columns),1,figsize=(7,18), dpi=95)
    for i, col in enumerate(data.columns):
        axs[i].boxplot(data[col], vert=False) # plots boxplot for each column
        axs[i].set_ylabel(col) # sets y-axis label to column name
    plt.tight_layout()
    plt.show()

## correlation analysis between features and target variable
def correlation_analysis(data):
    ## visual representation
    correlation = data.corr() # calculates correlation between features and target variable
    plt.figure(dpi=130)
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm') # plots heatmap with seaborn clearly
    plt.show()
    
    print(correlation["Diabetes_012"].sort_values(ascending=False)) # prints out correlation values of features with target variable

## visualise target variable distribution
def target_distribution(data):
    plt.pie(data['Diabetes_012'].value_counts(), #plotting pie chart of target var
        labels=[0,1,2],
        autopct='%1.1f%%')
    plt.title('Target Variable Distribution')
    plt.show()

## seperate features and target variable
X = data.drop(columns=['Diabetes_012']) # features
Y = data['Diabetes_012'] # target

## Feature scaling - normalisation and standardisation - left out for now?

 
# inspect_data(data)
# stat_summary(data)
# correlation_analysis(data)
# target_distribution(data)jani
