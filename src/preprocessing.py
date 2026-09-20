## Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import standardscaler, minmaxscaler
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
    
# inspect_data(data)
stat_summary(data)