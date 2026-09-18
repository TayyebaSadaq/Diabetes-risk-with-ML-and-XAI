import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/diabetes.csv") # loading dataset

## Inspecting the dataset
# print(df.head()) # prints out the first few rows of dataset
# print(df.info()) # prints out summary of the dataset
# print(df.isnull().sum()) # prints out the number of missing values in each column

## Stat summary and visualisation of outliers
