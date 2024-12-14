import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# 1. Read Dataset
data=pd.read_csv("Employee.csv")
# print(data["Education"].unique())
memory_before = data.memory_usage(deep=True).sum()
print(f"\nMemory usage before optimization: {memory_before / 1024**2:.2f} MB")

# print(data.head())
# print(data.info())
# print(data.describe())
# end 1. Read Dataset
# +++++++++++++++++++++++++++2. Explore the Data ++++++++++++++++++++++++++++++++++++++++++++
"""
1. Basic Exploration:
    o Print the first few rows to understand the structure of the dataset.
    o Drop irrelevant columns like IDs if they don`t contribute to the target variable.
"""
# print(data.head())
# data.drop("Age",axis=1,inplace=True)
# print(data.columns)
# print(data.dtypes)

"""
2. Check Datatypes:
    o Identify numerical and categorical columns.
    o Convert categorical columns to the category datatype for better memory efficiency.
"""
# بنقسم الاعمده الارقام عن غير الارقام
numeric_columns=data.select_dtypes(include=np.number).columns.tolist()
categorical_columns=data.select_dtypes(exclude=np.number).columns.tolist()
# هنا احنا بنلف على الاعمده عشان نحول الاعمده من object الي category
for col in categorical_columns:
    data[col]=data[col].astype("category")

"""
3. Categorical Column Analysis:
    o Display the number of unique categories in each categorical column.
"""
# هنا بنقسم categories وبيطلعلك كل نوع فى كام نوع
# for example print(data["Education"].unique())  = ['Bachelors' 'Masters' 'PHD']
for col in categorical_columns:
    # ده بيرجع categories اللى فى كل عمود فى array
    # print(f"{col}: {list(data[col].cat.categories)}")
    # print("++++++++++++++++++++++++++++++++++++++++++++")
    # بيرجع عدد categories 
    print(f"{col}: {data[col].nunique()}  {list(data[col].cat.categories)} unique categories")

"""
Missing Values:
    o Check for missing values in each column.
    o Calculate the percentage of missing values.
"""
print(f"sum of missing values in each column \n{data.isnull().sum()}")
print(f"percentage of missing values in each column \n{(data.isnull().sum())/len(data)} ")

memory_after = data.memory_usage(deep=True).sum()
print(f"Memory usage after optimization: {memory_after / 1024**2:.2f} MB")
input("Enter any key")