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
irrelevant_columns = [col for col in data.columns if 'id' in col.lower()]
data = data.drop(columns=irrelevant_columns, axis=1)

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
# 3. Handle Missing Values
"""
1. High Null Ratios:
    o Drop columns where the null-value percentage is too high (e.g., >50%).
"""
high_nullable_cols = [col for col in data.columns if data[col].isnull().mean() > 0.5]
# high_nullable_cols=[col for col in data.columns if data[col].isnull().sum()/len(data)]
print(f"Columns with high null ratios (>50%): {high_nullable_cols}")
"""
Categorical Columns:
    o Fill missing values with the mode of the column.
"""
print("=====================mode======================")
for col in categorical_columns:
    if(data[col].isnull().sum()>0):
        data[col].fillna(data[col].mode()[0],inplace=True)
        print(f"column {col} filled with mode value {data[col].mode()[0]}")
    else:
        print(f"no missing values in the column {col}")
print("=====================Numerical Columns:======================")
"""
Numerical Columns:
    o Visualize the distribution of each column (e.g., using histograms or skewness statistics).
    o If skewed, fill missing values with the median to reduce the effect of outliers.
    o For symmetric distributions, use the mean for imputation.
"""
for col in numeric_columns:
    data[col].hist(bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("frequency")
    plt.show()
for col in numeric_columns:
    if data[col].isnull().sum()>0:
        skewness=data[col].skew() # حساب الانحراف
        if abs(skewness) > 1:  # لو كان الانحراف كبير 
            median_value=data[col].median()
            data[col].fillna(median_value,inplace=True)
            print(f"column {col} filled with median_value = {median_value}")
        else:
            mean_value=data[col].mean()
            data[col].fillna(mean_value,inplace=True)
            print(f"column {col} filled with mean_value =  {mean_value}")
    else:
        print(f"no skew for column {col}")
"""
Validate Null Handling:
    o Recheck the dataset to ensure no missing values remain
"""
if  data.isnull().sum().sum() ==0: # اول مجموع هو مجموع كل عمود لوحده لكن الباقي مجموع كل الاعمده مع بعض
    print("all missing values have been handeled")
else:
    print(f"there is still {data.isnull().sum().sum()} missing values have been handeled")

for col in numeric_columns:
    data[col].drop_duplicates()
    print("Done")
print(f"Memory usage after optimization: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
input("Enter any key")
