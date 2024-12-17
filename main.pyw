import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import category_encoders as ce
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder ,StandardScaler
from sklearn.tree import DecisionTreeClassifier
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
# axis لو هي 1 معناها اعمده لكن لو هي 0 معناها صف
data = data.drop(columns=irrelevant_columns, axis=1)
# data = data.dropna(subset=['LeaveOrNot'])
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

# ++++++++++++++++++++++++++++++++++++++++4. Outlier Detection and Treatment
"""
1. Visualize Outliers:
    o Use box plots to detect outliers in numerical columns.
"""
for col in numeric_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data[col])
    plt.title(f"Box Plot for {col}")
    plt.xlabel(col)
    plt.show()
"""
Capping Outliers:
    o Replace values above the upper whisker with the maximum non-outlier value (upper bound).
    o Replace values below the lower whisker with the minimum non-outlier value (lower bound).
"""
# Handle Categorical Outliers
print("\n===================== Handling Categorical Outliers =====================")
for col in categorical_columns:
    category_counts = data[col].value_counts()
    rare_categories = category_counts[category_counts < 0.05 * len(data)].index

    if not rare_categories.empty:
        mode_value = data[col].mode()[0]
        # Handle categorical data outliers
        if data[col].dtype.name == 'category':
            data[col] = data[col].cat.set_categories(
                data[col].cat.categories.union([mode_value])
            )
        # data[col].replace(rare_categories, mode_value, inplace=True)
        print(f"Replaced rare categories in column {col} with mode value {mode_value}")
"""
5. Check for Duplicates
    • Remove duplicate rows using drop_duplicates().
"""
# data.drop_duplicates(inplace=True)
"""
Drop Low-Variance Columns
    • Remove columns with very low variance (e.g., standard deviation close to zero).
"""
print("Drop Low-Variance Columns")
# Calculate the standard deviation for each column
std_devs = data[numeric_columns].std()
# Set a threshold for low variance (e.g., columns with a standard deviation below 0.01)
low_variance_cols = std_devs[std_devs < 0.01].index
# Drop the low-variance columns
data.drop(columns=low_variance_cols, inplace=True)
print(f"Columns with low variance removed: {list(low_variance_cols)}")
"""
Feature and Label Separation
    • Split the dataset into:
    o Features (X): All independent variables.
    o Label (y): Target variable
"""
# Define the target column
target_column = 'LeaveOrNot'
# Features (X): All columns except the target column
X = data.drop(columns=[target_column])
# Label (y): The target column
y = data[target_column]
print(f"Features shape (X): {X.shape}")
print(f"Label shape (y): {y.shape}")

# =================== Encoding Categorical Columns ===================
"""
Encoding Categorical Columns
1. Ordinal Data:
    o Use label encoding.
2. High-Cardinality Columns:
    o Use binary encoding or frequency encoding.
3. Low to Medium Cardinality Columns (3-6 categories):
    o Use one-hot encoding to represent these categories.
"""
# 1. Ordinal Data: Use Label Encoding
label_encoder = LabelEncoder()
ordinal_columns = ['Education']  # التعليم يحتوي على ترتيب منطقي
for col in ordinal_columns:
    data[col] = label_encoder.fit_transform(data[col])
    print(f"Column {col} encoded using Label Encoding.")
# 2. High-Cardinality Columns: Use Frequency Encoding
high_cardinality_columns = ['City']  # عمود المدن ذو فئات كبيرة
for col in high_cardinality_columns:
    freq_encoding = data[col].value_counts() / len(data)
    data[col] = data[col].map(freq_encoding)
    print(f"Column {col} encoded using Frequency Encoding.")
# 3. Low to Medium Cardinality Columns: Use One-Hot Encoding
low_cardinality_columns = ['PaymentTier', 'Gender', 'EverBenched']  # فئات صغيرة إلى متوسطة
data = pd.get_dummies(data, columns=low_cardinality_columns, drop_first=True)
print(f"Columns {low_cardinality_columns} encoded using One-Hot Encoding.")
"""
1. Split Data
    • Divide the dataset into training and testing subsets (e.g., 80% train, 20% test).
    • Use train_test_split() from sklear
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# Define the column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', ce.OneHotEncoder(), ['Education', 'City', 'PaymentTier', 'Gender', 'EverBenched']),  # معالجة البيانات النصية
        ('num', 'passthrough', ['JoiningYear', 'Age', 'ExperienceInCurrentDomain'])  # البيانات العددية تبقى زي ما هي
    ])

# إنشاء الأنابيب (Pipeline) التي ستتضمن:
# 1. المعالج (Preprocessing)
# 2. موديل اللوجيستيك ريجريشن (Logistic Regression)
# logistic regression = 0.74
# DecisionTreeClassifier = 0.86
# RandomForestClassifier = 0.85
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),  # إضافة الموازنة
    ('classifier', DecisionTreeClassifier(random_state=42, max_depth=5))  # استخدام RandomForestClassifier
])

# تدريب الموديل على البيانات التدريبية
model.fit(X_train, y_train)
# التنبؤ على البيانات الاختبارية
y_pred = model.predict(X_test)
# تقييم أداء الموديل باستخدام الدقة
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# طباعة تقرير التصنيف (التقييم التفصيلي)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# حساب وتحليل Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# طباعة Confusion Matrix
print("Confusion Matrix:")
print(cm)

# رسم Confusion Matrix باستخدام Seaborn
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print(f"Memory usage after optimization: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
input("Enter any key")
# مثال: بيانات موظف جديد
new_employee = pd.DataFrame({
    'Education': ['Bachelors'],
    'City': ['Bangalore'],
    'PaymentTier': [1],
    'Gender': ['Male'],
    'EverBenched': ['No'],
    'JoiningYear': [2019],
    'Age': [28],
    'ExperienceInCurrentDomain': [5]
})

# تأكد أن البيانات الجديدة يتم ترميزها بنفس الطريقة المستخدمة مع البيانات الأصلية
new_employee['Education'] = label_encoder.transform(new_employee['Education'])
new_employee['City'] = new_employee['City'].map(data['City'].value_counts() / len(data))

# قم بترميز الفئات الصغيرة والمتوسطة (One-Hot Encoding)
new_employee = pd.get_dummies(new_employee, columns=['PaymentTier', 'Gender', 'EverBenched'], drop_first=True)

# قم بترتيب الأعمدة بحيث تتطابق مع بيانات X
new_employee = new_employee.reindex(columns=X.columns, fill_value=0)

# التنبؤ باستخدام النموذج
prediction = model.predict(new_employee)
print(f"The employee is predicted to {'leave' if prediction[0] == 1 else 'stay'}")
