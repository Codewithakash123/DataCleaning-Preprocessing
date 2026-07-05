import pandas as pd
data = pd.read_csv("Titanic-Dataset.csv")

print("First 5 Rows:")
print(data.head())

print("\n Dataset information:")
print("data.info()")

print("\n Missing values:")
print(data.isnull().sum()) 

data["Age"] = data["Age"].fillna(data["Age"].mean())

data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

data.drop("Cabin", axis = 1 , inplace = True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Sex"] = le.fit_transform(data["Sex"])

print("\n After Encoding:")
print(data.head())

from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()
data[["Age","Fare"]] = Scaler.fit_transform(data[["Age","Fare"]])

print("\nAfter Standardization:")
print(data[["Age","Fare"]].head())

import matplotlib.pyplot as plt
import seaborn as sns

sns.boxplot(x=data["Fare"])

plt.title("Boxplot of Fare")
plt.show()

Q1 = data["Fare"].quantile(0.25)
Q3 = data["Fare"].quantile(0.25)

IQR = Q3 -Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

data = data[(data["Fare"] >= lower) & (data["Fare"] <= upper)]

print("\n Dataset after removing outliers:")
print(data.shape)

data.to_csv("Cleaned_Titanic.csv", index = False)
print("\nCleaned dataset saved successfully .")