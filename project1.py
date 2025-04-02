import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)
df =pd.read_csv('/Users/ijsschool/Documents/vscode/NORS_20250320.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.columns)
print(df.shape)
print(df.isnull().sum())
print(df.nunique())
df.isna().sum(axis=0).sort_values(ascending=False) / df.shape[0] * 100

#finding the unique values of the columns
for column in df.columns:
  print(f"Column: {column}")
  print(f"Unique Values: {df[column].unique()}")
  print("-" * 20)


#the primary mode of contamination
df['Primary Mode'].value_counts().plot(kind='bar')
plt.title('Primary Mode of Contamination')
plt.xlabel('Primary Mode')
plt.ylabel('Count')
plt.show()

fig, ax = plt.subplots(figsize=(24, 12))
sns.countplot(x='Illnesses', hue='Primary Mode', data=df)
plt.title('Illness vs. Primary mode')
plt.xlabel('Illness')
plt.ylabel('Count')
ax.xaxis.set_major_locator(plt.MaxNLocator(10)) 
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.show() 

#year of illness
sns.countplot(x='Illnesses', hue='Primary Mode', data=df)
plt.title('Illness vs. Primary mode')
plt.xlabel('Illness')
plt.ylabel('Count')
ax.xaxis.set_major_locator(plt.MaxNLocator(10)) 
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.show()

# Scatter plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='Year', y='Illnesses', data=df)
plt.title('Year vs. Illnesses')
plt.xlabel('Year')
plt.ylabel('Illnesses')
ax.yaxis.set_major_locator(plt.MaxNLocator(20))
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df['Year'], df['Illnesses'], marker='o')
plt.title('Year vs. Illnesses')
plt.xlabel('Year')
plt.ylabel('Illnesses')
plt.grid(True)
plt.show()

# graph with Year and Primary Mode
plt.figure(figsize=(12, 6))
sns.countplot(x='Year', hue='Primary Mode', data=df)
plt.title('Year vs. Primary Mode of Contamination')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.legend(title='Primary Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


#most common setting of contamination
df['Setting'].value_counts().plot(kind='bar')
plt.title('Setting of Contamination')
plt.xlabel('Setting')
plt.ylabel('Count')
plt.locator_params(axis='x', nbins=20)
plt.show()

df['Animal Type'].fillna('Unknown', inplace=True) #filling missing values with 'Unknown'
df['Animal Type'].value_counts().plot(kind='bar')
plt.title('Animal Type')
plt.xlabel('Animal Type')
plt.locator_params(axis='x', nbins=10)
plt.ylim(0, 2000)
plt.ylabel('Count')
plt.show() 

#filling missing values often with 'Unknown'
for columns in df.columns:
  if df[columns].dtype == 'object':
    df[columns]= df[columns].fillna('Unknown')
  else:
    print(df[columns].isnull().sum()) 

#filling in the missing values of the float columns with the mean
for columns in df.columns:
  if df[columns].dtype == 'float64':
    df[columns]= df[columns].fillna(df[columns].mean())
  else:
    print(df[columns].isnull().sum())

df.isnull().sum()



#converting the categorical columns to numerical columns
from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
le = LabelEncoder()

# Iterate through the columns of the DataFrame
for column in df.columns:
  # Check if the column is of object data type
  if df[column].dtype == 'object':
    # Convert all values to strings before encoding
    df[column] = df[column].astype(str)  
    # Fit and transform the column using LabelEncoder
    df[column] = le.fit_transform(df[column]) 
print(df.info()) 


#doing a heatmap to check for correlation
sns.heatmap(df.corr(), annot=True)
plt.show()

df.drop(['Animal Type','Water Exposure'], axis=1, inplace=True)

#Visualizing the distribution of the target variable
plt.figure(figsize=(8, 6))
sns.histplot(df['Illnesses'])
plt.title('Distribution of Illnesses')
plt.xlabel('Illnesses')
plt.ylabel('Count')
plt.show()


#VIF calculation, any feature with a VIF value greater than 10 is considered to be highly correlated with other features
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_vif = df.drop('Illnesses', axis=1)
vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
print(vif_data)


#splitting the data into features and target
X = df.drop('Illnesses', axis=1)
y = df['Illnesses']
