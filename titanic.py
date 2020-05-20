# -*- coding: utf-8 -*-
"""
Created on Mon May 18 18:07:48 2020

@author: Mudhurika
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

titanic = pd.read_csv('train.csv')

#--------------Feature engineering------------------------------------------

#checking for null values in each column
null_columns=titanic.columns[titanic.isnull().any()]
titanic[null_columns].isnull().sum()

#replacing the null value of each column
titanic["Age"].fillna(titanic['Age'].mean(), inplace = True) # replace with mean value
titanic["Cabin"].fillna('No Data', inplace = True)
titanic["Embarked"].fillna('Not known', inplace = True)
titanic['Ticket'].fillna('Not Available',inplace = True)

#Finding the number of woemn and men on the journey who survived
women = titanic.loc[titanic.Sex == 'female']['Survived']
perc_women = sum(women)/len(women)

men = titanic.loc[titanic.Sex == 'male']['Survived']
perc_men = sum(men)/len(men) #We can see that the survival rate of women is more
 
cases = titanic.groupby(['Sex']).sum().reset_index()
#Visualize this using pi chart 
color = ['Pink','Blue']
plt.pie(cases['Survived'], labels=cases['Sex'], colors=color,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

#Finding each class of passenger who survived titanic 
classes =  titanic.groupby(['Pclass']).sum().reset_index()
first =  titanic.loc[titanic.Pclass == 1]['Survived']
perc_first= sum(first)/len(first)

second =  titanic.loc[titanic.Pclass == 2]['Survived']
perc_second= sum(second)/len(second)

third =  titanic.loc[titanic.Pclass == 3]['Survived']
perc_third= sum(third)/len(third)

color1 = ['red','green','blue']
plt.pie(classes['Pclass'], labels=classes['Pclass'], colors=color1,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()
#----------------------Data Preprocessing-----------------------------------------
X = titanic.iloc[:, [0,2,4,5,6,7,8,9,10,11]].values
y = titanic.iloc[:, 1].values

#Label Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 1])
labelencoder_X = LabelEncoder()
X[:, 6] = labelencoder_X.fit_transform(X[:, 5])
labelencoder_X = LabelEncoder()
X[:, 8] = labelencoder_X.fit_transform(X[:, 7])
labelencoder_X = LabelEncoder()
X[:, 9] = labelencoder_X.fit_transform(X[:, 8])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#----------------------Regression Model---------------------------------------
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

