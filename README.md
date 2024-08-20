# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import pandas numpy and matplotlib
2.Upload a file that contains the required data
3.find x,y using sklearn
4.Use line chart and disply the graph and print the mse, mae,rmse.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: RAMYA.P
RegisterNumber:212223240137 

```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('/content/studentscores.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(X_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')
m=lr.coef_
m[0]
b=lr.intercept_
b
```
## Output:
![image](https://github.com/user-attachments/assets/744a7a57-a258-492a-b947-34e5c20cc3b6)

![image](https://github.com/user-attachments/assets/9c832bc0-f779-4dff-83ec-f26db02a2eab)

![image](https://github.com/user-attachments/assets/d80b47ee-3f47-4e48-a56b-dea14aa9362d)

![image](https://github.com/user-attachments/assets/65182c39-0133-4799-8424-e7b3f5890c6c)

![image](https://github.com/user-attachments/assets/366cbbbe-3e63-48a7-9f36-1dfc2426de22)

![image](https://github.com/user-attachments/assets/f7de5af8-1444-4a4b-9628-cb2aecdfab80)

![image](https://github.com/user-attachments/assets/08cba8e2-07a2-4c9d-a48a-e25f4686f376)

![image](https://github.com/user-attachments/assets/11f0ac9e-60d7-4158-8b59-ca5a518f1332)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
