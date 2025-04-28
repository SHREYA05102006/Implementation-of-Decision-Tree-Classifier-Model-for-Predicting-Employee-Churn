# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: V.SHREYA
RegisterNumber:212224230266
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv("C:\\Users\\admin\\Downloads\\Employee (1).csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()    #no departments and no left
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
​
​
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()

```

## Output:
### data head
![image](https://github.com/user-attachments/assets/22fdc984-596c-4748-b4e4-b59d0d743b67)
### info
![image](https://github.com/user-attachments/assets/f4f9b603-489b-4798-998e-2fd7f306124b)
### null set
![image](https://github.com/user-attachments/assets/4602c794-0cff-4c13-a4e7-fa0a4ff6205a)
### VALUES COUNT IN THE LEFT COLUMN
![image](https://github.com/user-attachments/assets/5ae642f3-bd1d-4a83-8416-40500f67d8f9)
### DATASET TRANSFORMED HEAD:
![image](https://github.com/user-attachments/assets/2acdbb89-7a71-4861-9e3f-6792606d7c1c)
### X.HEAD:
![image](https://github.com/user-attachments/assets/93889d87-4a35-43f2-b7fb-54d5df472f71)
### ACCURACY:
![image](https://github.com/user-attachments/assets/92a46908-c61a-482e-b36b-1f44d825266b)
### DATA PREDICTION:
![image](https://github.com/user-attachments/assets/4cc05c6f-ddc4-42b0-a2cb-1cf6db882a5c)
![image](https://github.com/user-attachments/assets/b61b0502-ab04-46a4-b055-666cbdb0e508)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
