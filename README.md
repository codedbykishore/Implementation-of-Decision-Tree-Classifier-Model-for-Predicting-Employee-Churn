# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required libraries.

2. Upload and read the dataset.

3. Check for any null values using the isnull() function.

4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: KISHORE B
RegisterNumber:  212223240073
*/
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,206,6,0,1,2]])
```

## Output:
### Data Head:
![318633191-13726f73-a3be-462d-a126-276b0d8fee29](https://github.com/codedbykishore/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139122/30356310-f8c9-4f1d-91d0-3db92f3207c1)

### Dataset info:
![318633273-1b111b8d-93df-481a-a44a-39148c64d82a](https://github.com/codedbykishore/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139122/c9a133c2-e3b7-4857-be2e-0978eab4d3d1)


### Null Dataset:
![318633380-31fbbe21-fa4e-4614-a24d-a126ec4ae050](https://github.com/codedbykishore/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139122/2d617ee3-ca6a-465e-a38f-a471326bd4fb)


### Values count in left column:
![318633530-cf6320ac-df7f-48d7-bfc5-bbee674598f1](https://github.com/codedbykishore/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139122/edc068bd-1b23-4542-9fc2-21706170faa1)


### Dataset transformed head:
![318633660-5bcbbc7b-4e7f-41dd-82c0-3e59250ea9cd](https://github.com/codedbykishore/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139122/87b3c6a2-39d5-4d7b-899f-346a66b9976f)


### x.head:
![318633777-f9e0a1bf-e3d8-4c08-bba0-b9192176d3b1](https://github.com/codedbykishore/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139122/4ba2a006-4cbc-4fe0-acfa-6267897d3203)


### Accuracy:
![318633832-4563d8f3-9c78-4aa7-a99c-f0c5cdd39e73](https://github.com/codedbykishore/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139122/8c3b22d7-9c34-4528-8a5f-8c358ec1e129)

### Data Prediction:
![318634020-0be83ccc-14db-484a-a57c-6af6cef509e7](https://github.com/codedbykishore/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139122/10f526e1-7e91-4747-8ac0-0d9cdf596859)











## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
