# Developed by: Naveen Kumar.T
# RegisterNumber: 212223220067
# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Applynew unknown values




## Program:


Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

```

import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1) #removes the specified row or column.
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1


x=data1.iloc[:,:-1]
x

y=data1['status']
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size =0.2,random_sta

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear')# A library for large linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)


lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```


## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)

## PLACEMENT DATA:
![267522570-f40d603b-0f61-48da-b1e1-87c31d03bd70](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/154746066/83e629ef-6321-48f6-93b8-07feb9b2bf71)

## SALARY DATA:
![267525779-49f917bf-bce0-408e-8f94-d0bc79548d15](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/154746066/fd5c7b86-5ee4-4be1-9718-41721c1bdab8)


## CHECKING THE NULL() FUNCTION:]
![267522991-a22dfd91-b344-4266-91a0-2cd757a3623e](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/154746066/8a795fb6-ad90-4d55-bf0f-265f1dfba76c)


## DATA DUPLICATE:
![267523098-861c970b-ffeb-4246-af60-913e8388dea4](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/154746066/42249b41-9241-40d5-bb2c-cc58f30d2fb8)


## PRINT DATA:
![267523282-d30f3eca-3148-47df-85b8-5896fb036250](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/154746066/f7be264e-9c52-4c31-90b3-3fe785e41af1)

## DATA_STATUS:
![267523389-2f05d218-3b79-4c0e-a69b-407d74f9711d](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/154746066/b53a1c8f-d081-42e5-b062-555db2f8329b)


## Y_PREDICTION ARRAY:
![267524139-b6773f22-9c30-43b5-a3db-ca652e7b0009](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/154746066/30f26342-057f-4724-b25e-a822bdcc8064)

## ACCURACY VALUE:


![267524226-0f00fd95-00ee-483c-b333-fc18162a438e](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/154746066/d9a2b6d4-5c0b-43db-9b75-ef6e1b5626a1)

## CONFUSION ARRAY:
![267524290-54627a5d-0db9-440c-a5d5-3f4831edeb03](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/154746066/a33fe768-147a-4ba4-b9b3-11b4d3754aac)

## PREDICTION OF LR:
![267524441-b7be4b3d-54cd-4472-95cc-189b0f9248a0](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/154746066/9b13463c-e5a9-4885-934b-d7b9dda79ac7)


 ## CLASSIFICATION REPORT:
![267524355-97e25d65-6436-41c0-9a41-64cd9c457972](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/154746066/bef4b03d-48b6-4500-b76e-5ad3f56a99e3)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
