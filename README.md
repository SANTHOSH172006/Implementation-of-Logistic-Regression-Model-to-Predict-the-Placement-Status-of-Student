# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the required packages.
2. Print the present data and placement data and salary data.
3. Using logistic regression find the predicted values of accuracy confusion matrices.
4. Display the results.

## Program:
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: GOWTHAM N

RegisterNumber: 212223100008

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Placement_Data.csv')
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:, :-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = (y_test, y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test, y_pred)
print(classification_report1)
lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]])
```

## Output:

### Placement Data
![image](https://github.com/user-attachments/assets/0e1f1875-7f5e-4df5-9d45-237610c6905c)


### Checking the null() function
![image](https://github.com/user-attachments/assets/4f9a6289-84a9-4fd2-995f-fc44ee4cda7b)


### Print Data:
![image](https://github.com/user-attachments/assets/a62bad64-9985-41db-b87d-d27c8033ef60)


### Y_prediction array
![image](https://github.com/user-attachments/assets/8bc1b1ce-389e-495c-886c-377d1bcb8c2f)


### Accuracy value
![image](https://github.com/user-attachments/assets/44a1369b-30e9-4f2e-b97b-67876f7546e4)


### Confusion array
![image](https://github.com/user-attachments/assets/326ff7fb-9954-437e-b668-9b33cd570f00)


### Classification Report
![image](https://github.com/user-attachments/assets/51303492-e868-4b8b-b578-146037b6adb1)


### Prediction of LR
![image](https://github.com/user-attachments/assets/bc0cfe7a-1580-4fa2-9627-a623137e0c38)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
