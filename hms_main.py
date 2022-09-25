
# Heart Attack/Disease Prediction
# Team : StormBreaker
#
# Team No : 50
#
# Competition : IBM zDatathon
#
# Team Members : Harsh Jaiswal, Suraj Mishra, Kunal Mohite, Qainat Sheikh
#
# Overview:
# This program helps the people to find out if they are prone to any heart disease.The input of the application is quite basic which is usually available to everyone who has concerns over their health or if they have just performed a routine body test. This application using a form to collect such basic infomration via a website form and shows the client their results immedieatly.
#
# Method:
# Input: Age, Gender, Height, Weight, Blood pressure, Cholesterol Level, Glucose Level, Smoking, Alcohol Consumption
#
# ALgorithms Used:
# Random Forest Classification, Decission Tree, KneighbourClassfier, SGDClassifier, Linear Regression, Logistic Regression, XGBClassifier
#
# Use cases:
#
# Personal Heart Condition Monitoring.
# Can be used by General Physician to give basic advice to the patient.
# Can be used by Hospitals and Diagnostic Centers.

#Conclusion:

# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
import pickle

# Processing Data
warnings.filterwarnings("ignore")

# Loading the data csv from kaggle to a Pandas DataFrame
patient_data = pd.read_csv('D:\hms\patient_info.csv')

# test print the first 5 rows
patient_data.head()

# test print last 5 rows
patient_data.tail()

# number of rows and colums in data
patient_data.shape

# getting general information of the data
patient_data.info()

# checks for incomplete data
patient_data.isnull().sum()

# statistical measures about the data
patient_data.describe()

# checking the distribution of Cardio Variable
patient_data['cardio'].value_counts()

# Splitting the Features and Target
X = patient_data.drop(columns='cardio', axis=1)
Y = patient_data['cardio']

# Splitting the Data into Training data & Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Model Training

# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=1000)
rfc.fit(X_train,Y_train)
rfc.predict(X_test)
rfc.score(X_test,Y_test)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc= DecisionTreeClassifier()
dtc.fit(X_train,Y_train)
dtc.predict(X_test)
dtc.score(X_test,Y_test)

# Kneighborclassifier
from sklearn.neighbors import KNeighborsClassifier
knc=KNeighborsClassifier()
knc.fit(X_train,Y_train)
knc.predict(X_test)
knc.score(X_test,Y_test)

# SGDClassifier
from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier()
sgd.fit(X_train,Y_train)
sgd.predict(X_test)
sgd.score(X_test,Y_test)

# Linear Regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
lr.predict(X_test)
lr.score(X_test,Y_test)

# Logistic Regression
model = LogisticRegression()
model.fit(X_train, Y_train)
model.predict(X_test)
model.score(X_test,Y_test)

# X Gradient Boost
import xgboost as xgb
from sklearn.model_selection import cross_validate
from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
boost_model = XGBClassifier()
boost_model.fit(X_train, Y_train)
boost_pred = boost_model.predict(X_test)
boost_model.predict(X_test)
boost_model.score(X_test,Y_test)


# Model Evaluation
# Accuracy Score

# Accuracy on training data
X_train_prediction = boost_model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)

#Average Accuracy on Testing Data
print(f"Testing accuracy: {round((accuracy_score(boost_pred, Y_test)*100),2)}%")
xgb_cross = cross_validate(boost_model, X, Y, cv=11)
print(f"Average testing accuracy: {round((xgb_cross['test_score'].mean()*100),4)}%")

# Accuracy on test data
X_test_prediction = boost_model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data : ', test_data_accuracy)

pickle.dump(boost_model, open('check_health.pkl', 'wb'))
final_model = pickle.load(open('check_health.pkl', 'rb'))


### -------------------------------------------- ###

# input_data = (64,16045,1,170,69,120,70,1,1,0,0,1)
#
# # change the input data to a numpy array
# input_data_as_numpy_array= np.asarray(input_data)
#
# # reshape the numpy array as we are predicting for only on instance
# input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
#
# prediction = boost_model.predict(input_data_reshaped)
# print(prediction)
#
# if (prediction[0]== 0):
#   print('The Person does not have a Heart Disease')
# else:
#   print('The Person has Heart Disease')



