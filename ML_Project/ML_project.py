import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import pickle
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection as model_selection
import streamlit as st



data = pd.read_csv("healthcare-dataset-stroke-data.csv")
data = data.dropna()


data.drop(['ever_married','work_type','id'], axis='columns', inplace=True)
inputs = data.drop('stroke', axis='columns')
target = data['stroke']
from sklearn.preprocessing import LabelEncoder
lab_gender = LabelEncoder()
lab_Residence_type = LabelEncoder()
lab_smoking_status = LabelEncoder()
inputs['n_gender'] = lab_gender.fit_transform(inputs['gender'])
inputs['n_Residence_type'] = lab_Residence_type.fit_transform(inputs['Residence_type'])
inputs['n_smoking_status'] = lab_smoking_status.fit_transform(inputs['smoking_status'])
new_inputs = inputs.drop(['gender', 'Residence_type', 'smoking_status'], axis='columns')
new_inputs.head()


X_train,X_test,y_train,y_test = model_selection.train_test_split(new_inputs,target,test_size=0.2,random_state=4)   # ملكش دعوة بده ده علي حسب البروجيكت بتاعك من او هنا لحد فوق


classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm
global accuracy
accuracy = accuracy_score(y_test, y_pred)
# print("confusion matrix accuracy: ",accuracy)



dt_classifier = DecisionTreeClassifier()

# Train the classifier
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = dt_classifier.predict(X_test)

# Evaluate accuracy

accuracy2 = accuracy_score(y_test, predictions)
# print("Decision Tree classification accuracy:", accuracy2)



knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

predict = knn.predict(X_test)

accuracy3 = accuracy_score(y_test, predict)
# print("KNeighbors classification Accuracy:", accuracy3)

nb = GaussianNB()

# Train the classifier
nb.fit(X_train, y_train)

# Make predictions on the test set
predictions = nb.predict(X_test)

# Evaluate accuracy
accuracy4 = accuracy_score(y_test, predictions)
# print("Naive Bayes classification accuracy:", accuracy4)



# ----------------------------------------------------
import joblib
file = 'stroke'
joblib.dump( classifier,"stroke")
model = joblib.load(open("stroke","rb"))
# ----------------------------------------------------
