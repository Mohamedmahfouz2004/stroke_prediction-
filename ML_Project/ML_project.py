import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
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


X_train,X_test,y_train,y_test = model_selection.train_test_split(new_inputs,target,test_size=0.2,random_state=4)  

# ============================================================
# confusion matrix

classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm

accuracy = accuracy_score(y_test, y_pred)


# ==============================================================
# Desision Tree

dt_classifier = DecisionTreeClassifier()

# Train the classifier
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = dt_classifier.predict(X_test)

# Evaluate accuracy

accuracy2 = accuracy_score(y_test, predictions)


# ===============================================================
# Random Forcest

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Model evaluation
y_pred = rf_classifier.predict(X_test)
accuracy3 = accuracy_score(y_test, y_pred)


# ===============================================================
# Naive Bayes 

nb = GaussianNB()

# Train the classifier
nb.fit(X_train, y_train)

# Make predictions on the test set
predictions = nb.predict(X_test)

# Evaluate accuracy
accuracy4 = accuracy_score(y_test, predictions)




# ----------------------------------------------------
import joblib
file = 'stroke'
joblib.dump(classifier, "classifier.pkl")
joblib.dump(nb, "nb.pkl")
joblib.dump(rf_classifier, "rf_classifier.pkl")
joblib.dump(dt_classifier, "dt_classifier.pkl")
model = joblib.load(open("stroke","rb"))
# ----------------------------------------------------
