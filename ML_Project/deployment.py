import streamlit as st
import requests
import pandas as pd 
from streamlit_option_menu import option_menu
import joblib
import numpy as np 




st.set_page_config (
    page_title = "stroke prediction",
    page_icon = ":gem",
    initial_sidebar_state = 'collapsed'
)

def load_lottie(url):
    r=requests.get(url)
    if r.status_code !=200:
      return None
    return r.json()


model = joblib.load(open("stroke","rb"))

from ML_project import classifier , dt_classifier , rf_classifier , nb
def predict(age ,  hypertension , heart_disease , avg_glucose_level , bmi ,  gender_encoder ,  Residence_type_encoder ,  smoking_status_encoder):
    features = np.array([age ,  hypertension , heart_disease , avg_glucose_level , bmi ,  gender_encoder ,  Residence_type_encoder ,  smoking_status_encoder]).reshape(1, -1)

    prediction_cm = classifier.predict(features)
    # Prediction using decision tree
    prediction_dt = dt_classifier.predict(features)
    
    # Prediction using KNN
    prediction_rf = rf_classifier.predict(features)
    
    # Prediction using Naive Bayes
    prediction_nb = nb.predict(features)
    
    # Return all predictions
    return  prediction_dt, prediction_rf, prediction_nb , prediction_cm


with st.sidebar:
    choose = option_menu(None, ["Home", "Graphs", "About", "Contact"],
                           icons=['house', 'kanban', 'book','person lines fill'],
                           menu_icon="app-indicator",  default_index=0,
                           styles={ 
        "container": {"padding": "5!important", "background-color":  "#fafafa" },
        "icon": {"color": '#E0E0EF', "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px" , "--hover-color":"#eee" },
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

from ML_project import accuracy , accuracy2 , accuracy3 , accuracy4

if choose =='Home':
        st.write('# Stroke prediction')
        st.subheader('Enter your information th predict are you have a stroke or no')
          # user input
        age = st.number_input("enter your age: ",min_value = 0)
        hypertension = st.radio('select 1 if you have hypertension / 0 if not',(0,1))
        heart_disease = st.radio('select 1 if you have heart disease / 0 if not',(0,1))
        avg_glucose_level = st.number_input("enter your average glucose level: ",min_value = 0)
        bmi = st.number_input("enter your bmi: ",min_value=0)
        n_gender = st.radio('select your gender: ',('male','female'))
        gender_encoder=1 if n_gender=="male" else 0
        n_Residence_type = st.radio('select your residence type',('Urban','Rural'))
        Residence_type_encoder = 1 if n_Residence_type== "Urban" else 0
        n_smoking_status = st.radio("select your smoking status: ",('formerly smoked' , 'never smoked','smokes'))
        smoking_status_encoder = 1 if n_smoking_status == "formerly smoked" else 2 if n_smoking_status == "never smoked" else 3
        
      
        
        

          # predict
        sample_cm ,sample_dt, sample_rf, sample_nb = predict(age ,  hypertension , heart_disease , avg_glucose_level , bmi ,  gender_encoder ,  Residence_type_encoder ,  smoking_status_encoder)
          
        if st.button('predict stroke'):
          st.write("confusion matrix accuracy: ",accuracy)
          st.write("Decision Tree classification accuracy:", accuracy2)
          st.write("Random Forcet classification Accuracy:", accuracy3)
          st.write("Naive Bayes classification accuracy:", accuracy4)
          st.write(" ")
          st.write("confusion matrix prediction: ","Has a stroke"if sample_cm == 1 else "No having a stroke")
          st.write("Decision Tree prediction:", "Has a stroke" if sample_dt == 1 else "No having a stroke")
          st.write("Random Forcet prediction:", "Has a stroke" if sample_rf == 1 else "No having a stroke")
          st.write("Naive Bayes prediction:", "Has a stroke" if sample_nb == 1 else "No having a stroke")


        



elif choose == 'Graphs':
          st.write("sorry! currently not supported☺.")

elif choose == 'About':
          st.write("# about us")
          st.write("💡🎯 This project analyzes data entered for some people,")
          st.write("some of whom have a stroke and some of them do not.")
          st.write("After analyzing the data,")
          st.write("the program predicts whether the person has a stroke or not with the new data.✨🚀")


elif choose == 'Contact':
          st.write("# Contact Us")
          st.write("------")
          with st.form(key='columns_in_form2', clear_on_submit=True):
              st.write('## Please help us improve!')
              Name = st.text_input(label="Please enter your name: ")
              Email = st.text_input(label="please enter your email: ")
              Message = st.text_input(label="please enter your message: ")
              submitted = st.form_submit_button('Submit')
              if submitted:
                st.write("thank for you contacting us. We will respond to your question or inquiries as soon.")


