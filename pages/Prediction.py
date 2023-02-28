import streamlit as st 
import pandas as pd
import numpy as np
import pickle
import os

st.title("Survival Prediction")

file_dir=os.path.dirname(os.path.abspath(__file__))
parent_dir=os.path.join(file_dir,os.pardir)
dir_of_interest=os.path.join(parent_dir,"artifacts")

model_path=os.path.join(dir_of_interest,"model.pkl")
lencoder_path=os.path.join(dir_of_interest,"lencoder.pkl")

model = pickle.load(open(model_path, 'rb'))
lencoder = pickle.load(open(lencoder_path, 'rb'))

with st.form('user_inputs'):
    Embarked=st.selectbox("Point of Departure",('Cherbourg', 'Queenstown','Southampton'))
    Class=st.selectbox("Social Status",('Lower', 'Middle','Upper'))
    Sex=st.selectbox("Gender of passenger",('female','male'))
    No_of_siblings=st.number_input("No of siblings on ship",min_value=0,max_value=4,value=2,step=1)
    No_of_parents=st.number_input("No of parents on ship(include paternal and maternal )",min_value=0,max_value=4,value=2,step=1)
    Fare=st.number_input("Ticket Price",min_value=0.0,max_value=512.0,value=100.0,step=1.0)
    Age=st.number_input("Age of passenger",min_value=0.5,max_value=80.0,value=20.0,step=0.5)
    click=st.form_submit_button()

class_lower, class_middle, class_upper=0,0,0
if Class=="Lower":
   class_lower=1
elif Class=="Middle":
   class_middle=1
elif Class=="Upper":
   class_upper=1

sex_female,sex_male=0,0
if Sex=="female":
   sex_female=1
elif Sex=="male":
   sex_male=1

embark_cherbourg, embark_queenstown,embark_southampton=0,0,0
if Embarked=="Cherbourg":
   embark_cherbourg=1
elif Embarked=="Queenstown":
   embark_queenstown=1
elif Embarked=="Southampton":
   embark_southampton=1

data=[[Age, No_of_siblings, No_of_parents, Fare,class_lower, class_middle, class_upper,
      sex_female,sex_male,embark_cherbourg, embark_queenstown,embark_southampton ]]


if click:
   prediction=model.predict(data)
   prediction=lencoder.inverse_transform(prediction)[0]
   if prediction=="Survived":
      st.success(prediction)
   else:
      st.error(prediction)
