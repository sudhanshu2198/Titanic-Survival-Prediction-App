import streamlit as st 
import pandas as pd
import numpy as np
import os
from matplotlib import image

st.title("Titanic Survival Prediction")
st.subheader("Sudhanshu Rastogi")
col1,col2=st.columns(2)
with col1:
    github="https://github.com/sudhanshu2198"
    var1=st.write("Github Profile Link: {}".format(github))
with col2:
    kaggle="https://www.kaggle.com/sudhanshu2198"
    var1=st.write("Kaggle Link: {}".format(kaggle))

file_dir=os.path.dirname(os.path.abspath(__file__))
dir_of_interest = os.path.join(file_dir, "resources")
image_path= os.path.join(dir_of_interest, "images", "Titanic.jpg")

img = image.imread(image_path)
st.image(img)

st.write("On April 15, 1912, during her maiden voyage, the widely considered unsinkable"
         " RMS Titanic sank after colliding with an iceberg. Unfortunately, there were not"
         " enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.")
    


