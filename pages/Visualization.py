import streamlit as st 
import pandas as pd
import numpy as np
import os
from matplotlib import image
import plotly.express as px

file_dir=os.path.dirname(os.path.abspath(__file__))
parent_dir=os.path.join(file_dir,os.pardir)
dir_of_interest=os.path.join(parent_dir,"resources")
data_path=os.path.join(dir_of_interest,"data","Titanic.csv")
data=pd.read_csv(data_path)

#Pie Chart
var=st.selectbox("Pie Chart",("Survived","Class","Sex","Embarked"))
df=data[var].value_counts()
fig = px.pie(values=df.values, names=df.index)
st.plotly_chart(fig)
    
#Histogram Chart
var=st.selectbox("Histogram Chart",("Age","No_of_siblings","No_of_parents","Fare"))
fig = px.histogram(data, x=var)
st.plotly_chart(fig)
    
#Bar Plot
var=st.selectbox("Bar Chart",("Class","Sex","Embarked"))
df=data.groupby([var,"Survived"])[["Age"]].count().reset_index()
fig = px.bar(x=df[var], y=df["Age"], color=df["Survived"])
st.plotly_chart(fig)

col1,col2=st.columns(2)
var1,var2="Class","Age"
with col1:
    var1=st.selectbox("Box Plot",("Class","Sex","Embarked"))
with col2:
    var2=st.selectbox("Box Plot",("Age","Fare"))

#Box Plot
fig = px.box(data, x=var1, y=var2, color="Survived")
st.plotly_chart(fig)

#Tree Map
df=data.groupby(["Embarked","Class","Sex","Survived"])[["Fare"]].count().reset_index()
vars = st.multiselect('Choose Variable',["Embarked","Class","Sex","Survived"])
fig=px.treemap(df,path=vars,values='Fare')
st.plotly_chart(fig)

    
