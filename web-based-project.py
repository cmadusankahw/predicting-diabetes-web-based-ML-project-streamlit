#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
from PIL import Image


# # Web app initializing

# In[7]:


st.header("Diabities Detection app using Machine Learning")


# In[9]:


img=Image.open("diab.png")


# In[10]:


st.image(img)


# In[11]:


data=pd.read_csv("diabetes.csv")


# In[14]:


st.subheader("Trained Data information")


# In[15]:


st.dataframe(data)


# In[16]:


st.subheader("Trained Data Summary")


# In[18]:


st.write(data.iloc[:,:8].describe())


# # Training model

# In[19]:


x=data.iloc[:,:8].values
y=data.iloc[:,8].values


# In[20]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[21]:


model=RandomForestClassifier(n_estimators=500)


# In[22]:


model.fit(x_train,y_train)


# In[24]:


y_pred=model.predict(x_test)


# In[25]:


st.subheader("Accuracy of the trained model")


# In[26]:


st.write(accuracy_score(y_test,y_pred))


# # Getting user input to give predictions

# In[ ]:


st.header("Check your diabities level...")


# In[27]:


name = st.text_input("Enter your Name")


# In[32]:


def user_inputs():
    age=st.slider("Age",0,120,0)
    preg=st.slider("Pregnancies",0,20,0)
    glu=st.slider("Glucose",0,200,0)
    bp=st.slider("BloodPressure",0,130,0)
    sthick=st.slider("Skin Thickness",0,100,0)
    insulin=st.slider("Insulin",0.0,1000.0,0.0)
    bmi=st.slider("BMI",0.0,70.0,0.0)
    dpf=st.slider( "Diabetes Pedigree Function",0.0,5.000,0.0)
    
    input_dict={
        "Pregnancies":preg,
        "Glucose":glu,
        "BloodPressure":bp,
        "Skin Thickness":sthick,
        "Insulin":insulin,
        "BMI":bmi,
        "Diabetes Pedigree Function":dpf,
        "Age":age
    }
    
    return pd.DataFrame(input_dict,index=["Input value"])

ui=user_inputs()


# In[36]:


st.subheader("Inputs you have entered")


# In[37]:


st.write(ui)


# In[38]:


st.subheader("Predicted Results..")


# In[39]:


if (model.predict(ui)[0]== 0):
    st.write(name,", You don't have diabetes")
else:
    st.write(name, ", YOU HAVE DIABETES!!!!")


# In[ ]:




