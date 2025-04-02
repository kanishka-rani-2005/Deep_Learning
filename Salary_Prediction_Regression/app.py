import streamlit as st
import numpy as np;
import tensorflow as tf;
import pandas as pd;
import pickle;
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder;

from tensorflow import keras

model = tf.keras.models.load_model('regression_mode.h5')


with open ('gender.pkl','rb') as f:
    le_gender = pickle.load(f)

with open ('geo.pkl','rb') as f:
    ohe_geo = pickle.load(f)

with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)


## Streamlit app

st.title('Customer Salary Prediction')
st.write('This app predicts the Salary of a customer based on their demographic and geographic information')



## User Input

geo=st.selectbox('Geography',ohe_geo.categories_[0])
gender=st.selectbox('Gender',le_gender.classes_)
age=st.slider('Age',min_value=18,max_value=99)
balance=st.number_input('Balance',min_value=0,max_value=100000000)
credit=st.number_input('Credit Score')
exited=st.selectbox('exited',[0,1])
tenure=st.number_input('Tenure',0,10)
num_of_prod=st.number_input('Number of Products',1,4)
has_cr_card=int(st.selectbox('Credit Card',[0,1]))
is_active_member=int(st.selectbox('Active Or Not',[0,1]))

## Prepare input data
input_data=pd.DataFrame({
    'CreditScore': [credit],
    'Gender': [le_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_prod],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]

})

## One hot encoding
coder = ohe_geo.transform(pd.DataFrame([geo], columns=['Geography']))
df=pd.DataFrame(coder,columns=ohe_geo.get_feature_names_out(['Geography']))

## Combine 
input_data=pd.concat([input_data.reset_index(drop=True),df],axis=1)

## scale the data

print("Expected feature names:", scaler.feature_names_in_)

print("Input data columns:", input_data.columns)


input_data_scaled=scaler.transform(input_data)



## make prediction

prediction=model.predict(input_data_scaled)

prob=prediction[0][0]

st.write("The Estimated Salary  of customer  is: ",prob)

