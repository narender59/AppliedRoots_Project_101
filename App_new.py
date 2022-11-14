import streamlit as st
import numpy as np
import pandas as pd
# pip install matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")
import pickle
import joblib

def no_failure():
    txt="The machine is safe!"
    htmlstr1=f"""<p style='background-color:green;
                                           color:white;
                                           font-size:18px;
                                           border-radius:3px;
                                           line-height:60px;
                                           padding-left:17px;
                                           opacity:0.6'>
                                           {txt}</style>
                                           <br></p>""" 
    st.markdown(htmlstr1,unsafe_allow_html=True)
    

def failure():
    txt="The machine may fail"
    htmlstr1=f"""<p style='background-color:red;
                                           color:white;
                                           font-size:18px;
                                           border-radius:3px;
                                           line-height:60px;
                                           padding-left:17px;
                                           opacity:0.6'>
                                           {txt}</style>
                                           <br></p>""" 
    st.markdown(htmlstr1,unsafe_allow_html=True)
    
    
st.title('Machine Failure Prediction')

loaded_pipeline = joblib.load('encoder_pipeline.joblib')

st.header('Provide the data here: ')
    
# UDI = st.number_input('Enter UDI: ', 0.0)
# PID = st.text_input('Enter Product ID: ', 'L_XXXX')
# Type = st.text_input('Enter Type of operation(Low(L) or Medium(M) or High(H)): ', 'M')
Type = st.selectbox('Select the type of operation: ',
    ('H', 'L', 'M'))
AT = st.number_input('Enter air temperature in Kelvin: ', 0.0)
PT = st.number_input('Enter process temperature in Kelvin: ', 0.0)
rot = st.number_input('Enter rotational speed in rpm: ', 0)
torque = st.number_input('Enter torque in Nm: ', 0.0)
tool_wear = st.number_input('Enter tool_wear in minutes: ', 0)

data_new = {'type' : Type, 
            'air_temp' : AT,
            'proc_temp' : PT,
            'rot_speed' : rot,
            'torque' : torque,
            'tool_wear' : tool_wear}

data_new = pd.DataFrame(data_new.items())
data_new = data_new.transpose()
data_new.columns = data_new.iloc[0]
data_new = data_new.iloc[1:]

Type = loaded_pipeline.transform(data_new.type.values.reshape(-1,1)).toarray()
# st.dataframe(Type)
data_new[['H', 'L', 'M']] = Type

# st.dataframe(data_new)
data_new.drop(['H', 'type'], axis = 1, inplace = True)

# st.dataframe(data_new)

data_new['diff_temp'] = abs(data_new['air_temp'] - data_new['proc_temp'])
data_new['power'] = data_new['rot_speed'] * data_new['torque']
data_new['heat'] = data_new['diff_temp'] * data_new['rot_speed']
data_new['strain'] = data_new['tool_wear'] * data_new['torque']

# st.dataframe(data_new)

loaded_scaler = pickle.load(open('scaler.pkl', 'rb'))

data_new[['air_temp', 
          'proc_temp',
          'rot_speed',
          'torque',
          'tool_wear',
          'diff_temp',
          'power', 'heat',
          'strain']] = loaded_scaler.transform(data_new[['air_temp','proc_temp',
                                                         'rot_speed','torque',
                                                         'tool_wear','diff_temp',
                                                         'power','heat','strain']])

data_new.drop(['torque', 'air_temp', 'tool_wear', 'rot_speed'], axis = 1, inplace = True)

data_new = data_new[['L', 'M', 'proc_temp', 'diff_temp','power','heat','strain']]

# st.dataframe(data_new)
binary_model = pickle.load(open('BinaryClassifier_SVC.sav', 'rb'))
multilabel_model = pickle.load(open('multilabel_xgb.sav','rb'))

button_clicked = st.button("Predict")

if button_clicked and AT and PT and rot and torque:
    y = binary_model.predict(data_new)
    st.header('The result is: ')
    if y:
        failure()
#         st.subheader('The machine is going to fail!')
        y_multi = multilabel_model.predict(data_new)
        [TWF, HDF, PWF, OSF] = y_multi[0]
        st.write('Possible types of Failures are:' )
        if TWF:
            st.write('Tool Wear Failure')
        if HDF:
            st.write('Heat Decepation Failure')
        if PWF:
            st.write('Power Failure')
        if OSF:
            st.write('Over Strain Failure')
        if (y_multi[0].sum() == 0):
                st.write('Random Failure')
    else:
        no_failure()
#         st.write('-----The machine is SAFE!-----')
