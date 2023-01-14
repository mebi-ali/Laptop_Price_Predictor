# Libraries
import streamlit as st
import pickle
import sklearn
import numpy as np
import pandas as pd


# Load Model and Dataframe
model = pickle.load(open('./Model/pipe.pkl', 'rb'))
df = pickle.load(open('./Model/df.pkl', 'rb'))


# Title
st.title("Laptop Predictor")

# Laptop Brand
company = st.selectbox('Brand', df['Company'].unique())

# Laptop Type
typeName = st.selectbox('Type', df['TypeName'].unique())

# Laptop Ram
ram = st.selectbox('Ram (in GBs)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Laptop Weight
weight = st.number_input('Weight')

# Process Speed
pspeed = st.number_input('Process Speed')

# TouchScreen
touchscreen = st.selectbox('TouchScreen', ['Yes', 'No'])

# IPS
ips = st.selectbox('IPS', ['Yes', 'No'])

# PPI
scren_size = st.number_input('Screen Size')
resolution = st.selectbox('Screen Resolution', [
                          '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# CPU Brand
cpu = st.selectbox('CPU Brand', df['CpuBrand'].unique())

# HDD
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

# GPU Brand
gpu = st.selectbox('GPU', df['GpuBrand'].unique())

# Laptop OS
os = st.selectbox('OS', df['OpSys'].unique())


# Predictor
if st.button('Predict Price'):
    ppi = None

    if (touchscreen == 'Yes'):
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    # Calculate PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = np.sqrt(np.sqrt(X_res) + np.sqrt(Y_res))/scren_size

    # get data
    query = np.array([company, typeName, ram, os, weight,
                     touchscreen, ips, ppi, hdd, ssd, cpu, pspeed, gpu])

    # get Data format to send it to the model
    query = pd.DataFrame(query.reshape(1, 13))
    query.columns = df.columns

    st.title("The Predicted Price  Rs " +
             str(int(np.exp(model.predict(query)[0]))))
#
