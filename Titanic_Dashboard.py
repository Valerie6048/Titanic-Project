import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import plotly.express as px
import pickle

st.set_option('deprecation.showPyplotGlobalUse', False)

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

df_training = pd.read_csv(r'Train_Titanic.csv')

# Allow user input for new data
new_data = {}  # Add input fields for new data

with st.sidebar:
    st.image('OIG.jpeg')
    st.title('Titanic Survival Prediction')
    st.caption('Valerie6048')

icon("🛳️")
"""
# Titanic Survival Prediction Project
## Overview
In this final project, I aim to make a Data Visualization and Prediction for Titanic Passenger Survival Probability
"""

st.subheader('Titanic Passanger Dataset and Visualization')
st.write(df_training)

st.caption('Pengpol Kelompok 8 2023')
