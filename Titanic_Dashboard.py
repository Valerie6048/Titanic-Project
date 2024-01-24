import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import plotly.express as px

st.set_option('deprecation.showPyplotGlobalUse', False)

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

df_training = pd.read_csv(r'Train_Titanic.csv')

y = train_df['Survived']
X = train_df.drop('Survived',axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.138, random_state=2022, stratify=y)

model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Melakukan prediksi pada data validasi
y_pred = model.predict(X_val)

# Evaluasi performa model
accuracy = accuracy_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)
class_report = classification_report(y_val, y_pred)

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
