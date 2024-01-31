import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px

st.set_option('deprecation.showPyplotGlobalUse', False)

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

train_df = pd.read_csv(r'Train_Titanic.csv')

icon("🛳️")
"""
# Titanic Survival Prediction Project
## Overview
In this final project, I aim to make a Data Visualization and Prediction for Titanic Passenger Survival Probability
"""

st.write(train_df)

st.subheader('Titanic Passanger Dataset and Visualization')
st.write(df_training)

st.caption('Pengpol Kelompok 8 2023')
