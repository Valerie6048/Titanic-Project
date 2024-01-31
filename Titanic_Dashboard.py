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
import scipy.stats as stats

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

with st.sidebar:
    st.image('Titanic.jpg')
    st.title('Biodata')
    """
    Name: Akhmad Nizar Zakaria
    
    Github: [Valerie6048](https://github.com/Valerie6048)
    
    LinkedIn: [Akhmad Nizar Zakaria](https://www.linkedin.com/in/akhmad-nizar-zakaria-8a692b229/)

    """
    st.caption('@Valerie6048')

tabs1, tabs2= st.tabs(["Data Visualization", "Prediction Result"])

with tabs1:
    st.header('Titanic Passanger Dataset and Visualization')
    st.subheader('Titanic Passanger Dataset')
    st.write(train_df)
    
    st.subheader('Survival Probability by Gender Visualization')
    fig, ax = plt.subplots()
    sns.barplot(x='Sex', y='Survived', data=train_df, ci=None, palette='rocket', ax=ax)
    ax.set_ylabel('Survival Probability')
    ax.set_title('Survival Probability by Gender')
    st.pyplot(fig)

    st.subheader('Survival Probability by Passenger Class Visualization')
    class_plot = sns.barplot(x='Pclass', y='Survived', data=train_df, ci=None, palette='mako')
    plt.ylabel('Survival Probability')
    plt.xlabel('Passenger Class')
    plt.title('Survival Probability by Passenger Class')
    st.pyplot(class_plot)

with tabs2:
    st.header("ANJAY")
    




st.caption('Pengpol Kelompok 8 2023')
