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

model = RandomForestClassifier()
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=2022, stratify=y)
model.fit(X_train, y_train)

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
    st.header('Titanic Passenger  Visualization')
    
    st.subheader('Survival Probability by Gender Visualization')
    fig, ax = plt.subplots()
    sns.barplot(x='Sex', y='Survived', data=train_df, ci=None, palette='rocket', ax=ax)
    ax.set_ylabel('Survival Probability')
    ax.set_title('Survival Probability by Gender')
    ax.set_xticklabels(['Men', 'Women'])
    st.pyplot(fig)

    st.subheader('Survival Probability by Passenger Class Visualization')
    fig, ax = plt.subplots()
    sns.barplot(x='Pclass', y='Survived', data=train_df, ci=None, palette='mako', ax=ax)
    ax.set_ylabel('Survival Probability')
    ax.set_xlabel('Passenger Class')
    ax.set_title('Survival Probability by Passenger Class')
    st.pyplot(fig)

    st.subheader('Survival Probability by Passenger Class + Gender Visualization')
    fig, ax = plt.subplots()
    sns.barplot(x = 'Pclass', y ='Survived',hue= 'Sex', data = train_df, ci = None, palette = "crest", ax=ax)
    ax.set_ylabel('Survival Probability')
    ax.set_xlabel('Passenger Clas')
    ax.set_title('Survival Probability by Pclass + Gender')
    ax.set_xticklabels(['Class 1', 'Class 2', 'Class 3'])
    ax.legend(title='Gender', labels=['Men', 'Women'])
    st.pyplot(fig)

    st.subheader('Correlation Matrix for Survive Probability based on Sibsp, Parch, Age, and Fare')
    fig, ax = plt.subplots()
    sns.heatmap(train_df[['Survived', 'SibSp', 'Parch', 'Age', 'Fare']].corr(), annot = True, fmt = '.2f', cmap = 'flare', ax = ax)
    st.pyplot(fig)

    st.subheader('Survival Probability by Title Visualization')
    fig, ax = plt.subplots()
    sns.barplot(x = 'Title', y ='Survived', data = train_df, palette = 'rocket', ci = None, ax = ax)
    ax.set_ylabel('Survival Probability')
    ax.set_xticklabels(['Mr', 'Miss',' Mrs', 'Master', 'Other'])
    ax.set_title('Survival Probability by Title')
    st.pyplot(fig)

with tabs2:
    st.header('User Input Features')
    new_data = {}

    class_mapping = {
        '1st Class': 1,
        '2nd Class': 2,
        '3rd Class': 3
        }

    gender_mapping = {
        'Men': 0,
        'Women': 1
    }

    embark_mapping = {
        'Cherbourg': 1, 
        'Queenstown': 2, 
        'Southampton': 0
        }

    title_mapping = {
        'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Other': 5
    }
    
    default_age = 18

    left_column, right_column = st.columns(2)
    
    with left_column:
        inputPassengerClass = st.selectbox("Input Passenger Class",['1st Class', '2nd Class', '3rd Class'])
        inputGender = st.selectbox("Input Gender",['Men', 'Women'])
        inputAge = st.number_input("Input Age", 0, 150, value=default_age)
        inputSibSp = st.number_input("Input Number of siblings / spouses aboard the Titanic",0 , 5)
    
    with right_column:
        inputParch = st.number_input("Input Number of parents / children aboard the Titanic",0 , 5)
        inputFare = st.selectbox("Input Fare",['0', '1', '2', '3', '4', '5'])
        inputEmbark = st.selectbox("Input Embark",['Cherbourg', 'Queenstown', 'Southampton'])
        inputTitle = st.selectbox("Input Title",['Mr', 'Miss', 'Mrs', 'Master', 'Other'])
    
    selected_class_number = class_mapping[inputPassengerClass]
    selected_gender = gender_mapping[inputGender]
    if inputAge <= 11:
        age_category = 0
    elif 11 < inputAge <= 18:
        age_category = 1
    elif 18 < inputAge <= 25:
        age_category = 2
    elif 25 < inputAge <= 40:
        age_category = 3
    elif 40 < inputAge <= 65:
        age_category = 4
    else:
        age_category = 5
    
    selected_embark = embark_mapping[inputEmbark]
    selected_title = title_mapping[inputTitle]
    
    new_data = [{
    'Pclass': selected_class_number,
    'Sex':selected_gender,
    'Age':age_category,
    'SibSp':inputSibSp,
    'Parch':inputParch,
    'Fare':inputFare,
    'Embarked':selected_embark,
    'Title':selected_title
    }]
    
    df_input = pd.DataFrame(new_data)
    
    prediction = model.predict(df_input)
    probability = model.predict_proba(df_input)[:, 1]
    st.subheader('Prediction Result')

    color = 'green' if prediction[0] == 1 else 'red'

    # Construct the string with Markdown and HTML
    output_text = f"## You will <span style='color:{color}'>{'Survive' if prediction[0] == 1 else 'Not Survive'}</span> with a probability of <span style='color:blue'>{probability[0]:.2%}</span>."
    
    # Display the text using Markdown
    st.markdown(output_text, unsafe_allow_html=True)
    if prediction[0] == 1:
        st.image("survive.jpeg")
    else:
        st.image("notsurvive.jpeg")
st.caption('@Valerie6048')
