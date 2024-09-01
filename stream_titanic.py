import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# Load the dataset
data = pd.read_csv("titanic_train.csv")
dftitanic = data.copy()

# Streamlit app
st.title("Titanic Data Analysis")
st.write("### Data Overview")
st.write(dftitanic.describe())
st.write(dftitanic.head())
st.write(f"### Missing Values\n{dftitanic.isnull().sum()}")

# Replace Seaborn visualizations with Plotly

# 1. Survival Count by Gender
fig1 = px.histogram(dftitanic, x='Sex', color='Survived', barmode='group',
                    labels={'Sex': 'Gender', 'Survived': 'Survived'},
                    title='Survival Count by Gender')
st.plotly_chart(fig1)

# 2. Survival Count
fig2 = px.histogram(dftitanic, x='Survived', color='Survived',
                    labels={'Survived': 'Survived'},
                    title='Survival Count')
st.plotly_chart(fig2)

# 3. Class and Gender Distribution
fig3 = px.histogram(dftitanic, x='Pclass', color='Sex', barmode='group',
                    labels={'Pclass': 'Passenger Class', 'Sex': 'Gender'},
                    title='Class and Gender Distribution')
st.plotly_chart(fig3)

# 4. Title Extraction and Replacement
import re
dftitanic['Title'] = dftitanic['Name'].apply(lambda y: re.search(r'([A-Z][a-z]+)\.', y).group(1))
dftitanic['Title'] = dftitanic['Title'].replace({'Mme':'Mrs', 'Mlle':'Miss'})
dftitanic.loc[~dftitanic['Title'].isin(['Mr','Mrs','Miss','Master']), 'Title'] = 'Rare Title'

# 5. Title vs Survival by Class
fig4 = px.histogram(dftitanic, x='Title', color='Survived', facet_col='Pclass', barmode='group',
                    labels={'Title': 'Title', 'Survived': 'Survived', 'Pclass': 'Passenger Class'},
                    title='Title vs Survival by Class')
st.plotly_chart(fig4)

# 6. Family Size Analysis
dftitanic['Fsize'] = dftitanic['SibSp'] + dftitanic['Parch'] + 1

# 7. Family Size vs Survival
fig5 = px.histogram(dftitanic, x='Fsize', color='Survived', barmode='group',
                    labels={'Fsize': 'Family Size', 'Survived': 'Survived'},
                    title='Family Size vs Survival')
st.plotly_chart(fig5)

# 8. Group Size Calculation
dtemp = dftitanic['Ticket'].value_counts().reset_index()
dtemp.columns = ['Ticket', 'Tsize']
dftitanic = dftitanic.merge(dtemp, on='Ticket', how='inner')
dftitanic['Group'] = dftitanic[['Tsize', 'Fsize']].max(axis=1)

# 9. Group Size vs Survival by Class
fig6 = px.histogram(dftitanic, x='Group', color='Survived', facet_col='Pclass', barmode='group',
                    labels={'Group': 'Group Size', 'Survived': 'Survived', 'Pclass': 'Passenger Class'},
                    title='Group Size vs Survival by Class')
st.plotly_chart(fig6)

# 10. Fare Category Analysis
dftitanic['FareCat'] = pd.cut(dftitanic['Fare'], bins=[-1, 10, 25, 40, 70, 100, float('inf')],
                              labels=['0-10', '10-25', '25-40', '40-70', '70-100', '100+'])

# 11. Fare Category vs Survival
fig7 = px.histogram(dftitanic, x='FareCat', color='Survived', barmode='group',
                    category_orders={'FareCat': ['0-10','10-25','25-40','40-70','70-100','100+']},
                    labels={'FareCat': 'Fare Category', 'Survived': 'Survived'},
                    title='Fare Category vs Survival')
st.plotly_chart(fig7)

# 12. Age Category Analysis
dftitanic['AgeCat'] = pd.cut(dftitanic['Age'], bins=[-1, 10, 25, 40, 70, 100, float('inf')],
                             labels=['0-10', '10-25', '25-40', '40-70', '70-100', '100+'])

# 13. Age Category vs Survival
fig8 = px.histogram(dftitanic, x='AgeCat', color='Survived', barmode='group',
                    category_orders={'AgeCat': ['0-10','10-25','25-40','40-70','70-100','100+']},
                    labels={'AgeCat': 'Age Category', 'Survived': 'Survived'},
                    title='Age Category vs Survival')
st.plotly_chart(fig8)
