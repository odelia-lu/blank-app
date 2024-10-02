import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image  #for displaying images

st.title("Data Science App about Wine")
df = pd.read_csv("wine_quality_red.csv")

image_path = Image.open("Wine.jpg")
st.image(image_path, width = 600)

st.dataframe(df.head())

st.subheader("01 Description of the Dataset")
st.dataframe(df.describe())

st.subheader("02 Missing Values")
dfnull = df.isnull()/len(df)*100  #gives percentage of rows that are blank in full data set
total_missing = dfnull.sum().round(2)
st.write(total_missing)
if total_missing[0] == 0.0:
    st.success("Congrats, you have no missing values")

st.subheader("03 Data Visualization")
list_columns = df.columns
values = st.multiselect("Select two variables: ", list_columns, ['quality', 'citric acid'])

# creation of the line chart
st.line_chart(df, x = values[0], y = values[1])

# creation of the bar chart
st.bar_chart(df, x = values[0], y = values[1])

import seaborn as sns  # in terminal, type: pip install seaborn

# creating pairplot
values_pp = st.multiselect("Select four variables: ", list_columns, ['quality', 'citric acid', 'alcohol', 'chlorides'])
df2 = df[[values_pp[0], values_pp[1], values_pp[2], values_pp[3]]]
pair = sns.pairplot(df2)
st.pyplot(pair)

# type the following in the terminal:
# git add .
# git status