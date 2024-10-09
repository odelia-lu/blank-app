import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image  #for displaying images
import seaborn as sns  # in terminal, type: pip install seaborn

import codecs 
import streamlit.components.v1 as components

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics  # pip install scikit-learn

st.title("Data Science App about Wine")
df = pd.read_csv("wine_quality_red.csv")

image_path = Image.open("Wine.jpg")
st.image(image_path, width = 600)

# creating pages
app_page = st.sidebar.selectbox("Select Page", ['Data Exploration', 'Visualization', 'Prediction'])

if app_page == 'Data Exploration':

    st.dataframe(df.head())

    st.subheader("01 Description of the Dataset")
    st.dataframe(df.describe())

    st.subheader("Missing Values")
    dfnull = df.isnull()/len(df)*100  #gives percentage of rows that are blank in full data set
    total_missing = dfnull.sum().round(2)
    st.write(total_missing)

    if total_missing[0] == 0.0:
        st.success("Congrats, you have no missing values")

    if st.button("Generate Report"):

        # function to load html file
        def read_html_reporting(file_path):
            with codecs.open(file_path, 'r', encoding = "utf-8") as f:
                return f.read()
            
        # inputting the file path
        html_report = read_html_reporting("report.html")

        # displaying the file
        st.title("Streamlit Quality Report")
        st.components.v1.html(html_report, height = 1000, scrolling = True)

if app_page == 'Visualization':

    st.subheader("02 Data Visualization")
    list_columns = df.columns
    values = st.multiselect("Select two variables: ", list_columns, ['quality', 'citric acid'])

    # creation of the line chart
    st.line_chart(df, x = values[0], y = values[1])

    # creation of the bar chart
    st.bar_chart(df, x = values[0], y = values[1])

    # creating pairplot
    values_pp = st.multiselect("Select four variables: ", list_columns, ['quality', 'citric acid', 'alcohol', 'chlorides'])
    df2 = df[[values_pp[0], values_pp[1], values_pp[2], values_pp[3]]]
    pair = sns.pairplot(df2)
    st.pyplot(pair)

if app_page == "Prediction":

    st.title("03 Prediction")
    list_columns = df.columns
    input_lr = st.multiselect("Select variables:", list_columns, ["quality", "citric acid"])

    df2 = df[input_lr]

    # step 1 - splitting dataset into X and y
    X = df2
    y = df["alcohol"]  # target variable

    # step 2 - splitting into 4 chunks (X_train, X_test, y_train, y_test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    # step 3 - initialize linear regression
    lr = LinearRegression()

    # step 4 - train model
    lr.fit(X_train, y_train)

    # step 5 - prediction
    predictions = lr.predict(X_test)

    # step 6 - evaluation
    mae = metrics.mean_absolute_error(predictions, y_test)
    r2 = metrics.r2_score(predictions, y_test)

    st.write("Mean Absolute Error:", mae)
    st.write("R2 output:", r2)


# type the following in the terminal:
# git status
# git add .
# git commit -m"changes in my code"
# git push  <-- now code has been pushed onto github


