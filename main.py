# ------------------------------ Streamlit MyWebsite Project 02 ------------------------------
#Project_Date : 04.08.2022
#Project Purpose : Make a website with streamlit dictionary
#Project Process :
    #0 -  Install and Import Packages (streamlit, Pillow, streamlit_lottie)
    #1 - Set Page Configrations title, icon, layout
    #2 - Use Local Css to config. Website (create style directory)
    #3 - Load Assets (lottie, images) (Create images directory and put the images here)
    # 4 - Header Section01
    # 5- What I do Section02
    # 6 - Projects
        # 6.1 - Project01 Section03
        # 6.2 - Project02 Section04
    # 7 - DataFrame Read
    # 8 - Slider, Selection Box, Text Input and ML Model
        # 8.1 - Read Taxi DataFrama
        # 8.2 - Create slider, selectbox, text_input
        # 8.3 - ML Model RandomForestRegressor
        # 8.4 - Errors and R-Squared
    # 9 - Contact Form (Send Mail to You)
# ------------------------------ ------------------------------ ------------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image


# 8.20 - Not to Run always Data Frame , It saves Data and not reload if DataFrame not changed
@st.cache
def get_data(filename):
    df = pd.read_csv(filename)
    return df

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

path_main = '/Users/macbook/Desktop/projects/Github_Repositories/My Streamlit Projects/01 - My First Project/'

# 1 - Set Page Configrations
st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")
# 1.1 - Set Background Color
st.markdown(
    '''
    <style>
     .main {
     background-color: #F5F5F5;
     }
    </style>
    ''',
    unsafe_allow_html=True)

# 2 - Use Local Css to config. Website
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}<style>", unsafe_allow_html=True)

path = f'{path_main}style/style.css'
local_css(path)

# 3 - Load Assets (lottie, images)
lottie_coding =load_lottieurl('https://assets10.lottiefiles.com/packages/lf20_0yfsb3a1.json')
img01_contact_form = Image.open(f'{path_main}images/coding01.jpg')
img02_contact_form = Image.open((f'{path_main}images/coding02.jpg'))


# 4 - Headers Section01
st.subheader("Hi, I am Abdullah Cay :wave:")
st.title("A Data Analyst from Germany")
st.write("I am passionate about finding ways to use Python ")
st.write("[Learn More >](https://github.com/abdullahcayde)")


# 5- What I do Section02
st.write('---')
with st.container():
    left_column, right_column = st.columns(2)

    with left_column:
        st.header('What I do ?')
        st.write('##')
        st.write(
            '''
            I am learning everyday Coding with Python : 
            - I am on a new project because of that i have to learn streamlit.
            - I have to learn joblib.
            - I have to learn lightgm.
            - I have to learn ....
            ''')
        st.write('[Learn more>](https://github.com/abdullahcayde)')

        with right_column:
            st_lottie(lottie_coding, height=300, key='coding')

# 6 - Projects
# 6.1 - Project01 - Pandas Visualization - Section03
with st.container():
    st.write('---')
    st.header('Pandas Visualization')
    st.write('##')
    image_column , text_column = st.columns((1,2))

    with image_column:
        st.image(img01_contact_form)
    with text_column:
        st.subheader('Learn How to visualize your Data with Pandas')
        st.write(
            '''
            Learn how to use Lottie Files in Streamlit !!!!!
        ''')
        st.markdown("[Github Link ...](https://github.com/abdullahcayde/Trainings/blob/main/Pandas_Visualization/Pandas_Visualization_05.ipynb)")

# 6.2 - Project02 - Statistics for Machine Learning - Section04
with st.container():
    st.write('---')
    st.header('Statistics for Machine Learning')
    st.write('##')
    image_column, text_column = st.columns((1, 2))

    with image_column:
        st.image(img02_contact_form)
    with text_column:
        st.subheader('Basic Hypothesis Testing, A/B Testing, Varieance Testing ... ')
        st.write(
            '''
            Learn how to use Lottie Files in Streamlit !!!!!
        ''')
        st.markdown("[Github Link ...](https://github.com/abdullahcayde/Trainings/tree/main/istatistik)")

# 7 - DataFrame Read
path = f'{path_main}data'
name = 'ebay09_jazz'
use_cols =[1,2,3,4,5,6]

with st.container():
    st.write('---')
    st.header('Data Frame - Pandas')

    df_jazz = pd.read_csv(f'{path}/{name}.csv', usecols= use_cols)
    st.subheader('Data Frame- Jazz Head')
    st.write(df_jazz.head())

    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader('Data Frame- Jazz Describe')
        st.write(df_jazz.describe())
    with right_column:
        st.subheader('Year - Bar Chart')
        st.bar_chart(pd.DataFrame(df_jazz['Year'].value_counts()))
        st.markdown('* **first feature:** I want see which Model Year Distribution')


# 8 - Slider, Selection Box, Text Input and ML Model
with st.container():
    st.write('---')
    st.title('Machine Learning Model')

    set_col, disp_col = st.columns(2)

    # 8.1 - Read Taxi DataFrama
    set_col.subheader('Data Frame -TAXI Head')
    df = get_data(f'{path_main}data/taxis.csv')
    set_col.write(df.head())

    # 8.2 - Create slider, selectbox, text_input
    max_depth = set_col.slider('What should be the max_depth of the model ?', min_value=1, max_value=36, step=1)
    n_estimators = set_col.selectbox('What shoul be the n_estimators ?', options= [100, 200, 300, 'No Limit'], index= 0 )
    input_feature = set_col.text_input('Select your Feature for the prediction of "Total :"', 'distance',placeholder='Feature')

    if input_feature == 'No Limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    set_col.text('Here is the list of features in my data :')
    set_col.table(df.columns)

    disp_col.subheader('Data Frame - TAXI Describe ')
    disp_col.write(df.describe().T)

    # 8.3 - ML Model RandomForestRegressor
    #if input_feature.dtype() != (float, int64
    regr = RandomForestRegressor(max_depth= max_depth, n_estimators= n_estimators)

    X = df[[input_feature]]
    y = df['total']

    regr.fit(X, y)
    prediction = regr.predict(X)

    # 8.4 - Errors and R-Squared
    disp_col.subheader('Mean Absolute Error is:')
    mae = disp_col.write(mean_absolute_error(y, prediction))


    disp_col.subheader('Mean Squared Error is:')
    mse = disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader('R Squared Score of the Model is:')
    r_sqr = disp_col.write(r2_score(y, prediction))


# 9 - Contact Form (Send Mail to You)
with st.container():
    st.write('--')
    st.header('Get In Touch With Me!')
    st.write('##')

    # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
    contact_form = '''
    <form action="https://formsubmit.co/abdullahcaysa@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    '''
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()







