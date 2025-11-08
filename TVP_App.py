# Import necessary libraries

import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set up the title and description of the app
st.title('Traffic Volume Predictor') 
st.write("Utilize our advanced Machine Learning Application to predict traffic volume.")

# Display an image
st.image('traffic_image.gif', width = 600)

# Load the pre-trained model from the pickle file
# Get path relative to current script
BASE_DIR = os.path.dirname(__file__)
pickle_path = os.path.join(BASE_DIR, "traffic_volume.pickle")
with open(pickle_path, "rb") as XGB_pickle:
    XGb_reg = pickle.load(XGB_pickle)



# Create a sidebar for input collection
with st.sidebar:
    st.image('traffic_sidebar.jpg')
    st.header('**Input Features**')
    st.write('You can either upload your data file or manually enter input features.')

with st.sidebar.expander("Option 1: Upload CSV File"):
    traffic_file = st.file_uploader('Upload CSV')
    sample_csv = pd.read_csv("traffic_data_user.csv") 
    st.write("Example Data Format")
    st.dataframe(sample_csv)
    st.warning("Ensure your uploaded file has the same column names and data types as shown above.")


with st.sidebar.expander("Option 2: Fill Out Form"):
    st.write("Enter the data manually using the form below.")
    default_df = pd.read_csv('Traffic_Volume.csv')
    #default_df = default_df.dropna().reset_index(drop = True) 
    # Sidebar input fields for input features
    holiday = st.selectbox('Choose whether today is a designated holiday or not', options = ["None", "Columbus Day", "Veterans Day", "Thanksgiving Day", "Christmas Day", "New Years Day", "Washingtons Birthday", "Memorial Day", "Independence Day", "State Fair", "Labor Day", "Martin Luther King Jr Day"], help="Whether the day is a holiday or not")
    temp = st.number_input('Average temperature in Kelvin', help="Temperature in Kelvin")
    rain_1h = st.number_input('Rainfall in last 1 hour (mm)', help="Rainfall in last 1 hour (mm)")
    snow_1h = st.number_input('Snowfall in last 1 hour (mm)', help="Snowfall in last 1 hour (mm)")
    clouds_all = st.number_input('Percentage of cloud cover', help="Cloudiness (0-100%)")
    weather_main = st.selectbox('Choose the current weather', options = default_df['weather_main'].unique(), help="Current weather condition")
    date = st.date_input('Date', help="Date of interest")
    time = st.time_input('Time', help="Time of interest")
    predict_button = st.button("Submit Form Data")

    date_time = datetime.combine(date, time)

    # convert Holiday to NaN if "none" was selected
    if holiday == 'None':
        holiday = np.nan

    # convert date_time to proper inputs for XGBoost
    hour = date_time.hour
    dayofweek = date_time.weekday()
    month = date_time.month


# Display prompt to input data if no data has been submitted yet
if traffic_file is None and predict_button == False:
        st.info("Please enter data using one of the methods in the sidebar.")

# Confidence level slider
alpha_input = st.slider("Confidence Level for Prediction Interval", min_value=0.01, max_value=0.5, value=0.1, step=0.01)



# If There Is A CSV
if traffic_file is not None:
    # Loading data
    user_df = pd.read_csv(traffic_file) # User provided data
    original_df = pd.read_csv('Traffic_Volume.csv') # Original data used to create ML model

    # Dropping null values
    #user_df = user_df.dropna().reset_index(drop = True) 
    #original_df = original_df.dropna().reset_index(drop = True)

    # Remove output (price) column from original data
    original_df = original_df.drop(columns = ['traffic_volume'])
    # Remove year column from user data
    #user_df = user_df.drop(columns = ['year'])


    # fix datetime format in original df
    original_df['date_time'] = pd.to_datetime(original_df['date_time'])
    original_df['hour'] = original_df['date_time'].dt.hour
    original_df['dayofweek'] = original_df['date_time'].dt.dayofweek
    original_df['month'] = original_df['date_time'].dt.month 
    original_df.drop(columns=['date_time'], inplace=True)

    # rename to match model training format
    user_df.rename(columns={'weekday': 'dayofweek'}, inplace=True)

    # change day and month names to numbers
    user_df['month'] = pd.to_datetime(user_df['month'], format='%B').dt.month
    user_df['dayofweek'] = pd.to_datetime(user_df['dayofweek'], format='%A').dt.dayofweek
    
    # Ensure the order of columns in user data is in the same order as that of original data
    user_df = user_df[original_df.columns]# display dataframe head for checking


    # data previews for debugging
    # st.write("### OG Data Preview")
    # st.dataframe(original_df.head())
    # st.write("### Uploaded Data Preview")
    # st.dataframe(user_df.head())
    
    # fix datetime format in usr df [not needed]
    # user_df['date_time'] = pd.to_datetime(user_df['date_time'])
    # user_df['hour'] = user_df['date_time'].dt.hour
    # user_df['dayofweek'] = user_df['date_time'].dt.dayofweek
    # user_df['month'] = user_df['date_time'].dt.month
    # user_df.drop(columns=['date_time'], inplace=True)


    # Concatenate two dataframes together along rows (axis = 0)
    combined_df = pd.concat([original_df, user_df], axis = 0)

    # Number of rows in original dataframe
    original_rows = original_df.shape[0]

    # Create dummies for the combined dataframe
    combined_df_encoded = pd.get_dummies(combined_df)

    # Split data into original and user dataframes using row index
    original_df_encoded = combined_df_encoded[:original_rows]
    user_df_encoded = combined_df_encoded[original_rows:]

    # Predictions for user data
    user_pred, prediction_interval = XGb_reg.predict(user_df_encoded, alpha=alpha_input)


    # Predicted volume
    user_pred_TrafficVolume = user_pred

    # Adding predicted volume to user dataframe
    user_df['Predicted Traffic Volume'] = user_pred_TrafficVolume

    # Adding prediction intervals to user dataframe
    user_df['Prediction Interval Lower Bound'] = prediction_interval[:, 0]
    user_df['Prediction Interval Upper Bound'] = prediction_interval[:, 1]
    user_df['Alpha'] = alpha_input


    # Show the predicted volume on the app
    st.subheader("Predicted Traffic Volume from Uploaded CSV:")
    st.dataframe(user_df)


# If No CSV ...

if predict_button == True:
    # Encode the inputs for model prediction
    encode_df = default_df.copy()
    encode_df = encode_df.drop(columns = ['traffic_volume'])

    # fix datetime format in encode df
    encode_df['date_time'] = pd.to_datetime(encode_df['date_time'])
    encode_df['hour'] = encode_df['date_time'].dt.hour
    encode_df['dayofweek'] = encode_df['date_time'].dt.dayofweek
    encode_df['month'] = encode_df['date_time'].dt.month
    encode_df.drop(columns=['date_time'], inplace=True)




    # Combine the list of user inputed data as a row to default_df
    encode_df.loc[len(encode_df)] = [holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, hour, dayofweek, month]

    # Create dummies for encode_df
    encode_dummy_df = pd.get_dummies(encode_df)

    # Extract encoded user data
    user_encoded_df = encode_dummy_df.tail(1)

    # Using predict() with new data provided by the user
    new_prediction, prediction_interval = XGb_reg.predict(user_encoded_df, alpha=alpha_input)


    # Show the predicted price on the app
    st.subheader("Predicted Traffic Volume from Form Data:")
    st.subheader(round(new_prediction[0]))

    # chatgpt help for formatting this:
    st.write(f"Prediction Interval ({(1 - alpha_input) * 100:.0f}%): "
                    f"[{float(prediction_interval[0][0]):.0f}, {float(prediction_interval[0][1]):.0f}]")

# st.subheader("Predicting Traffic Volume")
# st.success('**We predict your traffic volume to be {} vehicles**'.format(round(new_prediction[0])) + " with a confidence level of {}%".format((1 - alpha_input) * 100))
#st.write(f"Prediction Interval: {prediction_interval} (α = {alpha_input})")





# display model information even if no prediction has been made yet
# Showing additional items in tabs
st.subheader("Model Performance and Inference")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Residuals Histogram", "Predicited vs Actual", "Coverage Plot"])

# Tab 1: Feature Importance Visualization
with tab1:
    st.write("### Feature Importance")
    st.image('feature_imp_XGB.svg')

# Tab 2: Residuals
with tab2:
    st.write("### Residuals Histogram")
    st.image('residuals_XGB.png')

# Tab 3: Scatter Plot
with tab3:
    st.write("### Predicted vs Actual Scatter Plot")
    st.image('scatter_XGB.png')
# Tab 4: Coverage Plot
with tab4:
    st.write("Coverage Plot")
    st.image('coverage_XGB.png')
