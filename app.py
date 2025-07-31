import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load the trained model and preprocessor
# Use st.cache_resource to avoid reloading on every interaction
@st.cache_resource
def load_assets():
    # Load the Keras model
    loaded_model = load_model('ann_model.h5')
    # Load the preprocessor
    with open('preprocessor.pkl', 'rb') as file:
        loaded_preprocessor = pickle.load(file)
    return loaded_model, loaded_preprocessor

model, preprocessor = load_assets()

# App title and description
st.title("Predictive Maintenance: Machine Failure Prediction")
st.markdown("Enter the machine's real-time sensor data to predict the likelihood of failure.")

# UI for user inputs in the sidebar
st.sidebar.header("Machine Sensor Readings")

def user_input_features():
    type_selection = st.sidebar.selectbox("Product Type", ['L', 'M', 'H'])
    air_temp = st.sidebar.slider("Air Temperature [K]", 295.0, 305.0, 300.1, 0.1)
    process_temp = st.sidebar.slider("Process Temperature [K]", 305.0, 315.0, 310.2, 0.1)
    rotational_speed = st.sidebar.slider("Rotational Speed [rpm]", 1100, 2900, 1530)
    torque = st.sidebar.slider("Torque [Nm]", 3.0, 80.0, 42.5, 0.1)
    tool_wear = st.sidebar.slider("Tool Wear [min]", 0, 260, 110)
    
    data = {
        'Type': type_selection,
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rotational_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear
    }
    # Create a DataFrame with a single row
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input
st.subheader("Current Input Parameters")
st.write(input_df)

# Prediction logic
if st.button("Predict Failure"):
    # Create a copy to avoid modifying the displayed dataframe
    prediction_df = input_df.copy()

    # 1. Create the same engineered features as in training
    prediction_df['Temp_Diff'] = prediction_df['Process temperature [K]'] - prediction_df['Air temperature [K]']
    prediction_df['Power'] = prediction_df['Rotational speed [rpm]'] * prediction_df['Torque [Nm]'] * (2 * np.pi / 60)
    prediction_df['Strain_Index'] = prediction_df['Tool wear [min]'] * prediction_df['Torque [Nm]']
    
    # 2. Reorder columns to match the training data order
    # The preprocessor expects columns in the exact same order it was trained on.
    training_columns_order = [
        'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
        'Torque [Nm]', 'Tool wear [min]', 'Temp_Diff', 'Power', 'Strain_Index', 'Type'
    ]
    
    prediction_df_ordered = prediction_df[training_columns_order]

    # 3. Transform the input data using the loaded preprocessor
    # The preprocessor handles scaling and one-hot encoding correctly.
    processed_input = preprocessor.transform(prediction_df_ordered)
    
    # 4. Make prediction
    prediction_proba = model.predict(processed_input)[0][0]
    prediction = (prediction_proba > 0.5).astype(int)
    
    # 5. Display the result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"Warning: Machine is LIKELY to fail. (Prediction Probability: {prediction_proba:.2f})")
    else:
        st.success(f"Success: Machine is UNLIKELY to fail. (Prediction Probability: {prediction_proba:.2f})")