import streamlit as st
import trafficaccidentprediction as tp  # Your prediction module
import pyttsx3 as pt

# Streamlit app title
st.markdown("<h1 style='text-align: center; color: black;'>Traffic Accident Prediction</h1>", unsafe_allow_html=True)

# Input fields
weather = st.text_input("Weather Condition")
road_type = st.text_input("Road Type")
time_of_day = st.text_input("Time of Day")
traffic_density = st.number_input("Traffic Density", min_value=0)
speed_limit = st.number_input("Speed Limit", min_value=0)
num_vehicles = st.number_input("Number of Vehicles", min_value=0)
driver_alcohol = st.number_input("Driver Alcohol (1 if under influence, else 0)", min_value=0, max_value=1)
accident_severity = st.text_input("Accident Severity")
road_condition = st.text_input("Road Condition")
vehicle_type = st.text_input("Vehicle Type")
driver_age = st.number_input("Driver Age", min_value=0)
driver_experience = st.number_input("Driver Experience (in years)", min_value=0)
road_light_condition = st.text_input("Road Light Condition")

# Predict button
if st.button("Predict"):
    sample_input = {
        'Weather': weather.strip(),
        'Road_Type': road_type.strip(),
        'Time_of_Day': time_of_day.strip(),
        'Traffic_Density': traffic_density,
        'Speed_Limit': speed_limit,
        'Number_of_Vehicles': num_vehicles,
        'Driver_Alcohol': driver_alcohol,
        'Accident_Severity': accident_severity.strip(),
        'Road_Condition': road_condition.strip(),
        'Vehicle_Type': vehicle_type.strip(),
        'Driver_Age': driver_age,
        'Driver_Experience': driver_experience,
        'Road_Light_Condition': road_light_condition.strip()
    }

    try:
        prediction = tp.func(sample_input)
        st.success(f"Prediction: {prediction}")
        engine=pt.init()
        engine.say(f'prediction is {prediction}')
        engine.runAndWait()
    except Exception as e:
        st.error(f"Error in prediction: {e}")
