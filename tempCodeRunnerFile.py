import pickle
import numpy as np
import streamlit as st

# Load the trained crop recommendation model
with open('E:/crop1/random_forest_model_crop.pkl', 'rb') as file:
    rf_model_crop = pickle.load(file)

# Load the trained fertilizer recommendation model
with open('E:/crop1/random_forest_model_fertilizer.pkl', 'rb') as file:
    rf_model_fertilizer = pickle.load(file)

# Define recommendation functions
def recommend_crop(ph, humidity, N, P, K, temperature, rainfall):
    input_data = np.array([[ph, humidity, N, P, K, temperature, rainfall]])
    crop_prediction = rf_model_crop.predict(input_data)
    return crop_prediction[0]

def recommend_fertilizer(temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous):
    soil_type_mapping = {'Sandy': 0, 'Clayey': 1, 'Loamy': 2, 'Silt': 3}
    crop_type_mapping = {'Wheat': 0, 'Rice': 1, 'Maize': 2, 'Barley': 3}

    soil_type_encoded = soil_type_mapping.get(soil_type, -1)
    crop_type_encoded = crop_type_mapping.get(crop_type, -1)

    if soil_type_encoded == -1 or crop_type_encoded == -1:
        raise ValueError("Invalid soil type or crop type provided!")

    input_data = np.array([[temperature, humidity, moisture, soil_type_encoded, crop_type_encoded, nitrogen, potassium, phosphorous]])
    fertilizer_prediction = rf_model_fertilizer.predict(input_data)
    return fertilizer_prediction[0]

# Custom CSS for background
def add_background_image():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://png.pngtree.com/thumb_back/fh260/background/20210302/pngtree-crop-green-rice-light-effect-wallpaper-image_571433.jpg");
            background-size: cover;
        }
        .custom-success {
            background-color: #90EE90;
            color: black;
            font-size: 30px;
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Streamlit Application
def main():
    add_background_image()  # Add the background image style

    st.title("Crop & Fertilizer Recommendation System")
    st.write("Provide soil and environmental parameters for machine learning-based recommendations.")

    # Input fields for user to provide data
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1, value=6.5)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=1.0, value=60.0)
    moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, step=1.0, value=50.0)
    soil_type = st.selectbox("Soil Type", options=['Sandy', 'Clayey', 'Loamy', 'Silt'])
    crop_type = st.selectbox("Crop Type", options=['Wheat', 'Rice', 'Maize', 'Barley'])
    nitrogen = st.number_input("Nitrogen (mg/Kg)", min_value=0.0, max_value=200.0, step=1.0, value=50.0)
    potassium = st.number_input("Potassium (mg/Kg)", min_value=0.0, max_value=200.0, step=1.0, value=50.0)
    phosphorous = st.number_input("Phosphorous (mg/Kg)", min_value=0.0, max_value=200.0, step=1.0, value=25.0)
    temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, step=1.0, value=25.0)

    if st.button("🌱 Get Recommendations"):
        crop_result = recommend_crop(ph, humidity, nitrogen, phosphorous, potassium, temperature, rainfall=200.0)
        fertilizer_result = recommend_fertilizer(temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous)
        

        st.markdown(f'<div class="custom-success">🌾 Recommended Crop: {crop_result}</div>', unsafe_allow_html=True)
        crop_desc = f"{crop_result} is a highly recommended crop for the given conditions. It is known for its adaptability and yield."
       
        st.markdown(
            f'<div style="background-color:#e0ffe0; padding:15px; border-radius:8px; margin-bottom:10px;">'
            f"<strong>About {crop_result}:</strong> {crop_desc}"
            '</div>',
            unsafe_allow_html=True
        )# Adjusting text size
        st.markdown('<p style="font-size:24px; font-weight:bold;"></p>', unsafe_allow_html=True)

    

        st.markdown(f'<div class="custom-success">🌿 Recommended Fertilizer: {fertilizer_result}</div>', unsafe_allow_html=True)
        fertilizer_desc = f"{fertilizer_result} is suitable for enhancing soil fertility and supporting the growth of {crop_result}."
       
        st.markdown(
            f'<div style="background-color:#e0ffe0; padding:15px; border-radius:8px; margin-bottom:10px;">'
            f"<strong>About {fertilizer_result}:</strong> {fertilizer_desc}"
            '</div>',
            unsafe_allow_html=True
        )

     
     

if _name_ == "_main_":
    main()