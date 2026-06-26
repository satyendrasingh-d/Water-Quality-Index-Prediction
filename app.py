import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# --- PAGE CONFIG ---
st.set_page_config(page_title="Water Quality Predictor", page_icon="💧", layout="wide")

# --- CACHE MODELS ---
# We use @st.cache_resource so TensorFlow doesn't reload models on every click
@st.cache_resource
def load_assets():
    wqi_model = tf.keras.models.load_model('wqi_model.keras')
    class_model = tf.keras.models.load_model('class_model.keras')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    median_values = joblib.load('median_values.pkl')
    return wqi_model, class_model, scaler, label_encoder, median_values

wqi_model, class_model, scaler, label_encoder, median_values = load_assets()
feature_cols = ['pH', 'EC', 'CO3', 'HCO3', 'Cl', 'SO4', 'NO3', 'TH', 'Ca', 'Mg', 'Na', 'K', 'F', 'TDS']

# --- UI HEADER ---
st.title("💧 Water Quality Assessment Tool")
st.markdown("Enter the chemical parameters of the water sample below to predict its **Water Quality Index (WQI)** and **Classification**.")
st.divider()

# --- INPUT FORM ---
st.subheader("Chemical Parameters")

# Create 4 columns for a clean grid layout
col1, col2, col3, col4 = st.columns(4)

with col1:
    pH = st.number_input("pH Level", value=7.5)
    EC = st.number_input("Electrical Conductivity (EC)", value=300.0)
    CO3 = st.number_input("Carbonate (CO3)", value=20.0)
    HCO3 = st.number_input("Bicarbonate (HCO3)", value=100.0)

with col2:
    Cl = st.number_input("Chloride (Cl)", value=30.0)
    SO4 = st.number_input("Sulfate (SO4)", value=25.0)
    NO3 = st.number_input("Nitrate (NO3)", value=5.0)
    TH = st.number_input("Total Hardness (TH)", value=150.0)

with col3:
    Ca = st.number_input("Calcium (Ca)", value=40.0)
    Mg = st.number_input("Magnesium (Mg)", value=20.0)
    Na = st.number_input("Sodium (Na)", value=50.0)
    K = st.number_input("Potassium (K)", value=3.0)

with col4:
    F = st.number_input("Fluoride (F)", value=0.5)
    TDS = st.number_input("Total Dissolved Solids (TDS)", value=200.0)

# --- PREDICTION LOGIC ---
st.divider()
if st.button("Predict Water Quality", type="primary"):
    with st.spinner("Analyzing parameters..."):
        # 1. Gather inputs into a dictionary
        input_data = {
            'pH': pH, 'EC': EC, 'CO3': CO3, 'HCO3': HCO3, 'Cl': Cl, 'SO4': SO4,
            'NO3': NO3, 'TH': TH, 'Ca': Ca, 'Mg': Mg, 'Na': Na, 'K': K, 'F': F, 'TDS': TDS
        }
        
        # 2. Process data exactly like the training script
        input_df = pd.DataFrame([input_data], columns=feature_cols)
        input_df_filled = input_df.fillna(median_values)
        input_scaled = scaler.transform(input_df_filled)
        
        # 3. Predict WQI (Regression)
        predicted_wqi = wqi_model.predict(input_scaled, verbose=0)[0][0]
        
        # 4. Predict Classification (Categorical)
        class_probabilities = class_model.predict(input_scaled, verbose=0)
        predicted_class_idx = np.argmax(class_probabilities, axis=1)[0]
        predicted_class_label = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # --- DISPLAY RESULTS ---
        st.subheader("Results")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric(label="Predicted WQI", value=f"{predicted_wqi:.2f}")
            
        with res_col2:
            st.metric(label="Predicted Classification", value=predicted_class_label)
