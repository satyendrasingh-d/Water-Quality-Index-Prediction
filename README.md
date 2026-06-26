# Water-Quality-Index-Prediction
Project
Streamlit : https://water-quality-index-prediction-002586.streamlit.app/

# 💧 Water Quality Assessment Tool

## 📌 Overview
This project is an end-to-end Deep Learning application that evaluates water quality based on chemical properties. It uses deep neural networks to perform two simultaneous tasks based on 14 input parameters:
1. **Regression Task:** Predicts the exact **Water Quality Index (WQI)**.
2. **Classification Task:** Categorizes the water into a specific **Water Quality Classification** (e.g., Good, Poor, Excellent).

An interactive web interface is provided using **Streamlit**, allowing users to easily input chemical metrics and instantly receive predictions without needing to write any code.

## 🚀 Features
* **Deep Learning Powered:** Utilizes Artificial Neural Networks (ANNs) built with TensorFlow and Keras.
* **Dual Output System:** Outputs both continuous (WQI) and categorical (Classification) metrics simultaneously.
* **Interactive UI:** Clean, grid-based web application built with Streamlit.
* **Robust Preprocessing:** Automatically handles missing values using median imputation and scales features using standard scaling.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Machine Learning & Preprocessing:** Scikit-Learn
* **Data Manipulation:** Pandas, NumPy
* **Web Framework:** Streamlit
* **Serialization:** Joblib
