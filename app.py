import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Para cargar el modelo guardado

# Cargar el modelo y el pipeline de preprocesamiento
modelo = joblib.load("modelo_regresion.pkl")  # Asegúrate de cambiar esto por la ruta real
pipeline = joblib.load("full_pipeline.pkl")

st.title("Predicción del Precio de Vivienda en California")

# Crear los campos de entrada para cada variable
longitude = st.number_input("Longitud", value=-122.23)
latitude = st.number_input("Latitud", value=37.88)
housing_median_age = st.number_input("Edad Media de la Vivienda", value=41.0)
total_rooms = st.number_input("Total de Habitaciones", value=880.0)
total_bedrooms = st.number_input("Total de Dormitorios", value=129.0)
population = st.number_input("Población", value=322.0)
households = st.number_input("Total de Hogares", value=126.0)
median_income = st.number_input("Ingreso Medio", value=8.3252)
ocean_proximity = st.selectbox("Proximidad al Océano", ["NEAR BAY", "INLAND", "NEAR OCEAN", "<1H OCEAN", "ISLAND"])

if st.button("Predecir Precio"):
    # Crear el DataFrame con los datos ingresados
    nuevos_datos = pd.DataFrame({
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],
        "ocean_proximity": [ocean_proximity]
    })
    
    # Aplicar preprocesamiento
    nuevos_datos_preparados = pipeline.transform(nuevos_datos)
    
    # Realizar la predicción
    prediccion = modelo.predict(nuevos_datos_preparados)
    
    # Mostrar el resultado
    st.success(f"El precio estimado de la vivienda es: ${prediccion[0] * 100000:.2f}")
