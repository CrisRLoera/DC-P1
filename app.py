import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Cargar el modelo guardado
from sklearn.base import BaseEstimator, TransformerMixin

# Índices de las columnas para las operaciones
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# Ahora puedes cargar el modelo

# Cargar el modelo y el pipeline de preprocesamiento
modelo = joblib.load("final_model.pkl")  # Asegúrate de cambiar esto por la ruta real
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
    st.success(f"El precio estimado de la vivienda es: ${prediccion[0]}")
