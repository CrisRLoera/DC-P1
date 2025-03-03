import streamlit as st
import numpy as np
import pandas as pd
import os
import tarfile
import urllib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Descargar datos
PROJECT_ROOT_DIR = "."
DOWNLOAD_ROOT = "http://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()

# Preparar datos
housing = housing.dropna()
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)
num_attribs = housing.select_dtypes(include=["number"]).columns.tolist()
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)

# Entrenar modelo
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# Interfaz de usuario con Streamlit
st.title("Predicción del Precio de una Propiedad")
st.write("Ingrese los valores de la propiedad para obtener una predicción.")

input_data = {}
for col in num_attribs:
    input_data[col] = st.number_input(col, value=float(housing[col].median()))

input_data["ocean_proximity"] = st.selectbox("Ubicación", housing["ocean_proximity"].unique())

if st.button("Predecir Precio"):
    input_df = pd.DataFrame([input_data])
    input_prepared = full_pipeline.transform(input_df)
    prediction = lin_reg.predict(input_prepared)
    st.write(f"Precio estimado: ${prediction[0]:,.2f}")
