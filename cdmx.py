import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import requests
import joblib
from streamlit_folium import st_folium

# ── Configuración de la página ──────────────────────────────
st.set_page_config(
    page_title="Inversión Inmobiliaria CDMX",
    page_icon="🏠",
    layout="wide"
)

# ── Cargar datos y modelo ────────────────────────────────────
@st.cache_data
def cargar_datos():
    df = pd.read_csv("housing_data_CDMX.csv")
    df = df[~df["property_type"].isin(["store", "PH"])]
    df[["lat", "lon"]] = (
        df["lat-lon"]
        .str.replace(":", ",")
        .str.split(",", expand=True)
        .apply(lambda x: x.str.strip())
        .astype(float)
    )
    df_clean = df[
        (df["price"] > df["price"].quantile(0.01)) &
        (df["price"] < df["price"].quantile(0.99)) &
        (df["surface_covered_in_m2"] > df["surface_covered_in_m2"].quantile(0.01)) &
        (df["surface_covered_in_m2"] < df["surface_covered_in_m2"].quantile(0.99))
    ]
    return df, df_clean

@st.cache_resource
def cargar_modelo():
    dt = joblib.load("modelo_arbol.pkl")
    le_places = joblib.load("encoder_places.pkl")
    le_type = joblib.load("encoder_type.pkl")
    return dt, le_places, le_type

df, df_clean = cargar_datos()
dt, le_places, le_type = cargar_modelo()

# ── Navegación lateral ───────────────────────────────────────
st.sidebar.title("🏠 Inversión Inmobiliaria CDMX")
seccion = st.sidebar.radio("Navegación", [
    "🏠 Inicio",
    "📊 Gráficas",
    "🤖 Modelo ML",
    "🗺️ Mapa",
    "📝 Observaciones",
    "🔮 Predicciones"
])

# ── Secciones ────────────────────────────────────────────────
if seccion == "🏠 Inicio":
    st.title("Análisis del Mercado Inmobiliario - CDMX")
    st.markdown("### ¿En qué alcaldía conviene más invertir para compra-venta de inmuebles?")
    st.write("Usa el menú lateral para navegar entre las secciones del análisis.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total propiedades", f"{len(df):,}")
    col2.metric("Alcaldías analizadas", df["places"].nunique())
    col3.metric("Precio mediano", f"${df_clean['price'].median():,.0f} MXN")

elif seccion == "📊 Gráficas":
    st.title("📊 Gráficas")
    st.info("Sección en construcción")

elif seccion == "🤖 Modelo ML":
    st.title("🤖 Modelo ML")
    st.info("Sección en construcción")

elif seccion == "🗺️ Mapa":
    st.title("🗺️ Mapa")
    st.info("Sección en construcción")

elif seccion == "📝 Observaciones":
    st.title("📝 Observaciones")
    st.info("Sección en construcción")

elif seccion == "🔮 Predicciones":
    st.title("🔮 Predicciones")
    st.info("Sección en construcción")
