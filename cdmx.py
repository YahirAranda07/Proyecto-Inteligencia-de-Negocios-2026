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

    # Filtros
    tipo_filtro = st.multiselect("Filtrar por tipo de inmueble", 
                                  options=df_clean["property_type"].unique(),
                                  default=df_clean["property_type"].unique())
    df_filtrado = df_clean[df_clean["property_type"].isin(tipo_filtro)]
    
    st.divider()

    # Gráfica 1 — Pastel de tipos de inmueble
    st.subheader("Distribución por tipo de inmueble")
    conteo_tipo = df_filtrado["property_type"].value_counts()
    col1, col2 = st.columns(2)
    with col1:
        fig4, ax4 = plt.subplots(figsize=(5, 5))
        ax4.pie(
            conteo_tipo.values,
            labels=conteo_tipo.index,
            autopct="%1.1f%%",
            colors=["steelblue", "salmon"],
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        ax4.set_title("Proporción de casas vs departamentos")
        plt.tight_layout()
        st.pyplot(fig4)
        
    st.subheader("Distribución por tipo de inmueble y alcaldía")
    conteo_tipo = df_filtrado["property_type"].value_counts()
    conteo_places = df_filtrado["places"].value_counts()
    col1, col2 = st.columns(2)

    with col2:
        fig5, ax5 = plt.subplots(figsize=(5, 5))
        conteo_places.sort_values().plot(kind="barh", ax=ax5, color="steelblue", edgecolor="white")
        ax5.set_title("Propiedades por alcaldía")
        ax5.set_xlabel("Número de propiedades")
        ax5.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig5)
    
    st.divider()

    # Gráfica 2 — Precio promedio por alcaldía
    st.subheader("Precio promedio por alcaldía")
    promedio = df_filtrado.groupby("places")["price"].mean().sort_values(ascending=False)
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    promedio.plot(kind="bar", ax=ax1, color="steelblue", edgecolor="white")
    ax1.set_xlabel("Alcaldía")
    ax1.set_ylabel("Precio promedio (MXN)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig1)

    st.divider()

    # Gráfica 3 — Distribución de precios (boxplot)
    st.subheader("Distribución de precios por alcaldía")
    orden = df_filtrado.groupby("places")["price"].median().sort_values(ascending=False).index
    df_filtrado = df_filtrado.copy()
    df_filtrado["places"] = pd.Categorical(df_filtrado["places"], categories=orden, ordered=True)
    df_filtrado = df_filtrado.sort_values("places")
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    df_filtrado.boxplot(column="price", by="places", ax=ax2, showfliers=False)
    ax2.set_xlabel("Alcaldía")
    ax2.set_ylabel("Precio (MXN)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.suptitle("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig2)

    st.divider()

    # Gráfica 4 — Volumen de oferta
    st.subheader("Volumen de oferta por alcaldía y tipo de inmueble")
    volumen = df_filtrado.groupby(["places", "property_type"]).size().unstack(fill_value=0)
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    volumen.plot(kind="bar", ax=ax3, edgecolor="white")
    ax3.set_xlabel("Alcaldía")
    ax3.set_ylabel("Número de propiedades")
    ax3.legend(title="Tipo de inmueble")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig3)
    

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
