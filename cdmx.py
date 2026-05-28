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
    st.subheader("Distribución por tipo de inmueble y alcaldía")
    conteo_tipo = df_filtrado["property_type"].value_counts()
    conteo_places = df_filtrado["places"].value_counts().sort_values()

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

    with col2:
        fig5, ax5 = plt.subplots(figsize=(5, 5))
        conteo_places.plot(kind="barh", ax=ax5, color="steelblue", edgecolor="white")
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

    st.markdown("""
    Para predecir el precio de los inmuebles se entrenaron y compararon dos modelos:
    - **Regresión Lineal** — asume una relación proporcional entre variables
    - **Árbol de Decisión** — aprende combinaciones de condiciones para estimar el precio
    """)

    st.divider()

    # Métricas de los modelos
    st.subheader("Comparación de modelos")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Regresión Lineal")
        st.metric("R²", "0.6658")
        st.metric("MAE", "$1,059,084 MXN")

    with col2:
        st.markdown("#### Árbol de Decisión")
        st.metric("R²", "0.9265")
        st.metric("MAE", "$267,296 MXN")

    st.divider()

    # Explicación de métricas
    st.subheader("¿Qué significan estas métricas?")
    col3, col4 = st.columns(2)

    with col3:
        st.info("**R²** indica qué porcentaje de la variación en precios explica el modelo. Más cercano a 1 es mejor.")

    with col4:
        st.info("**MAE** es el error promedio de predicción en pesos. Entre más bajo, más preciso es el modelo.")

    st.divider()

    # Variables importantes
    st.subheader("Variables utilizadas en el modelo")
    st.markdown("""
    | Variable | Descripción | Tipo |
    |---|---|---|
    | `surface_covered_in_m2` | Superficie cubierta | Cuantitativa continua |
    | `price_per_m2` | Precio por metro cuadrado | Cuantitativa continua |
    | `places` | Alcaldía | Cualitativa nominal |
    | `property_type` | Tipo de inmueble | Cualitativa nominal |
    """)

    st.divider()

    # Conclusión del modelo
    st.subheader("¿Por qué el Árbol de Decisión?")
    st.success("""
    El Árbol de Decisión supera a la Regresión Lineal porque el mercado inmobiliario 
    no es lineal — un m² en Miguel Hidalgo no vale lo mismo que uno en Iztapalapa. 
    El árbol aprende estas combinaciones de factores y predice con mayor precisión.
    """)

elif seccion == "🗺️ Mapa":
    st.title("🗺️ Mapa")
    st.info("Sección en construcción")

elif seccion == "📝 Observaciones":
    st.title("📝 Observaciones")
    st.info("Sección en construcción")

elif seccion == "🔮 Predicciones":
    st.title("🔮 Predicciones")
    st.info("Sección en construcción")
