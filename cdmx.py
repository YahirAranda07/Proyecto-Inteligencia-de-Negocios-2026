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
    "🗺️ Mapa",
    "🤖 Modelo ML",
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

elif seccion == "🗺️ Mapa":
    st.title("🗺️ Mapa")

    tab1, tab2 = st.tabs(["📍 Propiedades", "🌡️ Plusvalía por alcaldía"])

    with tab1:
        st.markdown("Visualización de propiedades...")
        # resto del código con 8 espacios de indentación

        # Filtros
        col1, col2 = st.columns(2)
        with col1:
            alcaldia_filtro = st.multiselect("Filtrar por alcaldía",
                                              options=sorted(df["places"].unique()),
                                              default=sorted(df["places"].unique()))
        with col2:
            tipo_filtro = st.multiselect("Filtrar por tipo de inmueble",
                                          options=sorted(df["property_type"].unique()),
                                          default=sorted(df["property_type"].unique()))

        # Slider de precio
        precio_min = int(df["price"].quantile(0.01))
        precio_max = int(df["price"].quantile(0.99))
        rango_precio = st.slider(
            "Filtrar por rango de precio (MXN)",
            min_value=precio_min,
            max_value=precio_max,
            value=(precio_min, precio_max),
            step=100000,
            format="$%d"
        )

        df_mapa = df[
            (df["places"].isin(alcaldia_filtro)) &
            (df["property_type"].isin(tipo_filtro)) &
            (df["price"] >= rango_precio[0]) &
            (df["price"] <= rango_precio[1])
        ].dropna(subset=["lat", "lon"])

        st.markdown(f"Mostrando **{len(df_mapa):,}** propiedades")

        mapa = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=11)

        cdmx_url = "https://raw.githubusercontent.com/edavgaun/GeoJson/refs/heads/main/CDMX/alcaldias.geojson"
        cdmx_json = requests.get(cdmx_url).json()

        folium.GeoJson(
            cdmx_json,
            style_function=lambda x: {"fillColor": "lightblue", "color": "gray", "weight": 1.5, "fillOpacity": 0.2},
            tooltip=folium.GeoJsonTooltip(fields=["nomgeo"], aliases=["Alcaldía:"])
        ).add_to(mapa)

        for _, row in df_mapa.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=3,
                color="crimson",
                fill=True,
                fill_opacity=0.6,
                popup=folium.Popup(
                    f"<b>Precio:</b> ${row['price']:,.0f} MXN<br>"
                    f"<b>Alcaldía:</b> {row['places']}<br>"
                    f"<b>Tipo:</b> {row['property_type']}<br>"
                    f"<b>Superficie:</b> {row['surface_covered_in_m2']} m²",
                    max_width=200
                )
            ).add_to(mapa)

        st_folium(mapa, width=1200, height=500)

    # ── Tab 2 — Plusvalía ────────────────────────────────────
    with tab2:
        st.markdown("Alcaldías coloreadas por precio mediano de las propiedades. Rojo indica mayor plusvalía.")

        mapeo = {
            "Benito Juárez": "BenitoJuarez",
            "Gustavo A. Madero": "GustavoAMadero",
            "Álvaro Obregón": "AlvaroObregon",
            "Cuajimalpa de Morelos": "Cuajimalpa",
            "Cuauhtémoc": "Cuauhtemoc",
            "Coyoacán": "Coyoacan",
            "La Magdalena Contreras": "MagdalenaContreras",
            "Tláhuac": "Tlahuac"
        }

        plusvalia = df_clean.groupby("places")["price"].median().reset_index()
        plusvalia.columns = ["places", "precio_mediano"]

        cdmx_url = "https://raw.githubusercontent.com/edavgaun/GeoJson/refs/heads/main/CDMX/alcaldias.geojson"
        cdmx_json2 = requests.get(cdmx_url).json()

        for feature in cdmx_json2["features"]:
            nombre_geo = feature["properties"]["nomgeo"]
            nombre_dataset = mapeo.get(nombre_geo, nombre_geo.replace(" ", ""))
            match = plusvalia[plusvalia["places"] == nombre_dataset]
            feature["properties"]["precio_mediano"] = int(match["precio_mediano"].values[0]) if not match.empty else 0

        precio_max = plusvalia["precio_mediano"].max()
        precio_min = plusvalia["precio_mediano"].min()

        def get_color(precio):
            if precio == 0:
                return "#d3d3d3"
            norm = (precio - precio_min) / (precio_max - precio_min)
            if norm > 0.8:
                return "#bd0026"
            elif norm > 0.6:
                return "#f03b20"
            elif norm > 0.4:
                return "#fd8d3c"
            elif norm > 0.2:
                return "#fecc5c"
            else:
                return "#ffffb2"

        mapa_plusvalia = folium.Map(location=[19.43, -99.13], zoom_start=11)

        folium.GeoJson(
            cdmx_json2,
            style_function=lambda x: {
                "fillColor": get_color(x["properties"]["precio_mediano"]),
                "color": "gray",
                "weight": 1.5,
                "fillOpacity": 0.7
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["nomgeo", "precio_mediano"],
                aliases=["Alcaldía:", "Precio mediano (MXN):"],
                localize=True
            )
        ).add_to(mapa_plusvalia)

        st_folium(mapa_plusvalia, width=1200, height=500)

elif seccion == "🤖 Modelo ML":
    st.title("🤖 Modelo ML")

    st.markdown("""
    Para predecir el precio de los inmuebles se entrenaron y compararon dos modelos:
    - **Regresión Lineal** — asume una relación proporcional entre variables
    - **Árbol de Decisión** — aprende combinaciones de condiciones para estimar el precio
    """)

    st.divider()

    # ── Correlaciones ────────────────────────────────────────
    st.subheader("1. Correlación de variables con el precio")
    st.markdown("Antes de entrenar el modelo analizamos qué variables tienen mayor relación con el precio.")

    columnas = ["price", "surface_covered_in_m2", "price_per_m2", "price_usd_per_m2"]
    correlaciones = df_clean[columnas].corr()

    col1, col2 = st.columns([1, 1])

    with col1:
        fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
        sns.heatmap(correlaciones, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax_corr, linewidths=0.5)
        ax_corr.set_title("Mapa de correlaciones")
        plt.tight_layout()
        st.pyplot(fig_corr)

    with col2:
        st.markdown("#### Interpretación")
        st.markdown("""
        - **price_per_m²** tiene la mayor correlación con el precio **(0.55)**
        - **surface_covered_in_m²** tiene correlación moderada **(0.50)**
        - Ambas variables fueron incluidas en el modelo
        - Las correlaciones mejoraron al limpiar outliers del dataset
        """)

    st.divider()

    # ── Comparación de métricas ──────────────────────────────
    st.subheader("2. Comparación de modelos")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### 📉 Regresión Lineal")
        st.metric("R²", "0.6658", delta=None)
        st.metric("MAE", "$1,059,084 MXN", delta=None)
        st.info("Explica el 67% de la variación en precios. Error promedio de 1 millón de pesos.")

    with col4:
        st.markdown("#### 🌳 Árbol de Decisión")
        st.metric("R²", "0.9265", delta=None)
        st.metric("MAE", "$267,296 MXN", delta=None)
        st.success("Explica el 93% de la variación en precios. Error promedio de $267k pesos.")

    st.divider()

    # ── Gráficas Real vs Predicho ────────────────────────────
    st.subheader("3. Real vs Predicho — Comparación visual")
    st.markdown("Entre más cerca estén los puntos de la línea roja, mejor es el modelo.")

    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    df_model = df_clean[["price", "surface_covered_in_m2", "price_per_m2", "places", "property_type"]].dropna().copy()
    df_model["places_enc"] = le_places.transform(df_model["places"])
    df_model["type_enc"] = le_type.transform(df_model["property_type"])

    X = df_model[["surface_covered_in_m2", "price_per_m2", "places_enc", "type_enc"]]
    y = df_model["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_pred_dt = dt.predict(X_test)

    col5, col6 = st.columns(2)

    with col5:
        fig_lr, ax_lr = plt.subplots(figsize=(6, 5))
        ax_lr.scatter(y_test, y_pred_lr, alpha=0.3, s=10, color="steelblue")
        ax_lr.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linewidth=1.5)
        ax_lr.set_title("Regresión Lineal")
        ax_lr.set_xlabel("Precio real (MXN)")
        ax_lr.set_ylabel("Precio predicho (MXN)")
        ax_lr.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
        ax_lr.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
        plt.tight_layout()
        st.pyplot(fig_lr)

    with col6:
        fig_dt, ax_dt = plt.subplots(figsize=(6, 5))
        ax_dt.scatter(y_test, y_pred_dt, alpha=0.3, s=10, color="steelblue")
        ax_dt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linewidth=1.5)
        ax_dt.set_title("Árbol de Decisión")
        ax_dt.set_xlabel("Precio real (MXN)")
        ax_dt.set_ylabel("Precio predicho (MXN)")
        ax_dt.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
        ax_dt.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
        plt.tight_layout()
        st.pyplot(fig_dt)

    st.divider()

    # ── Conclusión ───────────────────────────────────────────
    st.subheader("4. ¿Por qué el Árbol de Decisión?")
    st.markdown("""
    El Árbol de Decisión fue el modelo ganador porque el precio de un inmueble 
    no depende de una sola fórmula fija — depende de la combinación de varios 
    factores. Por ejemplo, un departamento de 100m² en Miguel Hidalgo tiene un 
    valor muy diferente al mismo departamento en Iztapalapa.
    """)

    st.markdown("#### Variables utilizadas:")
    st.markdown("""
    - 📐 **Metros cuadrados y precio por m²** — miden el tamaño y valor del inmueble
    - 📍 **Alcaldía** (Miguel Hidalgo, Cuauhtémoc, Iztapalapa, etc.) — captura la ubicación
    - 🏠 **Tipo de inmueble** (casa o departamento) — distingue entre segmentos del mercado
    """)

    st.markdown("""
    Para que el modelo no memorizara los datos sino que aprendiera patrones reales, 
    se limitó su complejidad: máximo 8 niveles de profundidad, mínimo 20 propiedades 
    para hacer una división y mínimo 10 propiedades en cada grupo final. Gracias a 
    estos ajustes el modelo logró un **R² de 0.93**, es decir, explica el **93% de la 
    variación en precios** del mercado inmobiliario de la CDMX.
    """)

    st.markdown("""
    | Variable | Descripción | Tipo |
    |---|---|---|
    | `surface_covered_in_m2` | Superficie cubierta | Cuantitativa continua |
    | `price_per_m2` | Precio por metro cuadrado | Cuantitativa continua |
    | `places` | Alcaldía | Cualitativa nominal |
    | `property_type` | Tipo de inmueble | Cualitativa nominal |
    """)

elif seccion == "📝 Observaciones":
    st.title("📝 Observaciones")

    st.markdown("### ¿En qué alcaldía conviene más invertir para compra-venta de inmuebles?")
    st.markdown("Con base en el análisis realizado, estas son las principales conclusiones:")

    st.divider()

    # Observación 1
    st.subheader("📊 Comportamiento del mercado")
    st.markdown("""
    - El mercado inmobiliario de la CDMX está dominado por **departamentos**, los cuales representan la mayor parte de la oferta en casi todas las alcaldías.
    - **Benito Juárez** es la alcaldía con mayor volumen de propiedades disponibles, seguida de **Miguel Hidalgo** y **Álvaro Obregón**.
    - Los precios varían significativamente entre alcaldías — **MagdalenaContreras** y **Cuajimalpa** tienen los precios medianos más altos, mientras que **Iztapalapa**, **Tláhuac** e **Iztacalco** tienen los más bajos.
    """)

    st.divider()

    # Observación 2
    st.subheader("🗺️ Plusvalía y ubicación")
    st.markdown("""
    - Las alcaldías del poniente y sur de la ciudad concentran los precios más altos por m².
    - **Miguel Hidalgo** y **Cuajimalpa** destacan como las zonas de mayor plusvalía.
    - Las alcaldías del oriente como **Iztapalapa**, **Iztacalco** y **Venustiano Carranza** tienen precios más accesibles pero con menor plusvalía histórica.
    """)

    st.divider()

    # Observación 3
    st.subheader("🤖 Modelo de predicción")
    st.markdown("""
    - El **Árbol de Decisión** fue el modelo más preciso con un R² de 0.93 y un error promedio de $267,296 MXN.
    - Las variables más importantes para predecir el precio fueron el **precio por m²** y la **superficie cubierta**.
    - La **alcaldía** y el **tipo de inmueble** también tienen un impacto significativo en el precio final.
    """)

    st.divider()

    # Observación 4 — Conclusión de negocio
    st.subheader("💰 Recomendación de inversión")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("""
        **Mayor volumen**
        
        🏆 Miguel Hidalgo
        
        328 propiedades subvaluadas con descuento promedio de 16.5%
        """)

    with col2:
        st.warning("""
        **Mayor descuento**
        
        🥇 Cuajimalpa
        
        86 propiedades con el mayor descuento promedio de 17.8%
        """)

    with col3:
        st.info("""
        **Menor capital**
        
        💡 Iztapalapa
        
        178 propiedades subvaluadas con precios de entrada más accesibles
        """)

    st.divider()

    st.markdown("""
    En conclusión, **Miguel Hidalgo** es la alcaldía más recomendable para invertir 
    en compra-venta de inmuebles por su combinación de alto volumen de oportunidades, 
    buena plusvalía y descuento promedio competitivo. Para inversores con menor 
    capital disponible, **Iztapalapa** representa una alternativa viable con buena 
    cantidad de oportunidades a precios accesibles.
    """)

elif seccion == "🔮 Predicciones":
    st.title("🔮 Predicciones")
    st.info("Sección en construcción")
