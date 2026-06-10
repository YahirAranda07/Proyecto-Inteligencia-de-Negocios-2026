import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import requests
import joblib
import os
import copy
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# ── Configuración de la página ──────────────────────────────
st.set_page_config(
    page_title="Inversión Inmobiliaria CDMX",
    page_icon="🏠",
    layout="wide"
)

st.markdown("""
    <style>
    div[role="radiogroup"] label p {
        font-size: 20px !important;
        font-weight: 600 !important;
    }
    .stRadio > label {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# ── Cargar datos y modelo ────────────────────────────────────
@st.cache_data
def cargar_datos():
    base = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base, "housing_data_CDMX.csv"))
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
    base = os.path.dirname(__file__)
    dt = joblib.load(os.path.join(base, "modelo_arbol.pkl"))
    le_places = joblib.load(os.path.join(base, "encoder_places.pkl"))
    le_type = joblib.load(os.path.join(base, "encoder_type.pkl"))
    return dt, le_places, le_type

@st.cache_data
def calcular_oportunidades(_dt, _le_places, _le_type, _df_clean):
    df_model2 = _df_clean[["price", "surface_covered_in_m2", "price_per_m2", "places", "property_type"]].dropna().copy()
    df_model2["places_enc"] = _le_places.transform(df_model2["places"])
    df_model2["type_enc"] = _le_type.transform(df_model2["property_type"])
    X_all = df_model2[["surface_covered_in_m2", "price_per_m2", "places_enc", "type_enc"]]
    df_model2["precio_predicho"] = _dt.predict(X_all)
    df_model2["diferencia_pct"] = ((df_model2["price"] - df_model2["precio_predicho"]) / df_model2["precio_predicho"]) * 100
    return df_model2

@st.cache_data
def calcular_modelos(_df_clean, _le_places, _le_type, _dt):
    df_model = _df_clean[["price", "surface_covered_in_m2", "price_per_m2", "places", "property_type"]].dropna().copy()
    df_model["places_enc"] = _le_places.transform(df_model["places"])
    df_model["type_enc"] = _le_type.transform(df_model["property_type"])
    X = df_model[["surface_covered_in_m2", "price_per_m2", "places_enc", "type_enc"]]
    y = df_model["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    dt_local = DecisionTreeRegressor(max_depth=6, min_samples_split=100, min_samples_leaf=50, random_state=42)
    dt_local.fit(X_train, y_train)
    y_pred_dt = dt_local.predict(X_test)
    return y_test, y_pred_lr, y_pred_dt

@st.cache_data
def cargar_geojson():
    cdmx_url = "https://raw.githubusercontent.com/edavgaun/GeoJson/refs/heads/main/CDMX/alcaldias.geojson"
    return requests.get(cdmx_url).json()

# ── Precalcular todo al inicio ───────────────────────────────
df, df_clean = cargar_datos()
dt, le_places, le_type = cargar_modelo()
df_model2 = calcular_oportunidades(dt, le_places, le_type, df_clean)
y_test, y_pred_lr, y_pred_dt = calcular_modelos(df_clean, le_places, le_type, dt)
cdmx_geojson = cargar_geojson()

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
    st.title("🏠 Análisis de Propiedades - CDMX")

    st.divider()


    # Objetivo
    st.subheader("🎯 ¿Para qué sirve este tablero?")
    st.markdown("""
    Este tablero nació para responder las preguntas que cualquier joven comprador se hace 
    antes de tomar la decisión más importante de su vida financiera:

    - **¿El precio que me están pidiendo es justo?** — o me están cobrando de más por la zona
    - **¿Dónde me conviene más invertir** con el presupuesto que tengo?
    - **¿Qué tan caro es el m²** en la alcaldía que me interesa comparado con las demás?
    - **¿Existe alguna propiedad subvaluada** que nadie más ha detectado?


    """)

    st.divider()

    # Qué encontrarás
    st.subheader("🗂️ ¿Qué encontrarás en este tablero?")

    col13, col14, col15 = st.columns(3)

    with col13:
        st.info("""
        **📊 Gráficas**
        
        Visualizaciones del mercado inmobiliario — distribución de precios, volumen de oferta 
        por alcaldía y tipo de inmueble.
        """)
        st.info("""
        **🗺️ Mapa**
        
        Mapa interactivo con la ubicación de propiedades y un mapa de calor de plusvalía 
        por alcaldía.
        """)

    with col14:
        st.info("""
        **🤖 Modelo ML**
        
        Comparación entre Regresión Lineal y Árbol de Decisión para predecir precios 
        con un R² de 0.89.
        """)
        st.info("""
        **📝 Observaciones**
        
        Hallazgos principales del análisis y recomendaciones de inversión por alcaldía 
        basadas en datos.
        """)

    with col15:
        st.info("""
        **🔮 Predicciones**
        
        Herramienta interactiva para estimar el precio de cualquier inmueble según 
        sus características y compararlo con el mercado.
        """)

    st.divider()
    
elif seccion == "📊 Gráficas":
    st.title("📊 Gráficas")

    tipo_filtro = st.multiselect("Filtrar por tipo de inmueble",
                                  options=df_clean["property_type"].unique(),
                                  default=df_clean["property_type"].unique())
    df_filtrado = df_clean[df_clean["property_type"].isin(tipo_filtro)]

    st.divider()

    # Gráfica 1 — Pastel y barras
    st.subheader("Distribución por tipo de inmueble y lugar en CDMX")
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
        fig_bar, ax_bar = plt.subplots(figsize=(5, 5))
        conteo_places.plot(kind="barh", ax=ax_bar, color="steelblue", edgecolor="white")
        ax_bar.set_title("Propiedades por lugar en CDMX")
        ax_bar.set_xlabel("Número de propiedades")
        ax_bar.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig_bar)
        
    st.divider()

    # Gráfica 5 — Histograma precios por tipo de inmueble
    st.subheader("Distribución de precios por tipo de inmueble")
    fig5, ax5 = plt.subplots(figsize=(14, 5))

    colores_tipo = {"apartment": "steelblue", "house": "salmon"}

    for tipo in df_clean["property_type"].unique():
        datos = df_clean[df_clean["property_type"] == tipo]["price"]
        ax5.hist(datos, bins=50, alpha=0.6, color=colores_tipo[tipo], edgecolor="white", label=tipo)

    ax5.set_title("Distribución de precios por tipo de inmueble - CDMX", fontsize=14)
    ax5.set_xlabel("Precio (MXN)", fontsize=12)
    ax5.set_ylabel("Número de propiedades", fontsize=12)
    ax5.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax5.set_xlim(100000, df_clean["price"].quantile(0.99))
    ax5.legend(title="Tipo de inmueble")
    plt.tight_layout()
    st.pyplot(fig5)
    
    st.divider()

    # Gráfica 3 — Precio promedio
    st.subheader("Precio promedio por lugar en CDMX")
    promedio = df_filtrado.groupby("places")["price"].mean().sort_values(ascending=False)
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    promedio.plot(kind="bar", ax=ax1, color="steelblue", edgecolor="white")
    ax1.set_xlabel("Lugar en CDMX")
    ax1.set_ylabel("Precio promedio (MXN)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig1)

    st.divider()

    # Gráfica 4 — Boxplot
    st.subheader("Distribución de precios por lugar en CDMX")
    orden = df_filtrado.groupby("places")["price"].median().sort_values(ascending=False).index
    df_filtrado = df_filtrado.copy()
    df_filtrado["places"] = pd.Categorical(df_filtrado["places"], categories=orden, ordered=True)
    df_filtrado = df_filtrado.sort_values("places")
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    df_filtrado.boxplot(column="price", by="places", ax=ax2)
    ax2.set_xlabel("Lugar en CDMX")
    ax2.set_ylabel("Precio (MXN)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.suptitle("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig2)

    st.divider()

    # Gráfica 5 — Volumen de oferta
    st.subheader("Volumen de oferta por lugar en CDMX y tipo de inmueble")
    volumen = df_filtrado.groupby(["places", "property_type"]).size().unstack(fill_value=0)
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    volumen.plot(kind="bar", ax=ax3, edgecolor="white")
    ax3.set_xlabel("Lugar en CDMX")
    ax3.set_ylabel("Número de propiedades")
    ax3.legend(title="Tipo de inmueble")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig3)



elif seccion == "🗺️ Mapa":
    st.title("🗺️ Mapa")

    tab1, tab2 = st.tabs(["📍 Propiedades", "🌡️ Plusvalía por lugar en CDMX"])

    with tab1:
        st.markdown("Visualización de propiedades en el mapa. Haz clic en cada punto para ver el precio y detalles.")

        col1, col2 = st.columns(2)
        with col1:
            lugar_filtro = st.multiselect("Filtrar por lugar en CDMX",
                                          options=sorted(df["places"].unique()),
                                          default=sorted(df["places"].unique()))
        with col2:
            tipo_filtro_mapa = st.multiselect("Filtrar por tipo de inmueble",
                                              options=sorted(df["property_type"].unique()),
                                              default=sorted(df["property_type"].unique()))

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
            (df["places"].isin(lugar_filtro)) &
            (df["property_type"].isin(tipo_filtro_mapa)) &
            (df["price"] >= rango_precio[0]) &
            (df["price"] <= rango_precio[1])
        ].dropna(subset=["lat", "lon"])

        df_mapa_muestra = df_mapa.sample(min(1000, len(df_mapa)), random_state=42)
        st.markdown(f"Mostrando **{len(df_mapa_muestra):,}** de **{len(df_mapa):,}** propiedades")

        mapa = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=11)
        geojson_mapa = copy.deepcopy(cdmx_geojson)

        folium.GeoJson(
            geojson_mapa,
            style_function=lambda x: {"fillColor": "lightblue", "color": "gray", "weight": 1.5, "fillOpacity": 0.2},
            tooltip=folium.GeoJsonTooltip(fields=["nomgeo"], aliases=["Lugar en CDMX:"])
        ).add_to(mapa)

        for _, row in df_mapa_muestra.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=3,
                color="crimson",
                fill=True,
                fill_opacity=0.6,
                popup=folium.Popup(
                    f"<b>Precio:</b> ${row['price']:,.0f} MXN<br>"
                    f"<b>Lugar en CDMX:</b> {row['places']}<br>"
                    f"<b>Tipo:</b> {row['property_type']}<br>"
                    f"<b>Superficie:</b> {row['surface_covered_in_m2']} m²",
                    max_width=200
                )
            ).add_to(mapa)

        st_folium(mapa, width=1200, height=500)

    with tab2:
        st.markdown("Lugares en CDMX coloreados por precio promedio de las propiedades. Rojo indica mayor plusvalía.")

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

        plusvalia = df_clean.groupby("places")["price"].mean().reset_index()
        plusvalia.columns = ["places", "precio_promedio"]

        geojson_plusvalia = copy.deepcopy(cdmx_geojson)

        for feature in geojson_plusvalia["features"]:
            nombre_geo = feature["properties"]["nomgeo"]
            nombre_dataset = mapeo.get(nombre_geo, nombre_geo.replace(" ", ""))
            match = plusvalia[plusvalia["places"] == nombre_dataset]
            feature["properties"]["precio_promedio"] = int(match["precio_promedio"].values[0]) if not match.empty else 0

        precio_max_map = plusvalia["precio_promedio"].max()
        precio_min_map = plusvalia["precio_promedio"].min()

        def get_color(precio):
            if precio == 0:
                return "#d3d3d3"
            norm = (precio - precio_min_map) / (precio_max_map - precio_min_map)
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
            geojson_plusvalia,
            style_function=lambda x: {
                "fillColor": get_color(x["properties"]["precio_promedio"]),
                "color": "gray",
                "weight": 1.5,
                "fillOpacity": 0.7
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["nomgeo", "precio_promedio"],
                aliases=["Lugar en CDMX:", "Precio promedio (MXN):"],
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

    # Correlaciones
    st.subheader("1. Correlación de variables con el precio")
    st.markdown("Antes de entrenar el modelo analizamos qué variables tienen mayor relación con el precio.")

    columnas = ["price", "surface_covered_in_m2", "price_per_m2", "price_usd_per_m2",
                "price_aprox_local_currency", "price_aprox_usd", "surface_total_in_m2"]
    correlaciones = df_clean[columnas].corr()

    col1, col2 = st.columns([1, 1])
    with col1:
        fig_corr, ax_corr = plt.subplots(figsize=(7, 6))
        sns.heatmap(correlaciones, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax_corr, linewidths=0.5, linecolor="white")
        ax_corr.set_title("Mapa de correlaciones")
        plt.tight_layout()
        st.pyplot(fig_corr)

    with col2:
        st.markdown("#### Interpretación")
        st.markdown("""
        - **price_per_m²** tiene la mayor correlación con el precio **(0.55)**
        - **surface_covered_in_m²** tiene correlación moderada **(0.50)**
        - `price_aprox_local_currency` y `price_aprox_usd` tienen correlación de **1.0** entre sí — son la misma variable en diferente moneda
        - `surface_total_in_m2` tiene correlación muy baja **(0.11)** — no aporta al modelo
        - Las variables seleccionadas para el modelo fueron `price_per_m2` y `surface_covered_in_m2`
        """)

    st.divider()

    # Métricas
    st.subheader("2. Comparación de modelos")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### 📉 Regresión Lineal")
        st.metric("R²", "0.6658", delta=None)
        st.metric("MAE", "$1,059,084 MXN", delta=None)
        st.info("Explica el 67% de la variación en precios. Error promedio de 1 millón de pesos.")

    with col4:
        st.markdown("#### 🌳 Árbol de Decisión")
        st.metric("R²", "0.8887", delta=None)
        st.metric("MAE", "$456,307 MXN", delta=None)
        st.success("Explica el 89% de la variación en precios. Error promedio de $456k pesos.")

    st.divider()

    # Gráficas Real vs Predicho
    st.subheader("3. Real vs Predicho — Comparación visual")
    st.markdown("Entre más cerca estén los puntos de la línea roja, mejor es el modelo.")

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
        ax_dt.set_title("Árbol de Decisión (depth=6)")
        ax_dt.set_xlabel("Precio real (MXN)")
        ax_dt.set_ylabel("Precio predicho (MXN)")
        ax_dt.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
        ax_dt.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
        plt.tight_layout()
        st.pyplot(fig_dt)

    st.divider()

    # Conclusión
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
    - 📍 **Lugar en CDMX** (Miguel Hidalgo, Cuauhtémoc, Iztapalapa, etc.) — captura la ubicación
    - 🏠 **Tipo de inmueble** (casa o departamento) — distingue entre segmentos del mercado
    """)

    st.markdown("""
    Para que el modelo no memorizara los datos sino que aprendiera patrones reales, 
    se limitó su complejidad: máximo 6 niveles de profundidad, mínimo 100 propiedades 
    para hacer una división y mínimo 50 propiedades en cada grupo final. Gracias a 
    estos ajustes el modelo logró un **R² de 0.89**, es decir, explica el **89% de la 
    variación en precios** del mercado inmobiliario de la CDMX.
    """)

    st.markdown("""
    | Variable | Descripción | Tipo |
    |---|---|---|
    | `surface_covered_in_m2` | Superficie cubierta | Cuantitativa continua |
    | `price_per_m2` | Precio por metro cuadrado | Cuantitativa continua |
    | `places` | Lugar en CDMX | Cualitativa nominal |
    | `property_type` | Tipo de inmueble | Cualitativa nominal |
    """)
elif seccion == "📝 Observaciones":
    st.title("📝 Observaciones")

    st.markdown("### ¿Dónde puede vivir un joven profesionista en la CDMX sin pagar de más?")
    st.markdown("Con base en el análisis realizado, estas son las principales conclusiones para un comprador entre 23 y 30 años.")

    st.divider()

    st.subheader("📊 El mercado que te vas a encontrar")
    st.markdown("""
    - El **87% de la oferta son departamentos** — si estás buscando tu primer inmueble, casi seguro será un departamento.
    - Los precios varían enormemente dentro de la misma alcaldía. En **Benito Juárez**, que tiene la mayor oferta con 3,659 propiedades, puedes encontrar desde $800,000 hasta más de $10,000,000 MXN — lo que hace casi imposible saber si lo que te ofrecen es justo sin un punto de referencia.
    - Alcaldías como **Roma, Condesa y Narvarte** — que pertenecen a **Cuauhtémoc y Benito Juárez** — han experimentado un alza sostenida de precios impulsada por la llegada de trabajadores extranjeros que perciben ingresos en dólares. Esto significa que el comprador mexicano recién egresado compite en un mercado que ya no está calibrado para su nivel de ingreso.
    """)

    st.divider()

    st.subheader("💸 ¿Qué alcaldías se ajustan a un presupuesto real?")
    st.markdown("""
    Un profesionista recién egresado que destina entre el 25% y 30% de su ingreso a vivienda 
    generalmente tiene acceso a un crédito de entre **$800,000 y $2,000,000 MXN**. 
    Con eso en mente, el panorama es el siguiente:

    - **Iztapalapa, Iztacalco, Tláhuac y Venustiano Carranza** son las alcaldías con precios medianos más accesibles — por debajo de $1,500,000 MXN.
    - **Azcapotzalco y Gustavo A. Madero** ofrecen un balance entre precio y ubicación, con medianas entre $1,200,000 y $1,800,000 MXN.
    - **Coyoacán y Álvaro Obregón** son opciones intermedias con buena calidad de vida pero precios ya más elevados — medianas entre $2,000,000 y $3,000,000 MXN.
    - **Benito Juárez, Miguel Hidalgo y Cuajimalpa** están fuera del alcance de la mayoría de los compradores jóvenes con medianas superiores a $3,500,000 MXN — y con el efecto del nearshoring, sus precios siguen subiendo.
    """)

    st.divider()

    st.subheader("🗺️ El vendedor siempre lleva ventaja — hasta ahora")
    st.markdown("""
    Quien vende lleva años en el mercado y sabe exactamente cuánto vale su propiedad y hasta dónde puede negociar. 
    El comprador que busca por primera vez llega sin ningún punto de referencia.

    Nuestro modelo de Machine Learning cambia eso. Con un **R² de 0.89** y un error promedio de **$456,307 MXN**, 
    el modelo es capaz de estimar el valor justo de una propiedad basándose en su alcaldía, 
    tipo de inmueble, metros cuadrados y precio por m². 

    Esto permite identificar propiedades cuyo precio real está **entre 10% y 40% por debajo** 
    de lo que el mercado indica que deberían costar — es decir, propiedades donde el comprador 
    tiene una ventaja real de negociación.
    """)

    st.divider()

    st.subheader("💰 Oportunidades detectadas por el modelo")
    st.markdown("Propiedades cuyo precio real está entre 10% y 40% por debajo del valor estimado por el modelo.")

    oportunidades_reales = df_model2[
        (df_model2["diferencia_pct"] < -10) &
        (df_model2["diferencia_pct"] > -40)
    ].copy()
    oportunidades_reales["diferencia_pct_pos"] = oportunidades_reales["diferencia_pct"].abs()

    col1, col2 = st.columns(2)

    with col1:
        cantidad = oportunidades_reales.groupby("places", observed=True).size().sort_values(ascending=False)
        cantidad = cantidad[cantidad > 0]
        colores = ["green" if x >= 500 else "steelblue" if x >= 200 else "lightblue" for x in cantidad]
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        bars = ax1.bar(cantidad.index, cantidad.values, color=colores, edgecolor="white")
        ax1.set_title("Cantidad de propiedades subvaluadas por lugar en CDMX", fontsize=11)
        ax1.set_xlabel("Lugar en CDMX")
        ax1.set_ylabel("Cantidad de propiedades")
        for bar, val in zip(bars, cantidad.values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     str(val), ha="center", va="bottom", fontsize=8, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig1)

    with col2:
        descuento = oportunidades_reales.groupby("places", observed=True)["diferencia_pct_pos"].mean().sort_values(ascending=False)
        descuento = descuento[descuento > 0]
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        bars2 = ax2.bar(descuento.index, descuento.values, color="salmon", edgecolor="white")
        ax2.set_title("% promedio por debajo del valor de mercado", fontsize=11)
        ax2.set_xlabel("Lugar en CDMX")
        ax2.set_ylabel("Descuento promedio (%)")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        for bar, val in zip(bars2, descuento.values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig2)

    st.divider()

    st.subheader("🏆 ¿Dónde están las mejores oportunidades?")

    col3, col4, col5 = st.columns(3)

    with col3:
        st.success("""
        **Mayor volumen de oportunidades**
        
        🏆 Benito Juárez
        
        770 propiedades subvaluadas — ideal si quieres variedad de opciones en una zona céntrica y con alta demanda de reventa.
        """)

    with col4:
        st.warning("""
        **Mayor margen de negociación**
        
        🥇 Tláhuac y Cuajimalpa
        
        Descuentos promedio de 22.9% y 22.2% — para quienes buscan el mayor diferencial entre precio de compra y valor real.
        """)

    with col5:
        st.info("""
        **Mejor opción para presupuesto limitado**
        
        💡 Iztapalapa
        
        392 oportunidades con 20.8% de descuento promedio y precios de entrada accesibles para un primer crédito hipotecario.
        """)

    st.divider()

    st.markdown("""
    **¿Qué significa esto para un comprador joven?**
    
    Si tu presupuesto es limitado y buscas tu primer inmueble, **Iztapalapa** ofrece la combinación 
    más realista de precio accesible y oportunidades detectadas por el modelo. Si tienes un poco más 
    de capital y buscas una zona con mayor plusvalía y facilidad de reventa, **Benito Juárez** 
    concentra la mayor cantidad de propiedades subvaluadas en una de las alcaldías más demandadas 
    de la ciudad. Y si lo que buscas es el mayor margen de negociación posible, 
    **Tláhuac y Cuajimalpa** son donde el modelo detecta las brechas más grandes entre 
    precio real y valor de mercado.
    """)
elif seccion == "🔮 Predicciones":
    st.title("🔮 Predicciones")
    st.markdown("Ingresa tus preferencias y te mostraremos el **Top 3 de mejores alcaldías** para comprar según tu presupuesto y necesidades.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        presupuesto_min = st.number_input(
            "💰 Presupuesto mínimo (MXN)",
            min_value=100000,
            max_value=50000000,
            value=800000,
            step=50000,
            format="%d"
        )
        presupuesto_max = st.number_input(
            "💰 Presupuesto máximo (MXN)",
            min_value=100000,
            max_value=50000000,
            value=2000000,
            step=50000,
            format="%d"
        )

    with col2:
        metros = st.number_input(
            "📐 Metros cuadrados deseados",
            min_value=20,
            max_value=500,
            value=70,
            step=5
        )
        tipo = st.selectbox(
            "🏠 Tipo de inmueble",
            options=["apartment", "house"],
            format_func=lambda x: "Departamento" if x == "apartment" else "Casa"
        )

    st.divider()

    if st.button("🔍 Buscar mejores opciones", use_container_width=True):

        if presupuesto_min >= presupuesto_max:
            st.error("El presupuesto mínimo debe ser menor al presupuesto máximo.")
        else:
            tipo_enc = le_type.transform([tipo])[0]
            resultados = []

            for lugar in sorted(le_places.classes_):
                lugar_enc = le_places.transform([lugar])[0]
                precio_m2_promedio = df_clean[
                    (df_clean["places"] == lugar) &
                    (df_clean["property_type"] == tipo)
                ]["price_per_m2"].mean()

                if pd.isna(precio_m2_promedio):
                    continue

                X_pred = pd.DataFrame([[metros, precio_m2_promedio, lugar_enc, tipo_enc]],
                                      columns=["surface_covered_in_m2", "price_per_m2", "places_enc", "type_enc"])
                precio_estimado = dt.predict(X_pred)[0]

                # Solo incluir si está dentro del presupuesto
                if presupuesto_min <= precio_estimado <= presupuesto_max:
                    precio_real_promedio = df_clean[
                        (df_clean["places"] == lugar) &
                        (df_clean["property_type"] == tipo)
                    ]["price"].mean()

                    props_disponibles = len(df_clean[
                        (df_clean["places"] == lugar) &
                        (df_clean["property_type"] == tipo) &
                        (df_clean["price"] >= presupuesto_min) &
                        (df_clean["price"] <= presupuesto_max)
                    ])

                    oport = len(df_model2[
                        (df_model2["places"] == lugar) &
                        (df_model2["property_type"] == tipo) &
                        (df_model2["diferencia_pct"] < -10) &
                        (df_model2["diferencia_pct"] > -40) &
                        (df_model2["price"] >= presupuesto_min) &
                        (df_model2["price"] <= presupuesto_max)
                    ])

                    resultados.append({
                        "lugar": lugar,
                        "precio_estimado": precio_estimado,
                        "precio_real_promedio": precio_real_promedio,
                        "props_disponibles": props_disponibles,
                        "oportunidades": oport,
                        "precio_m2": precio_m2_promedio
                    })

            # Ordenar por mayor número de oportunidades y propiedades disponibles
            resultados = sorted(resultados, key=lambda x: (x["oportunidades"], x["props_disponibles"]), reverse=True)

            if len(resultados) == 0:
                st.warning("No encontramos alcaldías que se ajusten a tu presupuesto y preferencias. Intenta ampliar tu rango de presupuesto o ajustar los metros cuadrados.")

            else:
                top3 = resultados[:3]
                st.subheader(f"🏆 Top 3 mejores opciones para ti")
                st.markdown(f"Resultados para: **{'Departamento' if tipo == 'apartment' else 'Casa'}** de **{metros} m²** con presupuesto de **${presupuesto_min:,.0f} — ${presupuesto_max:,.0f} MXN**")

                st.divider()

                medallas = ["🥇", "🥈", "🥉"]
                colores_card = ["success", "warning", "info"]

                for i, res in enumerate(top3):
                    if colores_card[i] == "success":
                        st.success(f"""
                        {medallas[i]} **{res['lugar']}**
                        """)
                    elif colores_card[i] == "warning":
                        st.warning(f"""
                        {medallas[i]} **{res['lugar']}**
                        """)
                    else:
                        st.info(f"""
                        {medallas[i]} **{res['lugar']}**
                        """)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Precio estimado", f"${res['precio_estimado']:,.0f} MXN")
                    c2.metric("Precio promedio real", f"${res['precio_real_promedio']:,.0f} MXN")
                    c3.metric("Propiedades disponibles", f"{res['props_disponibles']:,}")
                    c4.metric("Oportunidades subvaluadas", f"{res['oportunidades']:,}")

                    st.markdown(f"📐 Precio por m² promedio en {res['lugar']}: **${res['precio_m2']:,.0f} MXN**")
                    st.divider()

                if len(resultados) > 3:
                    st.markdown(f"*También encontramos {len(resultados) - 3} alcaldías adicionales que se ajustan a tu presupuesto.*")
