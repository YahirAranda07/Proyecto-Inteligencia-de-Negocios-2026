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
    st.title("Análisis del Mercado Inmobiliario - CDMX")
    st.markdown("### ¿En qué lugar en CDMX conviene más invertir para compra-venta de inmuebles?")
    st.write("Usa el menú lateral para navegar entre las secciones del análisis.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total propiedades", f"{len(df):,}")
    col2.metric("Lugares en CDMX analizados", df["places"].nunique())
    col3.metric("Precio mediano", f"${df_clean['price'].median():,.0f} MXN")

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

    # Gráfica 2 — Precio promedio
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

    # Gráfica 3 — Boxplot
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

    # Gráfica 4 — Volumen de oferta
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

    st.divider()

    # Gráfica 5 — Histograma precios
    st.subheader("Distribución de precios")
    fig5, ax5 = plt.subplots(figsize=(14, 5))
    ax5.hist(df_clean["price"], bins=50, color="steelblue", edgecolor="white")
    ax5.set_title("Distribución de precios - CDMX", fontsize=14)
    ax5.set_xlabel("Precio (MXN)", fontsize=12)
    ax5.set_ylabel("Número de propiedades", fontsize=12)
    ax5.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax5.set_xlim(100000, df_clean["price"].quantile(0.99))
    plt.tight_layout()
    st.pyplot(fig5)

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

        import copy
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
        st.markdown("Lugares en CDMX coloreados por precio mediano de las propiedades. Rojo indica mayor plusvalía.")

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

        import copy
        geojson_plusvalia = copy.deepcopy(cdmx_geojson)

        for feature in geojson_plusvalia["features"]:
            nombre_geo = feature["properties"]["nomgeo"]
            nombre_dataset = mapeo.get(nombre_geo, nombre_geo.replace(" ", ""))
            match = plusvalia[plusvalia["places"] == nombre_dataset]
            feature["properties"]["precio_mediano"] = int(match["precio_mediano"].values[0]) if not match.empty else 0

        precio_max_map = plusvalia["precio_mediano"].max()
        precio_min_map = plusvalia["precio_mediano"].min()

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
                "fillColor": get_color(x["properties"]["precio_mediano"]),
                "color": "gray",
                "weight": 1.5,
                "fillOpacity": 0.7
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["nomgeo", "precio_mediano"],
                aliases=["Lugar en CDMX:", "Precio mediano (MXN):"],
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

    st.markdown("### ¿En qué lugar en CDMX conviene más invertir para compra-venta de inmuebles?")
    st.markdown("Con base en el análisis realizado, estas son las principales conclusiones:")

    st.divider()

    st.subheader("📊 Comportamiento del mercado")
    st.markdown("""
    - El mercado inmobiliario de la CDMX está dominado por **departamentos**, los cuales representan la mayor parte de la oferta en casi todos los lugares.
    - **Benito Juárez** es el lugar con mayor volumen de propiedades disponibles, seguida de **Miguel Hidalgo** y **Álvaro Obregón**.
    - Los precios varían significativamente entre lugares — **MagdalenaContreras** y **Cuajimalpa** tienen los precios medianos más altos, mientras que **Iztapalapa**, **Tláhuac** e **Iztacalco** tienen los más bajos.
    """)

    st.divider()

    st.subheader("🗺️ Plusvalía y ubicación")
    st.markdown("""
    - Los lugares del poniente y sur de la ciudad concentran los precios más altos por m².
    - **Miguel Hidalgo** y **Cuajimalpa** destacan como los lugares de mayor plusvalía.
    - Los lugares del oriente como **Iztapalapa**, **Iztacalco** y **Venustiano Carranza** tienen precios más accesibles pero con menor plusvalía histórica.
    """)

    st.divider()

    st.subheader("🤖 Modelo de predicción")
    st.markdown("""
    - El **Árbol de Decisión** fue el modelo más preciso con un R² de 0.89 y un error promedio de $456,307 MXN.
    - Las variables más importantes para predecir el precio fueron el **precio por m²** y la **superficie cubierta**.
    - El **lugar en CDMX** y el **tipo de inmueble** también tienen un impacto significativo en el precio final.
    """)

    st.divider()

    st.subheader("💰 Oportunidades de inversión")
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

    st.subheader("🏆 Recomendación de inversión")

    col3, col4, col5 = st.columns(3)

    with col3:
        st.success("""
        **Mayor volumen**
        
        🏆 Benito Juárez
        
        770 propiedades subvaluadas con descuento promedio de 16.0%
        """)

    with col4:
        st.warning("""
        **Mayor descuento**
        
        🥇 Tláhuac
        
        95 propiedades con el mayor descuento promedio de 22.9%
        """)

    with col5:
        st.info("""
        **Balance volumen/descuento**
        
        💡 Iztapalapa
        
        392 propiedades subvaluadas con descuento promedio de 20.8%
        """)

    st.divider()

    st.markdown("""
    En conclusión, **Benito Juárez** es el lugar con mayor volumen de oportunidades para invertir 
    en compra-venta de inmuebles con 770 propiedades subvaluadas. Para inversores que buscan 
    el mayor margen de ganancia, **Tláhuac** y **Cuajimalpa** ofrecen los mayores descuentos 
    promedio con 22.9% y 22.2% respectivamente. Para un balance entre volumen y descuento, 
    **Iztapalapa** representa la mejor alternativa con 392 oportunidades y un descuento promedio de 20.8%.
    """)

elif seccion == "🔮 Predicciones":
    st.title("🔮 Predicciones")
    st.markdown("Ingresa las características del inmueble para estimar su valor de mercado según el modelo.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        lugar = st.selectbox("Lugar en CDMX", options=sorted(le_places.classes_))
        tipo = st.selectbox("Tipo de inmueble", options=sorted(le_type.classes_))

    with col2:
        metros = st.number_input("Superficie cubierta (m²)", min_value=10, max_value=1000, value=80, step=5)
        precio_m2 = st.number_input("Precio por m² (MXN)", min_value=1000, max_value=100000, value=20000, step=500)

    st.divider()

    if st.button("Estimar precio", use_container_width=True):
        lugar_enc = le_places.transform([lugar])[0]
        tipo_enc = le_type.transform([tipo])[0]

        X_pred = pd.DataFrame([[metros, precio_m2, lugar_enc, tipo_enc]],
                              columns=["surface_covered_in_m2", "price_per_m2", "places_enc", "type_enc"])

        precio_estimado = dt.predict(X_pred)[0]

        st.success(f"💰 Precio estimado: **${precio_estimado:,.0f} MXN**")

        st.divider()

        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric("Precio estimado (MXN)", f"${precio_estimado:,.0f}")
        with col4:
            st.metric("Precio estimado (USD)", f"${precio_estimado / 17.5:,.0f}")
        with col5:
            st.metric("Precio por m²", f"${precio_estimado / metros:,.0f} MXN")

        st.divider()

        promedio_lugar = df_clean[df_clean["places"] == lugar]["price"].mean()
        diferencia = precio_estimado - promedio_lugar
        diferencia_pct = (diferencia / promedio_lugar) * 100

        st.markdown("#### Comparación con el mercado")
        col6, col7 = st.columns(2)
        with col6:
            st.metric(
                f"Promedio en {lugar}",
                f"${promedio_lugar:,.0f} MXN",
                delta=f"{diferencia_pct:.1f}% vs estimado"
            )
        with col7:
            if diferencia < 0:
                st.success(f"✅ El precio estimado está **${abs(diferencia):,.0f} MXN por debajo** del promedio de {lugar} — posible oportunidad de inversión.")
            else:
                st.warning(f"⚠️ El precio estimado está **${diferencia:,.0f} MXN por encima** del promedio de {lugar}.")
