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
    "📍 Resumen por Alcaldía",
    "🗺️ Mapa",
    "🔮 Predicciones"
])

# ── Secciones ────────────────────────────────────────────────
if seccion == "🏠 Inicio":
    st.title("🏠 Análisis de Propiedades - CDMX")

    st.divider()

    st.subheader("🎯 ¿Para qué sirve este tablero?")
    st.markdown("""
    Este tablero está pensado para resolver dudas muy concretas a la hora de buscar dónde comprar:

    - ¿Cuánto cuesta en promedio vivir en cada alcaldía y cómo se compara con las demás?
    - ¿Qué alcaldías se ajustan a mi presupuesto y si tengo un rango de precio definido?
    - ¿Dónde están ubicadas las propiedades disponibles y cómo varían los precios geográficamente?
    - Con mi presupuesto, metros cuadrados deseados y tipo de inmueble, ¿cuáles son mis 3 mejores opciones de alcaldía?

    Usando datos reales del mercado y un modelo de Machine Learning, este tablero te permite 
    explorar el mercado inmobiliario de la CDMX de forma simple y tomar una decisión 
    más informada.
    """)

    st.divider()

    st.subheader("📊 Resumen del dataset")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total propiedades", f"{len(df):,}")
    col2.metric("Alcaldías analizadas", df["places"].nunique())
    col3.metric("Precio mediano", f"${df_clean['price'].median():,.0f} MXN")
    col4.metric("Precio promedio", f"${df_clean['price'].mean():,.0f} MXN")

    st.divider()

    st.subheader("🏠 Precios por tipo de inmueble")
    df_apt = df_clean[df_clean["property_type"] == "apartment"]
    df_house = df_clean[df_clean["property_type"] == "house"]

    st.markdown("#### 🏢 Departamentos")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total", f"{len(df_apt):,}")
    a2.metric("Mediana", f"${df_apt['price'].median():,.0f} MXN")
    a3.metric("Precio promedio", f"${df_apt['price'].mean():,.0f} MXN")
    a4.metric("Precio por m²", f"${df_apt['price_per_m2'].mean():,.0f} MXN")

    st.markdown("")

    st.markdown("#### 🏡 Casas")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Total", f"{len(df_house):,}")
    h2.metric("Mediana", f"${df_house['price'].median():,.0f} MXN")
    h3.metric("Precio promedio", f"${df_house['price'].mean():,.0f} MXN")
    h4.metric("Precio por m²", f"${df_house['price_per_m2'].mean():,.0f} MXN")

    st.divider()

    st.subheader("🗂️ ¿Qué encontrarás en este tablero?")
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.info("""
        **📍 Resumen por Alcaldía**
        
        Selecciona una alcaldía y ve sus estadísticas clave — precios, m², oportunidades 
        y distribución del mercado.
        """)
    with col6:
        st.info("""
        **🗺️ Mapa**
        
        Mapa interactivo con la ubicación de propiedades filtrables por alcaldía, 
        tipo de inmueble y rango de precio.
        """)
    with col7:
        st.info("""
        **🔮 Predicciones**
        
        Ingresa tu presupuesto, metros cuadrados y tipo de inmueble para obtener 
        el Top 3 de alcaldías que se ajustan a tus necesidades.
        """)
    with col8:
        st.info("""
        **🤖 Modelo ML**
        
        El análisis está respaldado por un Árbol de Decisión con R² de 0.89 entrenado 
        con datos reales del mercado inmobiliario de la CDMX.
        """)

elif seccion == "📍 Resumen por Alcaldía":
    st.title("📍 Resumen por Alcaldía")
    st.markdown("Selecciona una alcaldía y ve sus estadísticas clave del mercado inmobiliario.")

    st.divider()

    alcaldia_sel = st.selectbox("Selecciona una alcaldía", options=sorted(df_clean["places"].unique()))

    df_sel = df_clean[df_clean["places"] == alcaldia_sel]
    df_sel_apt = df_sel[df_sel["property_type"] == "apartment"]
    df_sel_house = df_sel[df_sel["property_type"] == "house"]

    st.divider()

    st.subheader(f"📊 {alcaldia_sel} — Panorama general")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total propiedades", f"{len(df_sel):,}")
    c2.metric("Precio promedio", f"${df_sel['price'].mean():,.0f} MXN")
    c3.metric("Precio mediano", f"${df_sel['price'].median():,.0f} MXN")
    c4.metric("Precio por m² promedio", f"${df_sel['price_per_m2'].mean():,.0f} MXN")

    st.divider()

    st.subheader("🏠 Por tipo de inmueble")

    st.markdown("#### 🏢 Departamentos")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total", f"{len(df_sel_apt):,}")
    a2.metric("Precio promedio", f"${df_sel_apt['price'].mean():,.0f} MXN")
    a3.metric("Precio mediano", f"${df_sel_apt['price'].median():,.0f} MXN")
    a4.metric("m² promedio", f"{df_sel_apt['surface_covered_in_m2'].mean():,.0f} m²")

    st.markdown("")

    st.markdown("#### 🏡 Casas")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Total", f"{len(df_sel_house):,}")
    h2.metric("Precio promedio", f"${df_sel_house['price'].mean():,.0f} MXN")
    h3.metric("Precio mediano", f"${df_sel_house['price'].median():,.0f} MXN")
    h4.metric("m² promedio", f"{df_sel_house['surface_covered_in_m2'].mean():,.0f} m²")

    st.divider()

    st.subheader("💰 Rango de precios")
    r1, r2, r3 = st.columns(3)
    r1.metric("Precio mínimo (10%)", f"${df_sel['price'].quantile(0.10):,.0f} MXN")
    r2.metric("Precio mediano (50%)", f"${df_sel['price'].quantile(0.50):,.0f} MXN")
    r3.metric("Precio máximo (90%)", f"${df_sel['price'].quantile(0.90):,.0f} MXN")

    st.divider()

    oport_alcaldia = df_model2[
        (df_model2["places"] == alcaldia_sel) &
        (df_model2["diferencia_pct"] < -10) &
        (df_model2["diferencia_pct"] > -40)
    ]

    st.subheader("🎯 Oportunidades detectadas por el modelo")
    o1, o2 = st.columns(2)
    o1.metric("Propiedades subvaluadas", f"{len(oport_alcaldia):,}")
    o2.metric("Descuento promedio", f"{oport_alcaldia['diferencia_pct'].abs().mean():.1f}%")

    st.divider()

    st.subheader("📈 Distribución de precios")
    fig, ax = plt.subplots(figsize=(14, 4))
    for tipo_p, color in [("apartment", "steelblue"), ("house", "salmon")]:
        datos = df_sel[df_sel["property_type"] == tipo_p]["price"]
        if len(datos) > 0:
            ax.hist(datos, bins=30, alpha=0.6, color=color, edgecolor="white",
                    label="Departamento" if tipo_p == "apartment" else "Casa")
    ax.set_xlabel("Precio (MXN)")
    ax.set_ylabel("Número de propiedades")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlim(df_sel["price"].quantile(0.05), df_sel["price"].quantile(0.95))
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

elif seccion == "🗺️ Mapa":
    st.title("🗺️ Mapa")

    tab1, tab2 = st.tabs(["📍 Propiedades", "🌡️ Valuación por alcaldía"])

    with tab1:
        st.markdown("Visualización de propiedades en el mapa. Haz clic en cada punto para ver el precio y detalles.")

        col1, col2 = st.columns(2)
        with col1:
            alcaldia_filtro = st.multiselect("Filtrar por alcaldía",
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
            (df["places"].isin(alcaldia_filtro)) &
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
            tooltip=folium.GeoJsonTooltip(fields=["nomgeo"], aliases=["Alcaldía:"])
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
                    f"<b>Alcaldía:</b> {row['places']}<br>"
                    f"<b>Tipo:</b> {row['property_type']}<br>"
                    f"<b>Superficie:</b> {row['surface_covered_in_m2']} m²",
                    max_width=200
                )
            ).add_to(mapa)

        st_folium(mapa, width=1200, height=500)

    with tab2:
        st.markdown("Alcaldías coloreadas por **precio promedio de las propiedades**. Entre más rojo, mayor es el precio promedio en esa alcaldía.")

        st.markdown("""
        ℹ️ Este mapa muestra qué tan caro es comprar en cada alcaldía en promedio. 
        No representa plusvalía (que requeriría datos históricos de tiempo), 
        sino el **nivel de precios actual** — útil para identificar alcaldías accesibles vs alcaldías de alto valor.
        """)

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

        valuacion = df_clean.groupby("places")["price"].mean().reset_index()
        valuacion.columns = ["places", "precio_promedio"]

        geojson_valuacion = copy.deepcopy(cdmx_geojson)

        for feature in geojson_valuacion["features"]:
            nombre_geo = feature["properties"]["nomgeo"]
            nombre_dataset = mapeo.get(nombre_geo, nombre_geo.replace(" ", ""))
            match = valuacion[valuacion["places"] == nombre_dataset]
            feature["properties"]["precio_promedio"] = int(match["precio_promedio"].values[0]) if not match.empty else 0

        precio_max_map = valuacion["precio_promedio"].max()
        precio_min_map = valuacion["precio_promedio"].min()

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

        mapa_valuacion = folium.Map(location=[19.43, -99.13], zoom_start=11)

        folium.GeoJson(
            geojson_valuacion,
            style_function=lambda x: {
                "fillColor": get_color(x["properties"]["precio_promedio"]),
                "color": "gray",
                "weight": 1.5,
                "fillOpacity": 0.7
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["nomgeo", "precio_promedio"],
                aliases=["Alcaldía:", "Precio promedio (MXN):"],
                localize=True
            )
        ).add_to(mapa_valuacion)

        st_folium(mapa_valuacion, width=1200, height=500)

        st.divider()

        st.subheader("📊 Ranking de valuación por alcaldía")
        valuacion_sorted = valuacion.sort_values("precio_promedio", ascending=False).reset_index(drop=True)
        valuacion_sorted.index += 1
        valuacion_sorted.columns = ["Alcaldía", "Precio promedio (MXN)"]
        valuacion_sorted["Precio promedio (MXN)"] = valuacion_sorted["Precio promedio (MXN)"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(valuacion_sorted, use_container_width=True)

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

            resultados = sorted(resultados, key=lambda x: (x["oportunidades"], x["props_disponibles"]), reverse=True)

            if len(resultados) == 0:
                st.warning("No encontramos alcaldías que se ajusten a tu presupuesto y preferencias. Intenta ampliar tu rango de presupuesto o ajustar los metros cuadrados.")
            else:
                top3 = resultados[:3]
                st.subheader("🏆 Top 3 mejores alcaldías para ti")
                st.markdown(f"Resultados para: **{'Departamento' if tipo == 'apartment' else 'Casa'}** de **{metros} m²** con presupuesto de **${presupuesto_min:,.0f} — ${presupuesto_max:,.0f} MXN**")

                st.divider()

                medallas = ["🥇", "🥈", "🥉"]
                colores_card = ["success", "warning", "info"]

                for i, res in enumerate(top3):
                    if colores_card[i] == "success":
                        st.success(f"{medallas[i]} **{res['lugar']}**")
                    elif colores_card[i] == "warning":
                        st.warning(f"{medallas[i]} **{res['lugar']}**")
                    else:
                        st.info(f"{medallas[i]} **{res['lugar']}**")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Precio estimado", f"${res['precio_estimado']:,.0f} MXN")
                    c2.metric("Precio promedio real", f"${res['precio_real_promedio']:,.0f} MXN")
                    c3.metric("Propiedades disponibles", f"{res['props_disponibles']:,}")
                    c4.metric("Oportunidades subvaluadas", f"{res['oportunidades']:,}")

                    st.markdown(f"📐 Precio por m² promedio en **{res['lugar']}**: **${res['precio_m2']:,.0f} MXN**")
                    st.divider()

                if len(resultados) > 3:
                    st.markdown(f"*También encontramos {len(resultados) - 3} alcaldías adicionales que se ajustan a tu presupuesto.*")
