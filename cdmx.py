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
seccion = st.sidebar.radio("Navegación", [
    "🏠 Inicio",
    "🗺️ Mapa",
    "🔮 Predicciones",
    "💼 Calculadora Salarial"
])

# ── Secciones ────────────────────────────────────────────────
if seccion == "🏠 Inicio":
    st.title("🏠 Análisis de Propiedades - CDMX")

    st.divider()

    # Contexto del problema
    st.subheader("📌 Contexto")
    st.markdown("""
    Tienes entre 25 y 30 años, estás saliendo de la carrera o llevas pocos años trabajando 
    y por primera vez estás pensando en serio en comprar una propiedad en la CDMX. 
    El problema es que no sabes por dónde empezar.

    Buscas en portales inmobiliarios y ves precios que van desde $800,000 hasta $15,000,000 MXN 
    en la misma alcaldía, sin entender por qué. Alguien te dice que en Benito Juárez está todo 
    caro, otro que en Iztapalapa hay oportunidades, y tu banco te aprueba un crédito que no 
    alcanza para lo que imaginabas.

    El mercado inmobiliario de la CDMX tiene un problema real: **la información está fragmentada, 
    es difícil de interpretar y favorece siempre al vendedor**. Quienes más pierden son los 
    compradores jóvenes que no tienen experiencia negociando ni acceso a datos del mercado.
    """)

    st.divider()

    # Objetivo
    st.subheader("🎯 ¿Para qué sirve este tablero?")
    st.markdown("""
    Este tablero nació para responder las preguntas que cualquier joven comprador se hace 
    antes de tomar la decisión más importante de su vida financiera:

    - **¿El precio que me están pidiendo es justo?** — o me están cobrando de más por la alcaldía
    - **¿En qué alcaldía me conviene más invertir** con el presupuesto que tengo?
    - **¿Qué tan caro es el m²** en la alcaldía que me interesa comparado con las demás?
    - **¿Existe alguna propiedad subvaluada** que nadie más ha detectado?

    Usando datos reales del mercado y un modelo de Machine Learning, este tablero te da 
    una ventaja que antes solo tenían los desarrolladores y agentes inmobiliarios con años 
    de experiencia.
    """)

    st.divider()

    # Métricas generales
    st.subheader("📊 Resumen del dataset")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total propiedades", f"{len(df):,}")
    col2.metric("Alcaldías analizadas", df["places"].nunique())
    col3.metric("Precio mediano", f"${df_clean['price'].median():,.0f} MXN")
    col4.metric("Precio promedio", f"${df_clean['price'].mean():,.0f} MXN")

    st.divider()

    # Métricas por tipo de inmueble
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

    # Qué encontrarás
    st.subheader("🗂️ ¿Qué encontrarás en este tablero?")

    col5, col6, col7 = st.columns(3)

    with col5:
        st.info("""
        **🗺️ Mapa**
        
        Mapa interactivo con la ubicación de propiedades filtrables por alcaldía, 
        tipo de inmueble y rango de precio. Incluye mapa de valuación por alcaldía.
        """)

    with col6:
        st.info("""
        **🔮 Predicciones**
        
        Herramienta interactiva — ingresa tu presupuesto, metros cuadrados y tipo 
        de inmueble para obtener el Top 3 de alcaldías que se ajustan a tus necesidades.
        """)

    with col7:
        st.info("""
        **🤖 Modelo ML**
        
        El análisis está respaldado por un Árbol de Decisión con R² de 0.89 entrenado 
        con datos reales del mercado inmobiliario de la CDMX.
        """)

    st.divider()

    # Nota metodológica
    st.subheader("📋 Nota metodológica")
    st.markdown("""
    - El dataset contiene **{:,} propiedades** después de eliminar tipos de inmueble no relevantes (locales comerciales y PH).
    - Se removieron outliers extremos (percentil 1-99) para garantizar análisis más representativos.
    - El modelo de Machine Learning fue entrenado con el **80% de los datos** y evaluado con el **20% restante**.
    - Las oportunidades de inversión se definen como propiedades cuyo precio real está entre **10% y 40% por debajo** del valor estimado por el modelo.
    """.format(len(df)))

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
    elif seccion == "💼 Calculadora Salarial":
    st.title("💼 ¿Cuánto necesito ganar para vivir en cada alcaldía?")
    st.markdown("""
    Un profesionista recién egresado debería destinar **máximo el 30% de su ingreso mensual** 
    a vivienda. Ingresa tu salario y te decimos en qué alcaldías puedes comprar sin comprometer tu economía.
    """)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        salario = st.number_input(
            "💰 Salario mensual neto (MXN)",
            min_value=5000,
            max_value=200000,
            value=20000,
            step=1000,
            format="%d"
        )
    with col2:
        plazo = st.selectbox(
            "📅 Plazo del crédito hipotecario",
            options=[10, 15, 20, 25, 30],
            index=2,
            format_func=lambda x: f"{x} años"
        )
        tasa = st.number_input(
            "📈 Tasa de interés anual (%)",
            min_value=1.0,
            max_value=20.0,
            value=10.5,
            step=0.1
        )

    st.divider()

    # Cálculo
    mensualidad_max = salario * 0.30
    tasa_mensual = (tasa / 100) / 12
    meses = plazo * 12
    # Fórmula de crédito hipotecario
    if tasa_mensual > 0:
        credito_maximo = mensualidad_max * ((1 - (1 + tasa_mensual) ** -meses) / tasa_mensual)
    else:
        credito_maximo = mensualidad_max * meses

    st.subheader("📊 Tu capacidad de compra")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mensualidad máxima (30%)", f"${mensualidad_max:,.0f} MXN")
    c2.metric("Crédito máximo estimado", f"${credito_maximo:,.0f} MXN")
    c3.metric("Plazo seleccionado", f"{plazo} años")

    st.divider()

    # Clasificar alcaldías
    st.subheader("🏙️ ¿En qué alcaldías puedes comprar?")
    st.markdown(f"Con un crédito estimado de **${credito_maximo:,.0f} MXN** basado en tu salario:")

    resumen_alcaldias = df_clean.groupby("places")["price"].agg(
        precio_mediano="median",
        precio_minimo=lambda x: x.quantile(0.10),
        propiedades_accesibles=lambda x: (x <= credito_maximo).sum()
    ).reset_index()

    def clasificar(row):
        if row["precio_mediano"] <= credito_maximo:
            return "✅ Alcanzable"
        elif row["precio_minimo"] <= credito_maximo:
            return "⚠️ Parcialmente alcanzable"
        else:
            return "❌ Fuera de rango"

    resumen_alcaldias["estado"] = resumen_alcaldias.apply(clasificar, axis=1)
    resumen_alcaldias = resumen_alcaldias.sort_values("precio_mediano")

    # Gráfica
    colores_barra = []
    for _, row in resumen_alcaldias.iterrows():
        if row["estado"] == "✅ Alcanzable":
            colores_barra.append("green")
        elif row["estado"] == "⚠️ Parcialmente alcanzable":
            colores_barra.append("orange")
        else:
            colores_barra.append("salmon")

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(resumen_alcaldias["places"], resumen_alcaldias["precio_mediano"],
                   color=colores_barra, edgecolor="white")
    ax.axvline(x=credito_maximo, color="blue", linestyle="--", linewidth=2,
               label=f"Tu crédito máximo: ${credito_maximo:,.0f}")
    ax.set_title("Precio mediano por alcaldía vs tu crédito máximo", fontsize=14)
    ax.set_xlabel("Precio mediano (MXN)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # Tabla resumen
    st.subheader("📋 Resumen por alcaldía")
    tabla = resumen_alcaldias[["places", "precio_mediano", "propiedades_accesibles", "estado"]].copy()
    tabla.columns = ["Alcaldía", "Precio mediano (MXN)", "Propiedades en tu rango", "Estado"]
    tabla["Precio mediano (MXN)"] = tabla["Precio mediano (MXN)"].apply(lambda x: f"${x:,.0f}")
    tabla = tabla.sort_values("Estado")
    st.dataframe(tabla, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("""
    **¿Cómo se calcula el crédito máximo?**
    
    Se usa la fórmula estándar de crédito hipotecario considerando tu mensualidad máxima 
    (30% de tu salario), la tasa de interés anual y el plazo seleccionado. 
    Es una estimación referencial — cada banco tiene sus propios criterios de aprobación.
    """)
