# 🏠 Análisis del Mercado Inmobiliario - CDMX

## ¿En qué lugar en CDMX conviene más invertir para compra-venta de inmuebles?

Este proyecto analiza el mercado inmobiliario de la Ciudad de México mediante técnicas de análisis de datos y Machine Learning, con el objetivo de identificar oportunidades de inversión y predecir precios de inmuebles.

---

## 📌 Problema

El mercado inmobiliario de la CDMX es opaco y complejo. Compradores, inversores y agentes carecen de herramientas para identificar si un precio es justo o detectar oportunidades reales de inversión. Este tablero busca democratizar el acceso a información del mercado inmobiliario mediante datos y modelos predictivos.

---

## 🎯 Objetivo

- Identificar los factores clave que determinan el precio de un inmueble
- Detectar propiedades subvaluadas que representen oportunidades de inversión
- Visualizar la distribución geográfica de precios y plusvalía por alcaldía
- Proveer una herramienta de predicción de precios basada en datos reales

---

## 📊 Dataset

- **Fuente:** Dataset de propiedades en venta en la CDMX
- **Total de registros:** 18,234 propiedades
- **Variables principales:**

| Variable | Descripción | Tipo |
|---|---|---|
| `price` | Precio en MXN | Cuantitativa continua |
| `places` | Alcaldía | Cualitativa nominal |
| `property_type` | Tipo de inmueble | Cualitativa nominal |
| `surface_covered_in_m2` | Superficie cubierta | Cuantitativa continua |
| `price_per_m2` | Precio por metro cuadrado | Cuantitativa continua |
| `lat-lon` | Coordenadas geográficas | Cuantitativa continua |

---

## 🗂️ Estructura del proyecto
📁 proyecto-inteligencia-de-negocios-2026/
├── cdmx.py                    # Aplicación principal de Streamlit
├── housing_data_CDMX.csv      # Dataset de propiedades
├── modelo_arbol.pkl           # Modelo de Árbol de Decisión entrenado
├── encoder_places.pkl         # Encoder de alcaldías
├── encoder_type.pkl           # Encoder de tipo de inmueble
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Documentación del proyecto
---

## 🤖 Modelo de Machine Learning

Se entrenaron y compararon dos modelos:

| Modelo | R² | MAE |
|---|---|---|
| Regresión Lineal | 0.6658 | $1,059,084 MXN |
| Árbol de Decisión | 0.8887 | $456,307 MXN |

### ¿Por qué el Árbol de Decisión?

El Árbol de Decisión supera a la Regresión Lineal porque el precio de un inmueble no depende de una sola fórmula fija — depende de la combinación de varios factores. El árbol aprende estas combinaciones y predice con mayor precisión.

**Hiperparámetros utilizados:**
- `max_depth = 6`
- `min_samples_split = 100`
- `min_samples_leaf = 50`

---

## 💰 Principales hallazgos

- **Benito Juárez** es el lugar con mayor volumen de oportunidades de inversión con 770 propiedades subvaluadas
- **Tláhuac** y **Cuajimalpa** ofrecen los mayores descuentos promedio con 22.9% y 22.2% respectivamente
- **Iztapalapa** representa el mejor balance entre volumen y descuento con 392 oportunidades y 20.8% de descuento promedio
- Los lugares del poniente y sur concentran los precios más altos por m² — **Miguel Hidalgo** y **Cuajimalpa** lideran en plusvalía

---

## 🗺️ Secciones del tablero

- **📊 Gráficas** — Distribución de precios, volumen de oferta y análisis por tipo de inmueble
- **🗺️ Mapa** — Mapa interactivo de propiedades y mapa de calor de plusvalía por alcaldía
- **🤖 Modelo ML** — Comparación de modelos y análisis de correlaciones
- **📝 Observaciones** — Hallazgos principales y recomendaciones de inversión
- **🔮 Predicciones** — Herramienta para estimar el precio de cualquier inmueble

---

## 🛠️ Tecnologías utilizadas

- **Python** — Lenguaje principal
- **Pandas** — Manipulación de datos
- **Matplotlib / Seaborn** — Visualización de datos
- **Scikit-learn** — Machine Learning
- **Folium** — Mapas interactivos
- **Streamlit** — Tablero interactivo
- **Joblib** — Serialización del modelo

---

## ▶️ Cómo ejecutar el proyecto

1. Clona el repositorio:
```bash
git clone https://github.com/tu_usuario/proyecto-inteligencia-de-negocios-2026.git
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecuta la aplicación:
```bash
streamlit run cdmx.py
```

---

## 🌐 Demo en línea

Puedes acceder al tablero en línea aquí:  
👉 [Ver tablero en Streamlit](https://tu-app.streamlit.app)

---

## 👥 Autores
0245536 Monserrath Sánchez
0286935 Yahir Aranda
0223985 Amir  Daoud
