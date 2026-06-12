# 🏠 Análisis de Propiedades - CDMX

## ¿En qué alcaldía conviene más comprar una propiedad en la CDMX?

Este proyecto analiza el mercado inmobiliario de la Ciudad de México mediante análisis de datos y Machine Learning, pensado para jóvenes profesionistas entre 23 y 30 años que están considerando comprar su primera propiedad.

---

## 📌 Contexto

Si tienes entre 23 y 30 años, estás saliendo de la carrera o llevas pocos años trabajando, y por primera vez estás pensando en comprar una propiedad en la CDMX, seguramente te has encontrado con varios problemas:

- **Precios sin sentido** — al buscar en portales encuentras rangos de precio muy alejados dentro de la misma alcaldía, sin forma de saber si lo que te ofrecen es justo.
- **Información fragmentada** — hay portales y recomendaciones por todos lados, pero ninguno te da un panorama claro de qué zonas se ajustan a tu presupuesto, tu trabajo y tu estilo de vida al mismo tiempo.
- **El vendedor siempre lleva ventaja** — quien vende lleva años en el mercado y sabe exactamente cuánto vale su propiedad y hasta dónde puede negociar. El comprador llega sin ningún punto de referencia.
- **Presupuesto limitado** — un profesionista recién egresado destina entre el 25% y 30% de su ingreso mensual a vivienda, y con eso las zonas accesibles no siempre coinciden con las que necesita por ubicación, seguridad o calidad de vida.
- **Los precios ya no reflejan la realidad del comprador local** — a partir de la llegada de trabajadores extranjeros que perciben ingresos en dólares se generó un alza sostenida en zonas como Roma, Condesa y Narvarte. El comprador mexicano recién egresado compite en un mercado que ya no está calibrado para su nivel de ingreso.

---

## 🎯 ¿Para qué sirve este tablero?

Este tablero está pensado para resolver dudas muy concretas a la hora de buscar dónde comprar:

- ¿Cuánto cuesta en promedio vivir en cada alcaldía y cómo se compara con las demás?
- ¿Qué alcaldías se ajustan a mi presupuesto si tengo un rango de precio definido?
- ¿Dónde están ubicadas las propiedades disponibles y cómo varían los precios geográficamente?
- Con mi presupuesto, metros cuadrados deseados y tipo de inmueble, ¿cuáles son mis 3 mejores opciones de alcaldía?

---

## 📊 Dataset

- **Fuente:** Dataset de propiedades en venta en la CDMX
- **Registros después de limpieza:** 18,053 propiedades
- **Variables principales:**

| Variable | Descripción | Tipo |
|---|---|---|
| `price` | Precio en MXN | Cuantitativa continua |
| `places` | Alcaldía | Cualitativa nominal |
| `property_type` | Tipo de inmueble (casa/departamento) | Cualitativa nominal |
| `surface_covered_in_m2` | Superficie cubierta | Cuantitativa continua |
| `price_per_m2` | Precio por metro cuadrado | Cuantitativa continua |
| `lat-lon` | Coordenadas geográficas | Cuantitativa continua |

---

## 🧹 Limpieza de datos

1. **Filtrado de tipos de inmueble** — se eliminaron `store` y `PH`, quedando solo `apartment` y `house`.
2. **Separación de coordenadas** — la columna `lat-lon` se dividió en `lat` y `lon`, corrigiendo inconsistencias de separadores (`:` vs `,`).
3. **Eliminación de outliers** — se removió el percentil 1-99 de `price` y `surface_covered_in_m2` para evitar distorsiones en correlaciones y modelo.


---

## 🗂️ Estructura del proyecto
```
📁 proyecto-inteligencia-de-negocios-2026/
├── cdmx.py                    # Aplicación principal de Streamlit
├── housing_data_CDMX.csv      # Dataset de propiedades
├── modelo_arbol.pkl           # Modelo de Árbol de Decisión entrenado
├── encoder_places.pkl         # Encoder de alcaldías
├── encoder_type.pkl           # Encoder de tipo de inmueble
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Documentación del proyecto
```
---

## 🤖 Modelo de Machine Learning

Se entrenaron y compararon dos modelos:

| Modelo | R² | MAE |
|---|---|---|
| Regresión Lineal | 0.6658 | $1,059,084 MXN |
| Árbol de Decisión | 0.8887 | $456,307 MXN |

### ¿Por qué el Árbol de Decisión?

El precio de un inmueble no depende de una sola fórmula fija — depende de la combinación de varios factores. Un departamento de 100 m² en Miguel Hidalgo tiene un valor muy diferente al mismo departamento en Iztapalapa. El Árbol de Decisión aprende estas combinaciones y predice con mayor precisión.

**Variables utilizadas:**
- `surface_covered_in_m2` — metros cuadrados
- `price_per_m2` — precio por m²
- `places` — alcaldía
- `property_type` — tipo de inmueble

**Hiperparámetros utilizados:**
- `max_depth = 6`
- `min_samples_split = 100`
- `min_samples_leaf = 50`

Estos parámetros limitan la complejidad del árbol para que aprenda patrones generales del mercado en lugar de memorizar casos individuales (overfitting).

---

## 🗺️ Secciones del tablero

- **🏠 Inicio** — Contexto del proyecto, objetivo y resumen general del dataset.
- **📍 Resumen por Alcaldía** — Selecciona una alcaldía y consulta sus estadísticas clave: precios, metros cuadrados, oportunidades y distribución del mercado.
- **🗺️ Mapa** — Mapa interactivo de propiedades filtrable por alcaldía, tipo de inmueble y precio, además de un mapa de valuación promedio por alcaldía.
- **🔮 Predicciones** — Ingresa tu presupuesto, metros cuadrados deseados y tipo de inmueble, y obtén el Top 3 de alcaldías que mejor se ajustan a tus necesidades.

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

👉 [Ver tablero en Streamlit](https://proyecto-inteligencia-de-negocios2026.streamlit.app/)

---



## 👥 Autores

- 0245536 Monserrath Sánchez

- 0286935 Yahir Aranda

