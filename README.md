# 📦 Demand Forecast — Previsión de Demanda con XGBoost

Aplicación de Machine Learning para predecir la demanda diaria de productos por tienda, desarrollada con **XGBoost** e interfaz interactiva construida con **Streamlit**.

---

## 🚀 Demo en vivo

Puedes probar la aplicación desplegada sin instalar nada:

🔗 **[demand-forecast-grupo09.streamlit.app](https://demand-forecast-grupo09.streamlit.app)**

---

## 📁 Estructura del repositorio

```
demand_forecast/
├── app.py                  # Aplicación Streamlit (interfaz principal)
├── xgb_final.pkl           # Modelo XGBoost entrenado
├── scaler.pkl              # Escalador StandardScaler
├── le_store.pkl            # Label Encoder para Store_ID
├── le_product.pkl          # Label Encoder para Product_ID
├── model_metadata.json     # Metadatos del modelo (features, versión)
├── requirements.txt        # Dependencias del proyecto
├── runtime.txt             # Versión de Python requerida
└── .python-version         # Versión de Python (Streamlit Cloud)
```

---

## ⚙️ Requisitos previos

Antes de ejecutar el proyecto localmente, asegúrate de tener instalado:

- **Python 3.10** — [Descargar aquí](https://www.python.org/downloads/release/python-31011/)
- **Git** — [Descargar aquí](https://git-scm.com/downloads)

Para verificar que tienes la versión correcta de Python:

```bash
python --version
# Debe mostrar: Python 3.10.x
```

---

## 🛠️ Instalación y ejecución local

### 1. Clonar el repositorio

```bash
git clone https://github.com/bryang1205/demand_forecast.git
cd demand_forecast
```

### 2. Instalar las dependencias

```bash
pip install -r requirements.txt
```

> ⚠️ **Nota:** Si tienes otros proyectos Python en tu máquina, se recomienda usar un entorno virtual para evitar conflictos de versiones:
> ```bash
> python -m venv venv
> # Windows:
> venv\Scripts\activate
> # Mac/Linux:
> source venv/bin/activate
>
> pip install -r requirements.txt
> ```

### 3. Ejecutar la aplicación

```bash
streamlit run app.py
```

La app se abrirá automáticamente en tu navegador en:

```
http://localhost:8501
```

---

## 🔮 Cómo usar la aplicación

Una vez abierta la app, sigue estos pasos:

**1.** En el panel lateral izquierda (*sidebar*), configura los parámetros de la predicción:

| Parámetro | Descripción |
|---|---|
| Store ID | Identificador de la tienda (ej. S001) |
| Product ID | Identificador del producto (ej. P0001) |
| Fecha | Fecha para la que se quiere predecir |
| Categoría / Región | Clasificación del producto y ubicación |
| Precio / Descuento | Precio unitario y porcentaje de descuento |
| Precio Competencia | Precio del competidor más cercano |
| Nivel de Inventario | Stock disponible actual |
| Promoción / Epidemia | Variables de contexto (activar/desactivar) |
| Lags (1, 7, 14, 28) | Ventas reales de días anteriores |
| Rolling Mean (7, 14, 28) | Promedio móvil de ventas recientes |

**2.** Presiona el botón **🔮 Predecir Demanda**.

**3.** El modelo devuelve la **demanda estimada en unidades** para esa combinación de tienda, producto y fecha.

### Ejemplo de prueba rápida

Si no tienes datos reales a mano, usa estos valores para verificar que todo funciona:

| Campo | Valor |
|---|---|
| Store ID | S003 |
| Product ID | P0007 |
| Fecha | 2024-01-15 |
| Categoría | Clothing |
| Región | South |
| Estacionalidad | Winter |
| Clima | Rainy |
| Precio | 45.50 |
| Descuento | 15 |
| Precio Competencia | 48.00 |
| Inventario | 320 |
| Promoción | ✅ Activado |
| Epidemia | ❌ Desactivado |
| Lag 1 / 7 / 14 / 28 | 85 / 90 / 88 / 92 |
| Rolling Mean 7 / 14 / 28 | 87.5 / 89.0 / 91.0 |

---

## 📦 Dependencias principales

| Librería | Versión | Uso |
|---|---|---|
| streamlit | 1.45.0 | Interfaz web interactiva |
| xgboost | 2.1.3 | Modelo de predicción |
| scikit-learn | 1.6.1 | Preprocesamiento (scaler, encoders) |
| pandas | 2.2.3 | Manipulación de datos |
| numpy | 1.26.4 | Operaciones numéricas |
| joblib | 1.4.2 | Carga de artefactos serializados |
| starlette | 0.41.3 | Dependencia de Streamlit |

---

## 🧠 Sobre el modelo

- **Algoritmo:** XGBoost Regressor
- **Versión:** 1.0.0
- **Optimización:** Búsqueda bayesiana de hiperparámetros con Optuna (50 trials)
- **Features:** 29 variables incluyendo lags temporales, rolling means, variables cíclicas (seno/coseno), ratios derivados y one-hot encoding de variables categóricas
- **Métrica objetivo:** WAPE (Weighted Absolute Percentage Error)

---

## 🐛 Problemas frecuentes

**`ModuleNotFoundError: No module named 'joblib'`**
→ Asegúrate de haber corrido `pip install -r requirements.txt` y de estar usando Python 3.10.

**`ImportError: cannot import name 'DEFAULT_EXCLUDED_CONTENT_TYPES'`**
→ Conflicto entre versiones de `starlette`. Solución:
```bash
pip install "streamlit==1.45.0" "starlette==0.41.3"
```

**La app abre pero la predicción da error**
→ Verifica que todos los archivos `.pkl` y `model_metadata.json` estén en la misma carpeta que `app.py`.

---

## 👥 Autores

Grupo 09 — Machine Learning  
Universidad Privada Antenor Orrego (UPAO)  
Ingeniería de Sistemas e Inteligencia Artificial — 2026
