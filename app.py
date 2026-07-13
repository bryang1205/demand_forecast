import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, date

# ── Configuración de página ───────────────────────────────
st.set_page_config(
    page_title='Previsión de Demanda',
    page_icon='📦',
    layout='wide'
)

# ── Cargar artefactos ─────────────────────────────────────
@st.cache_resource
def load_pipeline():
    model      = joblib.load('xgb_final.pkl')
    scaler     = joblib.load('scaler.pkl')
    le_store   = joblib.load('le_store.pkl')
    le_product = joblib.load('le_product.pkl')
    with open('model_metadata.json') as f:
        metadata = json.load(f)
    return model, scaler, le_store, le_product, metadata

model, scaler, le_store, le_product, metadata = load_pipeline()

# ── Función de predicción ─────────────────────────────────
def predict_demand(inputs: dict) -> int:
    df = pd.DataFrame([inputs])

    # Encoding
    df['Store_ID_enc']   = le_store.transform(df['Store_ID'])
    df['Product_ID_enc'] = le_product.transform(df['Product_ID'])

    # OHE
    ohe_cols = ['Seasonality', 'Weather_Condition', 'Category', 'Region']
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # Escalado con las 4 columnas exactas del entrenamiento
    scale_cols_fit = ['Units_Sold', 'Units_Ordered', 'Price', 'Discount']
    df['Units_Sold'] = 88  # media histórica — no disponible en producción
    df[scale_cols_fit] = scaler.transform(df[scale_cols_fit])

    # Features temporales
    fecha = pd.to_datetime(inputs['Date'])
    df['Year']          = fecha.year
    df['Month']         = fecha.month
    df['Week']          = fecha.isocalendar()[1]
    df['DayOfWeek']     = fecha.dayofweek
    df['DayOfYear']     = fecha.timetuple().tm_yday
    df['Quarter']       = fecha.quarter
    df['IsWeekend']     = int(fecha.dayofweek >= 5)
    df['Month_sin']     = np.sin(2 * np.pi * fecha.month / 12)
    df['Month_cos']     = np.cos(2 * np.pi * fecha.month / 12)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * fecha.dayofweek / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * fecha.dayofweek / 7)

    # Alinear columnas con las del entrenamiento
    for col in metadata['features']:
        if col not in df.columns:
            df[col] = 0
    df = df[metadata['features']]

    # Convertir bool → int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    pred = model.predict(df)[0]
    return max(0, int(round(pred)))


# ══════════════════════════════════════════════════════════
# INTERFAZ
# ══════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────
st.title('📦 Previsión de Demanda — Grupo 09')
st.caption(f"Modelo: {metadata['model_name']} v{metadata['model_version']}  |  "
           f"WAPE test: {metadata['wape_test']}%  |  "
           f"KPI: {metadata['kpi_objetivo']}")
st.divider()

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.header('⚙️ Parámetros de Entrada')

with st.sidebar:
    st.subheader('🏪 Identificación')
    store_id   = st.selectbox('Tienda',   ['S001','S002','S003','S004','S005'])
    product_id = st.selectbox('Producto', [f'P{str(i).zfill(4)}' for i in range(1, 21)])
    fecha      = st.date_input('Fecha', value=date(2024, 2, 1))

    st.subheader('📦 Inventario y Operaciones')
    inventory  = st.number_input('Nivel de Inventario', min_value=0, value=500)
    units_ord  = st.number_input('Unidades Ordenadas',  min_value=0, value=89)
    price      = st.number_input('Precio (S/.)',         min_value=0.0, value=67.73)
    discount   = st.slider('Descuento (%)', 0, 50, 10)
    comp_price = st.number_input('Precio Competencia (S/.)', min_value=0.0, value=69.45)

    st.subheader('🌤️ Contexto')
    category  = st.selectbox('Categoría',      ['Electronics','Groceries','Clothing','Furniture','Toys'])
    region    = st.selectbox('Región',         ['North','South','East','West'])
    season    = st.selectbox('Estacionalidad', ['Spring','Summer','Autumn','Winter'])
    weather   = st.selectbox('Clima',          ['Sunny','Rainy','Cloudy','Snowy'])
    promotion = st.toggle('Promoción activa',   value=False)
    epidemic  = st.toggle('Contexto epidémico', value=False)

    st.subheader('📈 Historial reciente')
    lag1  = st.number_input('Demanda ayer (Lag 1)',  min_value=0, value=100)
    lag7  = st.number_input('Demanda hace 7 días',   min_value=0, value=95)
    lag14 = st.number_input('Demanda hace 14 días',  min_value=0, value=98)
    lag28 = st.number_input('Demanda hace 28 días',  min_value=0, value=102)
    rm7   = st.number_input('Media móvil 7 días',    min_value=0.0, value=97.0)
    rm14  = st.number_input('Media móvil 14 días',   min_value=0.0, value=99.0)
    rm28  = st.number_input('Media móvil 28 días',   min_value=0.0, value=101.0)

# ── Botón de predicción ───────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predecir = st.button('🔮 Predecir Demanda', use_container_width=True, type='primary')

# ── Resultado ─────────────────────────────────────────────
if predecir:
    inputs = {
        'Date'              : str(fecha),
        'Store_ID'          : store_id,
        'Product_ID'        : product_id,
        'Category'          : category,
        'Region'            : region,
        'Inventory_Level'   : inventory,
        'Units_Ordered'     : units_ord,
        'Price'             : price,
        'Discount'          : discount,
        'Competitor_Pricing': comp_price,
        'Weather_Condition' : weather,
        'Promotion'         : int(promotion),
        'Seasonality'       : season,
        'Epidemic'          : int(epidemic),
        'Lag_1'             : lag1,
        'Lag_7'             : lag7,
        'Lag_14'            : lag14,
        'Lag_28'            : lag28,
        'Rolling_Mean_7'    : rm7,
        'Rolling_Mean_14'   : rm14,
        'Rolling_Mean_28'   : rm28,
        'Price_Ratio'       : price / (comp_price + 1e-9),
        'Stock_to_Sales'    : inventory / (lag1 + 1e-9),
    }

    with st.spinner('Calculando predicción...'):
        demanda_pred = predict_demand(inputs)

    st.divider()

    # Métricas principales
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric('📦 Demanda Predicha', f'{demanda_pred} unidades')
    with col_b:
        st.metric('🏪 Tienda / Producto', f'{store_id} / {product_id}')
    with col_c:
        st.metric('📅 Fecha', str(fecha))

    st.divider()

    # Detalle
    st.subheader('📊 Contexto de la Predicción')
    col_d, col_e = st.columns(2)

    with col_d:
        st.markdown('**Parámetros utilizados:**')
        st.dataframe(pd.DataFrame({
            'Variable': ['Categoría','Región','Estacionalidad','Clima',
                         'Promoción','Epidemia','Precio','Descuento'],
            'Valor'   : [category, region, season, weather,
                         '✅ Sí' if promotion else '❌ No',
                         '✅ Sí' if epidemic  else '❌ No',
                         f'S/. {price:.2f}', f'{discount}%']
        }), hide_index=True, use_container_width=True)

    with col_e:
        st.markdown('**Historial reciente:**')
        st.dataframe(pd.DataFrame({
            'Período' : ['Hace 28 días','Hace 14 días','Hace 7 días','Ayer'],
            'Demanda' : [lag28, lag14, lag7, lag1],
        }), hide_index=True, use_container_width=True)

        st.line_chart(pd.DataFrame({
            'Demanda': [lag28, lag14, lag7, lag1, demanda_pred]
        }), height=150)
        st.caption('← historial | predicción →')

    # Info del modelo
    with st.expander('ℹ️ Información del modelo'):
        col_f, col_g, col_h = st.columns(3)
        col_f.metric('WAPE test', f"{metadata['wape_test']}%")
        col_g.metric('RMSE test', f"{metadata['rmse_test']} uds")
        col_h.metric('MAE test',  f"{metadata['mae_test']} uds")
        st.caption(f"Entrenado: {metadata['train_period']}  |  "
                   f"Versión: {metadata['model_version']}  |  "
                   f"Creado: {metadata['created_at']}")