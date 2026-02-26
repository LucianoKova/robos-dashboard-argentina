# =============================
# IMPORTS
# =============================

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# =============================
# CONFIGURACIÓN GENERAL
# =============================

st.set_page_config(
    page_title="Panel Inteligencia Criminal",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Inter:wght@400;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 1px;
}

.stMetric {
    background-color: #1A1D24;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =============================
# CARGA DE DATOS
# =============================

@st.cache_data
def cargar_datos():
    df = pd.read_csv("data/raw/dnrpa-robos-recuperos-autos-202601.csv")
    df["tramite_fecha"] = pd.to_datetime(df["tramite_fecha"], errors="coerce")
    df["edad"] = 2026 - df["titular_anio_nacimiento"]
    return df

df = cargar_datos()

# =============================
# TÍTULO
# =============================

st.markdown("""
<h1 style='text-align: center; color: #00F5D4;'>
🛰 PANEL DE INTELIGENCIA CRIMINAL AUTOMOTOR
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align: center; font-size:16px;'>
Monitoreo analítico de robos y recuperos vehiculares - Argentina
</p>
""", unsafe_allow_html=True)

# =============================
# MÉTRICAS PRINCIPALES
# =============================

robos = df[df["tramite_tipo"].str.contains("ROBO", na=False)]
recuperos = df[df["tramite_tipo"].str.contains("RECUPERO", na=False)]

tasa = (len(recuperos) / len(robos) * 100) if len(robos) > 0 else 0

col1, col2, col3 = st.columns(3)

col1.markdown("### 🚨 EVENTOS DE ROBO")
col1.metric("Cantidad", len(robos))

col2.markdown("### 🔁 EVENTOS DE RECUPERO")
col2.metric("Cantidad", len(recuperos))

col3.markdown("### 📊 EFICIENCIA OPERATIVA")

if tasa >= 10:
    color = "#00FF9C"
elif tasa >= 5:
    color = "#FFC300"
else:
    color = "#FF4B4B"

col3.markdown(
    f"<h2 style='color:{color};'>{tasa:.2f} %</h2>",
    unsafe_allow_html=True
)

st.divider()

# =============================
# FILTRO PROVINCIA
# =============================

provincia = st.selectbox(
    "Seleccionar Provincia",
    ["Todas"] + sorted(df["registro_seccional_provincia"].unique())
)

df_filtrado = df if provincia == "Todas" else df[df["registro_seccional_provincia"] == provincia]

# =============================
# GRÁFICOS DESCRIPTIVOS
# =============================

st.subheader("🚗 Incidencia por Marca")
st.bar_chart(df_filtrado["automotor_marca_descripcion"].value_counts().head(10))

st.subheader("👤 Perfil Demográfico del Titular")
st.bar_chart(df_filtrado["titular_genero"].value_counts())

st.subheader("📈 Evolución Temporal")
serie = df_filtrado.groupby(df_filtrado["tramite_fecha"].dt.date).size()
st.line_chart(serie)

# =============================
# MAPA
# =============================

st.subheader("🗺 Distribución Geográfica")

coordenadas = {
    "Buenos Aires": (-34.6, -58.4),
    "Ciudad Autónoma de Buenos Aires": (-34.6, -58.4),
    "Córdoba": (-31.4, -64.2),
    "Santa Fe": (-31.6, -60.7),
    "Mendoza": (-32.9, -68.8),
}

map_df = (
    df.groupby("registro_seccional_provincia")
    .size()
    .reset_index(name="cantidad")
)

map_df["lat"] = map_df["registro_seccional_provincia"].map(lambda x: coordenadas.get(x, (None, None))[0])
map_df["lon"] = map_df["registro_seccional_provincia"].map(lambda x: coordenadas.get(x, (None, None))[1])
map_df = map_df.dropna()

fig_map = px.scatter_mapbox(
    map_df,
    lat="lat",
    lon="lon",
    size="cantidad",
    hover_name="registro_seccional_provincia",
    zoom=4,
)

fig_map.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig_map)

# =============================
# MODELO PREDICTIVO
# =============================

st.subheader("🤖 Modelo Predictivo - Recupero")

modelo_df = df.copy()

modelo_df["objetivo"] = modelo_df["tramite_tipo"].apply(
    lambda x: 1 if "RECUPERO" in x else 0
)

features = ["automotor_marca_descripcion", "registro_seccional_provincia", "edad"]
modelo_df = modelo_df[features + ["objetivo"]].dropna()

le = LabelEncoder()

for col in ["automotor_marca_descripcion", "registro_seccional_provincia"]:
    modelo_df[col] = le.fit_transform(modelo_df[col])

X = modelo_df[features]
y = modelo_df["objetivo"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

modelo = RandomForestClassifier(class_weight="balanced")
modelo.fit(X_train, y_train)

probs = modelo.predict_proba(X_test)[:, 1]
pred = (probs > 0.30).astype(int)

accuracy = accuracy_score(y_test, pred)

st.metric("Precisión del modelo", f"{accuracy:.2f}")

# Reporte
st.subheader("📊 Reporte de Clasificación")
report = classification_report(y_test, pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Matriz
st.subheader("📊 Matriz de Confusión")
cm = confusion_matrix(y_test, pred)

fig_cm, ax = plt.subplots(figsize=(5,4))
ax.set_facecolor("#0E1117")
fig_cm.patch.set_facecolor("#0E1117")
ax.imshow(cm, cmap="viridis")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Robo", "Recupero"], color="white")
ax.set_yticklabels(["Robo", "Recupero"], color="white")
ax.spines[:].set_visible(False)

st.pyplot(fig_cm)

# ROC
st.subheader("🎯 Curva ROC")
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

fig_roc, ax = plt.subplots(figsize=(6,4))
ax.set_facecolor("#0E1117")
fig_roc.patch.set_facecolor("#0E1117")
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="#00F5D4")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.legend()
ax.spines[:].set_visible(False)

st.pyplot(fig_roc)

# =============================
# INTELIGENCIA OPERATIVA REAL
# =============================

st.subheader("🚗 Tasa Real de Recupero por Marca")

marca_stats = (
    df.groupby("automotor_marca_descripcion")
    .agg(
        robos=("tramite_tipo", lambda x: (x.str.contains("ROBO")).sum()),
        recuperos=("tramite_tipo", lambda x: (x.str.contains("RECUPERO")).sum())
    )
)

marca_stats["tasa_recupero_%"] = (
    marca_stats["recuperos"] / marca_stats["robos"] * 100
)

marca_stats = marca_stats[marca_stats["robos"] > 20]
marca_stats = marca_stats.sort_values("tasa_recupero_%", ascending=False)

fig_tasa = px.bar(
    marca_stats.head(10),
    x=marca_stats.head(10).index,
    y="tasa_recupero_%",
    color="tasa_recupero_%",
    color_continuous_scale="viridis",
)

fig_tasa.update_layout(
    xaxis_title="Marca",
    yaxis_title="Tasa de Recupero (%)",
)

st.plotly_chart(fig_tasa, use_container_width=True)