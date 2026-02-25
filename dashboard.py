import streamlit as st
import plotly.express as px
import pandas as pd
st.markdown(
    """
    <style>
    .metric-container {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="Dashboard Robos Automotores", layout="wide")

@st.cache_data
def cargar_datos():
    df = pd.read_csv("data/raw/dnrpa-robos-recuperos-autos-202601.csv")
    df["tramite_fecha"] = pd.to_datetime(df["tramite_fecha"], errors="coerce")
    df["edad"] = 2026 - df["titular_anio_nacimiento"]
    return df

df = cargar_datos()

st.title("📊 Dashboard - Robos y Recuperos Automotores")

robos = df[df["tramite_tipo"].str.contains("ROBO", na=False)]
recuperos = df[df["tramite_tipo"].str.contains("RECUPERO", na=False)]

tasa = (len(recuperos) / len(robos) * 100) if len(robos) > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Robos", len(robos))
col2.metric("Total Recuperos", len(recuperos))
col3.metric("Tasa Recuperación (%)", f"{tasa:.2f}")

st.divider()

provincia = st.selectbox(
    "Seleccionar Provincia",
    ["Todas"] + sorted(df["registro_seccional_provincia"].unique())
)

df_filtrado = df if provincia == "Todas" else df[df["registro_seccional_provincia"] == provincia]

st.subheader("🚗 Top 10 Marcas")
st.bar_chart(df_filtrado["automotor_marca_descripcion"].value_counts().head(10))

st.subheader("👤 Distribución por Género")
st.bar_chart(df_filtrado["titular_genero"].value_counts())

st.subheader("📅 Evolución Temporal")
serie = df_filtrado.groupby(df_filtrado["tramite_fecha"].dt.date).size()
st.line_chart(serie)
st.subheader("🗺 Distribución Geográfica (Visualización Aproximada)")

# Coordenadas aproximadas de provincias principales
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

fig = px.scatter_mapbox(
    map_df,
    lat="lat",
    lon="lon",
    size="cantidad",
    hover_name="registro_seccional_provincia",
    zoom=4,
)

fig.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
st.subheader("🤖 Modelo Predictivo - Probabilidad de Recupero")

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

pred = modelo.predict(X_test)

accuracy = accuracy_score(y_test, pred)

st.metric("Precisión del modelo", f"{accuracy:.2f}")