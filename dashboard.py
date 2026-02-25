import streamlit as st
import pandas as pd

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