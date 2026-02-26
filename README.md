\# 🛰 Panel de Inteligencia Criminal Automotor



Dashboard interactivo desarrollado en \*\*Python + Streamlit\*\* para el análisis estratégico de robos y recuperos de automotores en Argentina, utilizando datos abiertos oficiales.



---



\## 🎯 Objetivo



Construir un centro de monitoreo analítico que permita:



\- Visualizar incidencia por marca y provincia

\- Analizar eficiencia operativa

\- Estudiar patrones temporales

\- Modelar probabilidad de recupero

\- Detectar oportunidades estratégicas



---



\## 🛠 Tecnologías Utilizadas



\- Python

\- Pandas

\- Streamlit

\- Plotly

\- Scikit-Learn

\- Matplotlib



---



\## 📊 Funcionalidades



✔ Métricas operativas (Robos / Recuperos / Eficiencia)

✔ Filtro dinámico por provincia

✔ Visualización temporal y geográfica

✔ Modelo predictivo (Random Forest)

✔ Matriz de confusión y Curva ROC

✔ Tasa real de recupero por marca



---



\## 🧠 Enfoque Analítico



Se identificó desbalance de clases en el dataset, por lo que:



\- Se utilizó `class\\\_weight="balanced"`

\- Se ajustó el threshold de clasificación

\- Se analizó recall y precision de la clase minoritaria



El modelo no se evalúa únicamente por accuracy, sino por su capacidad de detectar eventos de recupero.



---



\## 📁 Dataset



Fuente: Datos abiertos oficiales - DNRPA

Archivo: robos y recuperos automotores (Argentina)



---



\## 🚀 Cómo ejecutar



pip install -r requirements.txt

streamlit run dashboard.py



---



\## 👤 Autor



Luciano Hernán Kovacevich

Tecnicatura en Ciencia de Datos

Policía Federal Argentina

