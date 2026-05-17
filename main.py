import streamlit as st
import pandas as pd
from pathlib import Path
import base64

from modules.modulo_visualizacion import mostrar_modulo_visualizacion
from modules.modulo_prediccion_tendencias import mostrar_modulo_prediccion_tendencias
from modules.modulo_factores_riesgo import mostrar_modulo_factores_riesgo
import textwrap

st.set_page_config(
    page_title="Mortalidad Materna 2002-2024",
    layout="wide"
)


def mostrar_header():
    base_dir = Path(__file__).resolve().parent
    header_img = base_dir / "assets" / "header_matern_ia_mx.jpeg"

    if not header_img.exists():
        st.warning(f"No se encontró la imagen en: {header_img}")
        return

    img_base64 = base64.b64encode(header_img.read_bytes()).decode("utf-8")

    html = textwrap.dedent(f"""
    <div style="border:4px solid #9fd3f5; border-radius:24px; padding:18px 28px;
                display:flex; align-items:center; gap:32px; background-color:#ffffff;
                margin-bottom:24px;">
        <img src="data:image/jpeg;base64,{img_base64}"
             style="width:220px; border-radius:12px;">
        <div style="color:#06265c; font-size:42px; line-height:1.15; font-weight:800;">
            Análisis de Mortalidad Materna<br>
                           2002-2024
        </div>
    </div>
    """)

    st.markdown(html, unsafe_allow_html=True)

def normalizar_texto_estado(x):
    if pd.isna(x):
        return x

    x = str(x).strip().upper()

    reemplazos = {
        "Á": "A",
        "É": "E",
        "Í": "I",
        "Ó": "O",
        "Ú": "U",
        "Ü": "U"
    }

    for a, b in reemplazos.items():
        x = x.replace(a, b)

    mapa = {
        "AGUASCALIENTES": "AGUASCALIENTES",
        "BAJA CALIFORNIA": "BAJA CALIFORNIA",
        "BAJA CALIFORNIA SUR": "BAJA CALIFORNIA SUR",
        "CAMPECHE": "CAMPECHE",
        "COAHUILA": "COAHUILA",
        "COAHUILA DE ZARAGOZA": "COAHUILA",
        "COLIMA": "COLIMA",
        "CHIAPAS": "CHIAPAS",
        "CHIHUAHUA": "CHIHUAHUA",
        "CIUDAD DE MEXICO": "CIUDAD DE MEXICO",
        "DISTRITO FEDERAL": "CIUDAD DE MEXICO",
        "DURANGO": "DURANGO",
        "GUANAJUATO": "GUANAJUATO",
        "GUERRERO": "GUERRERO",
        "HIDALGO": "HIDALGO",
        "JALISCO": "JALISCO",
        "MEXICO": "MEXICO",
        "ESTADO DE MEXICO": "MEXICO",
        "MICHOACAN": "MICHOACAN",
        "MICHOACAN DE OCAMPO": "MICHOACAN",
        "MORELOS": "MORELOS",
        "NAYARIT": "NAYARIT",
        "NUEVO LEON": "NUEVO LEON",
        "OAXACA": "OAXACA",
        "PUEBLA": "PUEBLA",
        "QUERETARO": "QUERETARO",
        "QUERETARO DE ARTEAGA": "QUERETARO",
        "QUINTANA ROO": "QUINTANA ROO",
        "SAN LUIS POTOSI": "SAN LUIS POTOSI",
        "SINALOA": "SINALOA",
        "SONORA": "SONORA",
        "TABASCO": "TABASCO",
        "TAMAULIPAS": "TAMAULIPAS",
        "TLAXCALA": "TLAXCALA",
        "VERACRUZ": "VERACRUZ",
        "VERACRUZ DE IGNACIO DE LA LLAVE": "VERACRUZ",
        "YUCATAN": "YUCATAN",
        "ZACATECAS": "ZACATECAS",
        "ESTADOS UNIDOS DE NORTEAMERICA": "OTROS PAISES",   
        "ESTADOS UNIDOS DE NORTEAMÉRICA": "OTROS PAISES",
        "ESTADOS UNIDOS": "OTROS PAISES",
        "USA": "OTROS PAISES",
        "OTRO PAIS": "OTROS PAISES",
        "OTROS PAISES": "OTROS PAISES",
    }

    return mapa.get(x, x)

def limpiar_dataframe(df):
    df = df.copy()

    columnas_estado = [
        "ENTIDAD_RESIDENCIAD",
        "ENTIDAD_OCURRENCIAD",
        "ENTIDAD_REGISTROD"
    ]

    for col in columnas_estado:
        if col in df.columns:
            df[col] = df[col].apply(normalizar_texto_estado)

    if "EDAD" in df.columns:
        df["EDAD"] = pd.to_numeric(df["EDAD"], errors="coerce")
        df.loc[~df["EDAD"].between(10, 60), "EDAD"] = pd.NA

    return df


def cargar_datos():
    st.sidebar.header("Carga de archivo")

    base_dir = Path(__file__).resolve().parent
    default_csv = base_dir / "mortalidad_materna_2002_2024.csv"

    opcion = st.sidebar.radio(
        "Fuente de datos",
        [
            "Usar archivo por defecto",
            "Subir archivo CSV"
        ]
    )

    if opcion == "Usar archivo por defecto":
        if default_csv.exists():
            df = pd.read_csv(default_csv)
            df = limpiar_dataframe(df)
            st.sidebar.success("Archivo por defecto cargado.")
            return df
        else:
            st.sidebar.warning(f"No se encontró el archivo: {default_csv}")
            return None

    archivo = st.sidebar.file_uploader(
        "Carga el archivo CSV",
        type=["csv"]
    )

    if archivo is not None:
        df = pd.read_csv(archivo)
        df = limpiar_dataframe(df)
        return df

    return None

def main():
    mostrar_header()

    df = cargar_datos()

    if df is None:
        st.info("Carga el archivo CSV para comenzar.")
        return

    st.success("Archivo cargado correctamente.")

    st.sidebar.divider()
    st.sidebar.header("Módulos")

    modulo = st.sidebar.radio(
        "Selecciona un módulo",
        [
            "Visualización de tablas y gráficas",
            "Predicción de tendencias",
            "Análisis de factores de riesgo",
        ]
    )

    if modulo == "Visualización de tablas y gráficas":
        mostrar_modulo_visualizacion(df)

    elif modulo == "Predicción de tendencias":
        mostrar_modulo_prediccion_tendencias(df)

    elif modulo == "Análisis de factores de riesgo":
        mostrar_modulo_factores_riesgo(df)


if __name__ == "__main__":
    main()
