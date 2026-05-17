import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import math
from scipy.stats import chi2_contingency
import networkx as nx
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re


def agrupar_categoria(codigo, descripcion):
    codigo = str(codigo).upper()
    descripcion = str(descripcion).upper()

    if codigo.startswith(("O00", "O01", "O02", "O03", "O04", "O05", "O06", "O07", "O08")):
        return "Aborto y embarazo terminado"

    elif codigo.startswith(("O10", "O11", "O12", "O13", "O14", "O15", "O16")):
        return "Trastornos hipertensivos"

    elif codigo.startswith(("O20", "O21", "O22", "O23", "O24", "O25", "O26", "O28", "O29")):
        return "Complicaciones del embarazo"

    elif codigo.startswith(("O30", "O31", "O32", "O33", "O34", "O35", "O36", "O40", "O41")):
        return "Atención materna fetal"

    elif codigo.startswith(("O42", "O43", "O44", "O45", "O46")):
        return "Hemorragias obstétricas"

    elif codigo.startswith(("O60", "O61", "O62", "O63", "O64", "O65", "O66", "O67", "O68", "O69", "O70", "O71", "O72", "O73", "O74", "O75")):
        return "Complicaciones del parto"

    elif codigo.startswith(("O85", "O86")):
        return "Infecciones puerperales"

    elif codigo.startswith(("O87", "O88", "O89", "O90", "O91", "O92")):
        return "Complicaciones puerperio"

    elif codigo.startswith(("A", "B")):
        return "Enfermedades infecciosas"

    elif codigo.startswith(("C", "D")):
        return "Neoplasias y hematológicas"

    elif codigo.startswith(("I")):
        return "Cardiovasculares"

    elif codigo.startswith(("J")):
        return "Respiratorias"

    elif codigo.startswith(("K")):
        return "Digestivas"

    elif codigo.startswith(("N")):
        return "Genitourinarias"

    else:
        return "Otras causas"


def filtrar_top_causas(df):
    st.markdown("### Filtros para distribución clínica")

    df_filtrado = df.copy()

    tipo_geo = st.radio(
        "Ubicación",
        [
            "Todo el país",
            "Entidad de ocurrencia",
            "Entidad de residencia"
        ],
        horizontal=True,
        key="top_causas_tipo_geo"
    )

    if tipo_geo == "Entidad de ocurrencia":
        col_estado = "ENTIDAD_OCURRENCIAD"
        col_municipio = "MUNICIPIO_OCURRENCIAD"
        col_localidad = "LOCALIDAD_OCURRENCIAD"

    elif tipo_geo == "Entidad de residencia":
        col_estado = "ENTIDAD_RESIDENCIAD"
        col_municipio = "MUNICIPIO_RESIDENCIAD"
        col_localidad = "LOCALIDAD_RESIDENCIAD"

    else:
        col_estado = None
        col_municipio = None
        col_localidad = None

    if tipo_geo != "Todo el país":
        estados = sorted(df_filtrado[col_estado].dropna().astype(str).unique())

        estados_sel = st.multiselect(
            "Estado(s)",
            estados,
            key="top_causas_estados"
        )

        if estados_sel:
            df_filtrado = df_filtrado[
                df_filtrado[col_estado].astype(str).isin(estados_sel)
            ]

        municipios = sorted(df_filtrado[col_municipio].dropna().astype(str).unique())

        municipios_sel = st.multiselect(
            "Municipio(s)",
            municipios,
            key="top_causas_municipios"
        )

        if municipios_sel:
            df_filtrado = df_filtrado[
                df_filtrado[col_municipio].astype(str).isin(municipios_sel)
            ]

        localidades = sorted(df_filtrado[col_localidad].dropna().astype(str).unique())

        localidades_sel = st.multiselect(
            "Localidad(es)",
            localidades,
            key="top_causas_localidades"
        )

        if localidades_sel:
            df_filtrado = df_filtrado[
                df_filtrado[col_localidad].astype(str).isin(localidades_sel)
            ]

    st.markdown("### Filtros sociodemográficos")

    filtro_extra = st.selectbox(
        "Filtro adicional",
        [
            "Ninguno",
            "Grupo de edad",
            "Ocupación",
            "Estado conyugal",
            "Derechohabiencia",
            "Vivía sola"
        ],
        key="top_causas_filtro_extra"
    )

    if filtro_extra == "Grupo de edad" and "EDAD" in df_filtrado.columns:
        df_filtrado["EDAD"] = pd.to_numeric(df_filtrado["EDAD"], errors="coerce")

        edad_min = int(df_filtrado["EDAD"].dropna().min())
        edad_max = int(df_filtrado["EDAD"].dropna().max())

        col1, col2 = st.columns(2)

        with col1:
            edad_ini = st.number_input(
                "Edad mínima",
                min_value=edad_min,
                max_value=edad_max,
                value=edad_min,
                step=1,
                key="top_causas_edad_ini"
            )

        with col2:
            edad_fin = st.number_input(
                "Edad máxima",
                min_value=edad_min,
                max_value=edad_max,
                value=edad_max,
                step=1,
                key="top_causas_edad_fin"
            )

        df_filtrado = df_filtrado[
            df_filtrado["EDAD"].between(edad_ini, edad_fin)
        ]

    elif filtro_extra == "Ocupación" and "OCUPACION_HABITUALD" in df_filtrado.columns:
        opciones = sorted(df_filtrado["OCUPACION_HABITUALD"].dropna().astype(str).unique())

        seleccion = st.multiselect(
            "Ocupación(es)",
            opciones,
            key="top_causas_ocupacion"
        )

        if seleccion:
            df_filtrado = df_filtrado[
                df_filtrado["OCUPACION_HABITUALD"].astype(str).isin(seleccion)
            ]

    elif filtro_extra == "Estado conyugal" and "ESTADO_CONYUGALD" in df_filtrado.columns:
        opciones = sorted(df_filtrado["ESTADO_CONYUGALD"].dropna().astype(str).unique())

        seleccion = st.multiselect(
            "Estado(s) conyugal(es)",
            opciones,
            key="top_causas_conyugal"
        )

        if seleccion:
            df_filtrado = df_filtrado[
                df_filtrado["ESTADO_CONYUGALD"].astype(str).isin(seleccion)
            ]

    elif filtro_extra == "Derechohabiencia" and "DERECHOHABIENCIAD" in df_filtrado.columns:
        opciones = sorted(df_filtrado["DERECHOHABIENCIAD"].dropna().astype(str).unique())

        seleccion = st.multiselect(
            "Derechohabiencia",
            opciones,
            key="top_causas_derechohabiencia"
        )

        if seleccion:
            df_filtrado = df_filtrado[
                df_filtrado["DERECHOHABIENCIAD"].astype(str).isin(seleccion)
            ]

    elif filtro_extra == "Vivía sola":
        columnas_posibles = [
            "VIVIA_SOLAD",
            "VIVIA_SOLA",
            "VIVE_SOLAD",
            "VIVE_SOLA",
            "CONDICION_VIVIENDAD"
        ]

        columna_vivia_sola = None

        for col in columnas_posibles:
            if col in df_filtrado.columns:
                columna_vivia_sola = col
                break

        if columna_vivia_sola is None:
            st.warning(
                "No encontré una columna clara para 'vivía sola'. "
                "Si la columna tiene otro nombre, agrégala a columnas_posibles."
            )
        else:
            opciones = sorted(df_filtrado[columna_vivia_sola].dropna().astype(str).unique())

            seleccion = st.multiselect(
                f"{columna_vivia_sola}",
                opciones,
                key="top_causas_vivia_sola"
            )

            if seleccion:
                df_filtrado = df_filtrado[
                    df_filtrado[columna_vivia_sola].astype(str).isin(seleccion)
                ]

    st.markdown(f"**Registros incluidos:** {df_filtrado.shape[0]}")

    return df_filtrado


def mostrar_top_causas_por_entidad(df):
    st.divider()

    st.subheader("Top causas por territorio")

    st.info(
        "Esta gráfica compara la composición porcentual de las principales causas "
        "según el nivel territorial seleccionado. Las causas con proporciones muy pequeñas "
        "pueden agruparse como OTRAS CAUSAS para mejorar la lectura."
    )

    columnas_requeridas = [
        "CAUSA_CIE_4",
        "CAUSA_CIE_4D",
        "ENTIDAD_RESIDENCIAD",
        "ENTIDAD_OCURRENCIAD",
        "MUNICIPIO_RESIDENCIAD",
        "MUNICIPIO_OCURRENCIAD",
        "LOCALIDAD_RESIDENCIAD",
        "LOCALIDAD_OCURRENCIAD"
    ]

    faltantes = [
        col for col in columnas_requeridas if col not in df.columns
    ]

    if faltantes:
        st.warning(f"Faltan columnas necesarias: {faltantes}")
        return

    tipo_entidad = st.selectbox(
        "Usar ubicación por:",
        [
            "Entidad de residencia",
            "Entidad de ocurrencia"
        ],
        key="top_causas_territorio_tipo"
    )

    if tipo_entidad == "Entidad de residencia":
        col_estado = "ENTIDAD_RESIDENCIAD"
        col_municipio = "MUNICIPIO_RESIDENCIAD"
        col_localidad = "LOCALIDAD_RESIDENCIAD"
    else:
        col_estado = "ENTIDAD_OCURRENCIAD"
        col_municipio = "MUNICIPIO_OCURRENCIAD"
        col_localidad = "LOCALIDAD_OCURRENCIAD"

    df_temp = df.copy()

    estados = sorted(df_temp[col_estado].dropna().astype(str).unique())

    estado_sel = st.selectbox(
        "Estado",
        ["Todo el país"] + estados,
        key="top_causas_territorio_estado"
    )

    nivel_comparacion = "Estado"
    col_comparacion = col_estado

    if estado_sel != "Todo el país":
        df_temp = df_temp[
            df_temp[col_estado].astype(str) == estado_sel
        ].copy()

        municipios = sorted(df_temp[col_municipio].dropna().astype(str).unique())

        municipio_sel = st.selectbox(
            "Municipio",
            ["Todo el estado"] + municipios,
            key="top_causas_territorio_municipio"
        )

        nivel_comparacion = "Municipio"
        col_comparacion = col_municipio

        if municipio_sel != "Todo el estado":
            df_temp = df_temp[
                df_temp[col_municipio].astype(str) == municipio_sel
            ].copy()

            localidades = sorted(df_temp[col_localidad].dropna().astype(str).unique())

            localidad_sel = st.selectbox(
                "Localidad",
                ["Todo el municipio"] + localidades,
                key="top_causas_territorio_localidad"
            )

            nivel_comparacion = "Localidad"
            col_comparacion = col_localidad

            if localidad_sel != "Todo el municipio":
                df_temp = df_temp[
                    df_temp[col_localidad].astype(str) == localidad_sel
                ].copy()

    if df_temp.empty:
        st.warning("No hay registros para los filtros territoriales seleccionados.")
        return

    col1, col2 = st.columns(2)

    with col1:
        top_n = st.number_input(
            "Número de causas principales a comparar",
            min_value=3,
            max_value=30,
            value=10,
            step=1,
            key="top_causas_territorio_topn"
        )

    with col2:
        cota_porcentaje = st.number_input(
            "Agrupar como OTRAS si es menor a (%)",
            min_value=0.0,
            max_value=20.0,
            value=3.0,
            step=0.5,
            key="top_causas_cota_porcentaje"
        )

    df_top_global = (
        df_temp
        .groupby(["CAUSA_CIE_4", "CAUSA_CIE_4D"])
        .size()
        .reset_index(name="NUM_DEF")
        .sort_values("NUM_DEF", ascending=False)
        .head(top_n)
    )

    if df_top_global.empty:
        st.warning("No hay causas disponibles con los filtros seleccionados.")
        return

    causas_top = df_top_global["CAUSA_CIE_4"].astype(str).tolist()

    df_plot_base = df_temp[
        df_temp["CAUSA_CIE_4"].astype(str).isin(causas_top)
    ].copy()

    df_plot_base[col_comparacion] = df_plot_base[col_comparacion].astype(str)
    df_plot_base["CIE4D_CLAVE"] = df_plot_base["CAUSA_CIE_4"].astype(str)
    df_plot_base["DESCRIPCION"] = df_plot_base["CAUSA_CIE_4D"].astype(str)

    df_agrupado = (
        df_plot_base
        .groupby([col_comparacion, "CIE4D_CLAVE", "DESCRIPCION"])
        .size()
        .reset_index(name="Muertes")
    )

    if df_agrupado.empty:
        st.warning("No hay datos suficientes para graficar.")
        return

    totales = df_agrupado.groupby(col_comparacion)["Muertes"].transform("sum")
    df_agrupado["Proporcion"] = df_agrupado["Muertes"] / totales
    df_agrupado["Porcentaje"] = df_agrupado["Proporcion"] * 100

    df_agrupado["CIE4D_GRAFICA"] = df_agrupado["CIE4D_CLAVE"]
    df_agrupado["DESCRIPCION_GRAFICA"] = df_agrupado["DESCRIPCION"]

    mascara_otras = df_agrupado["Porcentaje"] < cota_porcentaje

    df_agrupado.loc[mascara_otras, "CIE4D_GRAFICA"] = "OTRAS"
    df_agrupado.loc[mascara_otras, "DESCRIPCION_GRAFICA"] = (
        "Otras causas por debajo de la cota seleccionada"
    )

    df_plot = (
        df_agrupado
        .groupby([col_comparacion, "CIE4D_GRAFICA", "DESCRIPCION_GRAFICA"])
        .agg(
            Muertes=("Muertes", "sum"),
            Causas_incluidas=("CIE4D_CLAVE", lambda x: ", ".join(sorted(set(x))))
        )
        .reset_index()
    )

    totales_plot = df_plot.groupby(col_comparacion)["Muertes"].transform("sum")
    df_plot["Proporcion"] = df_plot["Muertes"] / totales_plot
    df_plot["Porcentaje"] = df_plot["Proporcion"] * 100

    df_plot = df_plot.sort_values(
        [col_comparacion, "Porcentaje"],
        ascending=[True, False]
    )

    df_plot["Texto_barra"] = df_plot.apply(
        lambda row: f"{row['CIE4D_GRAFICA']}<br>{row['Porcentaje']:.1f}%"
        if row["Porcentaje"] >= cota_porcentaje
        else "",
        axis=1
    )

    orden_territorios = (
        df_plot
        .groupby(col_comparacion)["Muertes"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    diccionario = (
        df_top_global[["CAUSA_CIE_4", "CAUSA_CIE_4D", "NUM_DEF"]]
        .rename(columns={
            "CAUSA_CIE_4": "CIE4D_CLAVE",
            "CAUSA_CIE_4D": "DESCRIPCION",
            "NUM_DEF": "Defunciones en filtro actual"
        })
    )

    st.markdown("### Diccionario de causas incluidas")

    st.dataframe(
        diccionario,
        use_container_width=True
    )

    fig = px.bar(
        df_plot,
        x=col_comparacion,
        y="Proporcion",
        color="CIE4D_GRAFICA",
        text="Texto_barra",
        category_orders={
            col_comparacion: orden_territorios
        },
        title=f"Distribución porcentual de top {top_n} causas por {nivel_comparacion.lower()}",
        labels={
            col_comparacion: nivel_comparacion,
            "Proporcion": "Proporción",
            "CIE4D_GRAFICA": "Clave CIE-4"
        },
        hover_data={
            "DESCRIPCION_GRAFICA": True,
            "Muertes": True,
            "Porcentaje": ":.2f",
            "Causas_incluidas": True
        }
    )

    fig.update_traces(
        textposition="inside",
        insidetextanchor="middle"
    )

    fig.update_layout(
        height=780,
        barmode="stack",
        xaxis=dict(
            tickangle=-45,
            automargin=True
        ),
        yaxis=dict(
            tickformat=".0%",
            title="Proporción del total de muertes"
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.32,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=80, b=300)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        f"Nota: se solicitaron {top_n} causas principales. "
        f"Los segmentos con porcentaje menor a {cota_porcentaje:.1f}% "
        "se agrupan como OTRAS dentro de cada territorio. "
        "Si una entidad, municipio o localidad no tiene registros para alguna causa, "
        "esa causa no aparece en su barra."
    )

    with st.expander("Ver tabla territorio-causa"):
        st.dataframe(
            df_plot.sort_values(
                [col_comparacion, "Porcentaje"],
                ascending=[True, False]
            ),
            use_container_width=True
        )

    with st.expander("Ver detalle antes de agrupar OTRAS"):
        st.dataframe(
            df_agrupado.sort_values(
                [col_comparacion, "Porcentaje"],
                ascending=[True, False]
            ),
            use_container_width=True
        )

def mostrar_nube_palabras_causas(df_counts_sorted):
    st.divider()

    st.subheader("Nube de palabras de causas principales")

    st.info(
        "La nube resume las palabras más frecuentes en las descripciones de las causas "
        "de muerte seleccionadas. Se eliminan palabras vacías en español y términos generales "
        "como embarazo, parto, materna y puerperio."
    )

    top_n = st.number_input(
        "Número de causas principales usadas para la nube",
        min_value=10,
        max_value=min(100, len(df_counts_sorted)),
        value=min(40, len(df_counts_sorted)),
        step=5,
        key="nube_top_n_causas"
    )

    exclusiones_adicionales = {
        "embarazo", "embarazada", "embarazadas", "gestante", "gestantes",
        "gestacion", "gestación", "parto", "materna", "materno", "maternidad",
        "puerpera", "puérpera", "puerperio", "perinatal", "trabajo",
        "muerte", "defuncion", "defunción", "causa", "causas",
        "ocurre", "ocurrida", "ocurrido", "despues", "después",
        "antes", "durante", "asociada", "asociado", "debido",
        "otras", "otra", "otros", "otro", "sistema",
        "enfermedad", "enfermedades", "complicacion", "complicación",
        "complicaciones", "afeccion", "afección", "afecciones",
        "clasificadas", "clasificada", "clasificado",
        "no", "si", "sí"
    }

    stopwords_es = {
        "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como",
        "con", "contra", "cual", "cuando", "de", "del", "desde", "donde",
        "durante", "e", "el", "ella", "ellas", "ellos", "en", "entre",
        "era", "eran", "es", "esa", "esas", "ese", "eso", "esos", "esta",
        "estaba", "estado", "están", "estar", "estas", "este", "esto",
        "estos", "fue", "fueron", "ha", "han", "hasta", "hay", "la",
        "las", "le", "les", "lo", "los", "más", "me", "mi", "mientras",
        "muy", "ni", "o", "otra", "otras", "otro", "otros", "para",
        "pero", "por", "porque", "que", "se", "ser", "si", "sin", "sobre",
        "su", "sus", "también", "te", "tiene", "un", "una", "uno", "unos",
        "y", "ya"
    }

    stopwords_ampliadas = STOPWORDS.union(stopwords_es).union(exclusiones_adicionales)

    def limpiar_texto(texto):
        texto = str(texto).lower()
        texto = texto.replace("á", "a").replace("é", "e").replace("í", "i")
        texto = texto.replace("ó", "o").replace("ú", "u").replace("ü", "u")
        texto = re.sub(r"[^a-zñ\s]", " ", texto)
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto

    def extraer_causa_principal(descripcion):
        descripcion = limpiar_texto(descripcion)
        palabras = descripcion.split()

        palabras_filtradas = [
            p for p in palabras
            if p not in stopwords_ampliadas and len(p) > 2
        ]

        if len(palabras_filtradas) >= 2:
            return " ".join(palabras_filtradas[:2])
        elif palabras_filtradas:
            return palabras_filtradas[0]
        else:
            return ""

    df_top = df_counts_sorted.head(top_n).copy()

    df_top["Causa_principal_estimada"] = df_top["DESCRIPCION"].apply(
        extraer_causa_principal
    )

    texto = " ".join(
        (
            (row["Causa_principal_estimada"] + " ") * int(row["NUM_DEF"])
        )
        for _, row in df_top.iterrows()
        if row["Causa_principal_estimada"]
    )

    if not texto.strip():
        st.warning("No hay texto suficiente para generar la nube de palabras.")
        return

    wordcloud = WordCloud(
        width=1400,
        height=650,
        background_color="white",
        colormap="Reds",
        stopwords=stopwords_ampliadas,
        max_words=120,
        collocations=False
    ).generate(texto)

    fig, ax = plt.subplots(figsize=(16, 7), dpi=160)

    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        f"Nube de palabras basada en causas principales estimadas Top {top_n}",
        fontsize=16,
        fontweight="bold"
    )

    st.pyplot(fig, use_container_width=True)

    with st.expander("Ver términos usados para la nube"):
        st.dataframe(
            df_top[[
                "CIE4D_CLAVE",
                "DESCRIPCION",
                "NUM_DEF",
                "Causa_principal_estimada"
            ]],
            use_container_width=True
        )


def mostrar_top_causas(df):
    st.divider()

    st.subheader("Distribución clínica de causas de muerte")

    if "CAUSA_CIE_4D" not in df.columns or "CAUSA_CIE_4" not in df.columns:
        st.warning("No se encontraron las columnas CAUSA_CIE_4 y/o CAUSA_CIE_4D.")
        return

    df_filtrado = filtrar_top_causas(df)

    if df_filtrado.empty:
        st.warning("No hay registros con los filtros seleccionados.")
        return

    df_counts = (
        df_filtrado.groupby(["CAUSA_CIE_4", "CAUSA_CIE_4D"])
        .size()
        .reset_index(name="NUM_DEF")
        .sort_values("NUM_DEF", ascending=False)
    )

    df_counts = df_counts.rename(columns={
        "CAUSA_CIE_4": "CIE4D_CLAVE",
        "CAUSA_CIE_4D": "DESCRIPCION"
    })

    st.markdown("### Tabla de causas de muerte")

    st.dataframe(
        df_counts,
        use_container_width=True
    )


    st.markdown("### Top 50 causas agrupadas clínicamente")

    df_50causas = df_counts.head(50).copy()

    df_50causas["GRUPO_CIE"] = df_50causas.apply(
        lambda row: agrupar_categoria(
            row["CIE4D_CLAVE"],
            row["DESCRIPCION"]
        ),
        axis=1
    )

    df_50causas["ETIQUETA_CORTA"] = (
        df_50causas["CIE4D_CLAVE"].astype(str)
        + " - "
        + df_50causas["DESCRIPCION"].astype(str).str.slice(0, 60)
    )

    grupo_sel = st.selectbox(
        "Selecciona grupo clínico para visualizar",
        ["Todos"] + sorted(df_50causas["GRUPO_CIE"].dropna().unique()),
        key="grupo_clinico_top50"
    )

    if grupo_sel != "Todos":
        df_plot = df_50causas[
            df_50causas["GRUPO_CIE"] == grupo_sel
        ].copy()
    else:
        df_plot = df_50causas.copy()

    df_plot = df_plot.sort_values("NUM_DEF", ascending=True)

    fig_top = px.bar(
        df_plot,
        x="NUM_DEF",
        y="ETIQUETA_CORTA",
        color="GRUPO_CIE",
        orientation="h",
        hover_data={
            "CIE4D_CLAVE": True,
            "DESCRIPCION": True,
            "NUM_DEF": True,
            "GRUPO_CIE": True,
            "ETIQUETA_CORTA": False
        },
        title="Top 50 causas de muerte materna agrupadas clínicamente",
        labels={
            "NUM_DEF": "Número de defunciones",
            "ETIQUETA_CORTA": "Causa",
            "GRUPO_CIE": "Grupo clínico"
        }
    )

    fig_top.update_layout(
        height=max(700, 24 * len(df_plot)),
        yaxis=dict(
            automargin=True,
            tickfont=dict(size=10)
        ),
        xaxis_title="Número de defunciones",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=80, b=160)
    )

    st.plotly_chart(
        fig_top,
        use_container_width=True
    )

    st.divider()

    st.subheader("Distribución proporcional de causas")

    df_counts_sorted = (
        df_counts
        .sort_values("NUM_DEF", ascending=False)
        .reset_index(drop=True)
    )

    total_muertes = df_counts_sorted["NUM_DEF"].sum()

    df_counts_sorted["Porcentaje"] = (
        df_counts_sorted["NUM_DEF"] / total_muertes * 100
    ).round(2)

    df_counts_sorted["GRUPO_CIE"] = df_counts_sorted.apply(
        lambda row: agrupar_categoria(
            row["CIE4D_CLAVE"],
            row["DESCRIPCION"]
        ),
        axis=1
    )

    st.markdown("### Diccionario de claves CIE-4")

    st.dataframe(
        df_counts_sorted[[
            "CIE4D_CLAVE",
            "DESCRIPCION",
            "GRUPO_CIE",
            "NUM_DEF",
            "Porcentaje"
        ]],
        use_container_width=True
    )

    tipo_prop = st.selectbox(
        "Tipo de visualización proporcional",
        [
            "Treemap interactivo",
            "Barras proporcionales horizontales"
        ],
        key="tipo_prop_causas"
    )

    if tipo_prop == "Treemap interactivo":

        fig_tree = px.treemap(
            df_counts_sorted,
            path=["GRUPO_CIE", "CIE4D_CLAVE"],
            values="NUM_DEF",
            color="Porcentaje",
            color_continuous_scale="Reds",
            hover_data={
                "DESCRIPCION": True,
                "NUM_DEF": True,
                "Porcentaje": True
            },
            title="Distribución proporcional de causas de muerte"
        )

        fig_tree.update_traces(
            textinfo="label+percent parent"
        )

        fig_tree.update_layout(
            height=750,
            margin=dict(l=20, r=20, t=80, b=20)
        )

        st.plotly_chart(
            fig_tree,
            use_container_width=True
        )

    else:

        top_n_prop = st.number_input(
            "Número de causas a mostrar",
            min_value=10,
            max_value=min(100, len(df_counts_sorted)),
            value=min(50, len(df_counts_sorted)),
            step=5,
            key="top_n_prop_causas"
        )

        df_prop = df_counts_sorted.head(top_n_prop).copy()
        df_prop = df_prop.sort_values("Porcentaje", ascending=True)

        df_prop["ETIQUETA_CORTA"] = (
            df_prop["CIE4D_CLAVE"].astype(str)
            + " - "
            + df_prop["DESCRIPCION"].astype(str).str.slice(0, 50)
        )

        fig_prop = px.bar(
            df_prop,
            x="Porcentaje",
            y="ETIQUETA_CORTA",
            color="GRUPO_CIE",
            orientation="h",
            text="CIE4D_CLAVE",
            hover_data={
                "CIE4D_CLAVE": True,
                "DESCRIPCION": True,
                "NUM_DEF": True,
                "Porcentaje": True,
                "ETIQUETA_CORTA": False
            },
            title="Proporción de causas de muerte",
            labels={
                "Porcentaje": "Porcentaje del total",
                "ETIQUETA_CORTA": "Causa",
                "GRUPO_CIE": "Grupo clínico"
            }
        )

        fig_prop.update_traces(
            textposition="inside",
            insidetextanchor="middle"
        )

        fig_prop.update_layout(
            height=max(700, 24 * len(df_prop)),
            yaxis=dict(
                automargin=True,
                tickfont=dict(size=10)
            ),
            xaxis_title="Porcentaje del total",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=40, r=40, t=80, b=160)
        )

        st.plotly_chart(
            fig_prop,
            use_container_width=True
        )

    mostrar_nube_palabras_causas(df_counts_sorted)

def mostrar_modulo_factores_riesgo(df):
    st.header("Módulo 3: Análisis de factores de riesgo")

    st.info(
        "Este módulo permitirá explorar variables asociadas a la mortalidad materna, "
        "como edad, escolaridad, derechohabiencia, ocupación, estado civil, tamaño de localidad "
        "y sitio de defunción."
    )

    columnas_sugeridas = [
        "EDAD",
        "EDAD_QUINQUENALD",
        "ESCOLARIDADD",
        "DERECHOHABIENCIAD",
        "OCUPACION_HABITUALD",
        "ESTADO_CONYUGALD",
        "TAMANIO_LOCALIDADD",
        "SITIO_DEFUNCIOND",
        "ASISTENCIA_MEDICAD",
        "CAUSA_CIE_4D"
    ]

    columnas_disponibles = [
        col for col in columnas_sugeridas if col in df.columns
    ]

    if not columnas_disponibles:
        st.warning("No se encontraron columnas sugeridas para este módulo.")
        return

    variable = st.selectbox(
        "Selecciona un factor de riesgo o característica",
        columnas_disponibles
    )

    conteo = (
        df[variable]
        .value_counts(dropna=False)
        .reset_index()
    )

    conteo.columns = [variable, "Frecuencia"]

    st.subheader("Tabla de frecuencia")

    st.dataframe(
        conteo,
        use_container_width=True
    )

    st.subheader("Gráfica de distribución")

    conteo["Etiqueta"] = conteo[variable].astype(str)

    fig = px.bar(
        conteo,
        x="Etiqueta",
        y="Frecuencia",
        title=f"Distribución de {variable}",
        labels={
            "Etiqueta": variable,
            "Frecuencia": "Frecuencia"
        }
    )

    fig.update_layout(
        height=550,
        xaxis=dict(
            tickangle=-90,
            automargin=True
        ),
        margin=dict(l=40, r=40, t=80, b=180)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    mostrar_top_causas(df)
    mostrar_top_causas_por_entidad(df)
    mostrar_red_asociacion_factores(df)
    mostrar_arbol_causa_muerte(df)


def cramers_v(x, y):
    tabla = pd.crosstab(x, y)

    if tabla.shape[0] < 2 or tabla.shape[1] < 2:
        return np.nan

    chi2, _, _, _ = chi2_contingency(tabla)
    n = tabla.sum().sum()

    if n == 0:
        return np.nan

    r, k = tabla.shape
    return np.sqrt((chi2 / n) / min(k - 1, r - 1))


def calcular_matriz_cramers(df, variables):
    matriz = pd.DataFrame(
        index=variables,
        columns=variables,
        dtype=float
    )

    for var1 in variables:
        for var2 in variables:
            if var1 == var2:
                matriz.loc[var1, var2] = 1.0
            else:
                datos = df[[var1, var2]].dropna()
                matriz.loc[var1, var2] = cramers_v(datos[var1], datos[var2])

    return matriz


def crear_red_asociaciones(matriz, umbral=0.10):
    G = nx.Graph()

    for var in matriz.index:
        G.add_node(var)

    for i, var1 in enumerate(matriz.index):
        for var2 in matriz.columns[i + 1:]:
            valor = matriz.loc[var1, var2]

            if pd.notna(valor) and valor >= umbral:
                G.add_edge(var1, var2, weight=valor)

    return G


def graficar_red_plotly(G):
    if G.number_of_edges() == 0:
        st.warning("No hay asociaciones por encima del umbral seleccionado.")
        return

    pos = nx.spring_layout(G, seed=42, k=0.8)

    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1),
        hoverinfo="none",
        mode="lines",
        showlegend=False
    )

    node_x = []
    node_y = []
    node_text = []
    node_size = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_size.append(20 + 5 * G.degree(node))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            size=node_size,
            line=dict(width=1)
        ),
        showlegend=False
    )

    fig = go.Figure(
        data=[edge_trace, node_trace]
    )

    fig.update_layout(
        title="Red de asociación entre factores",
        height=650,
        margin=dict(l=20, r=20, t=80, b=20),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False)
    )

    st.plotly_chart(fig, use_container_width=True)


def mostrar_red_asociacion_factores(df):
    st.divider()

    st.subheader("Red de asociación entre factores")

    st.info(
        "Esta sección usa Cramér’s V para medir asociación entre variables categóricas. "
        "Valores cercanos a 0 indican poca asociación; valores cercanos a 1 indican asociación fuerte."
    )

    variables_sugeridas = [
        "EDAD_QUINQUENALD",
        "ESTADO_CONYUGALD",
        "ESCOLARIDADD",
        "DERECHOHABIENCIAD",
        "OCUPACION_HABITUALD",
        "TAMANIO_LOCALIDADD",
        "SITIO_DEFUNCIOND",
        "ASISTENCIA_MEDICAD",
        "CAUSA_CIE_4D",
        "RAZON_MORTALIDAD_MATERNAD"
    ]

    variables_disponibles = [
        v for v in variables_sugeridas if v in df.columns
    ]

    variables_sel = st.multiselect(
        "Selecciona variables para analizar",
        options=variables_disponibles,
        default=variables_disponibles[:8]
    )

    if len(variables_sel) < 2:
        st.warning("Selecciona al menos dos variables.")
        return

    umbral = st.selectbox(
        "Umbral mínimo de asociación para dibujar enlaces",
        [0.05, 0.10, 0.15, 0.20, 0.30],
        index=1
    )

    df_red = df[variables_sel].copy()

    matriz = calcular_matriz_cramers(df_red, variables_sel)

    st.markdown("**Matriz de asociación Cramér’s V**")

    st.dataframe(
        matriz.round(3),
        use_container_width=True
    )

    fig_heatmap = px.imshow(
        matriz.astype(float),
        text_auto=".2f",
        color_continuous_scale="Reds",
        title="Mapa de calor de asociaciones entre factores"
    )

    fig_heatmap.update_layout(
        height=650,
        margin=dict(l=40, r=40, t=80, b=120)
    )

    st.plotly_chart(
        fig_heatmap,
        use_container_width=True
    )

    G = crear_red_asociaciones(matriz, umbral=umbral)

    graficar_red_plotly(G)

    componentes = list(nx.connected_components(G))

    resumen_componentes = []

    for i, comp in enumerate(componentes, start=1):
        resumen_componentes.append({
            "Grupo": f"Grupo {i}",
            "Variables asociadas": ", ".join(sorted(comp)),
            "Número de variables": len(comp)
        })

    st.markdown("**Agrupamiento de variables por componentes de la red**")

    st.dataframe(
        pd.DataFrame(resumen_componentes),
        use_container_width=True
    )


def preparar_datos_arbol_causa(df, causa_objetivo, variables_predictoras):
    df_modelo = df.copy()

    df_modelo = df_modelo[
        df_modelo["CAUSA_CIE_4D"].notna()
    ].copy()

    df_modelo["OBJETIVO_CAUSA"] = (
        df_modelo["CAUSA_CIE_4D"] == causa_objetivo
    ).astype(int)

    columnas_modelo = variables_predictoras + ["OBJETIVO_CAUSA"]

    df_modelo = df_modelo[columnas_modelo].copy()

    for col in variables_predictoras:
        df_modelo[col] = df_modelo[col].astype(str).fillna("SIN_DATO")

    X = pd.get_dummies(
        df_modelo[variables_predictoras],
        drop_first=False
    )

    y = df_modelo["OBJETIVO_CAUSA"]

    return X, y


def entrenar_arbol_binario(X, y, modo_parametros):
    if y.nunique() < 2:
        return None, None, None, None

    test_size = 0.30

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    if modo_parametros == "Automático recomendado":
        mejores = []

        for profundidad in [2, 3, 4, 5, 6]:
            for min_leaf in [5, 10, 20, 30]:
                modelo = DecisionTreeClassifier(
                    max_depth=profundidad,
                    min_samples_leaf=min_leaf,
                    class_weight="balanced",
                    random_state=42
                )

                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)

                f1 = f1_score(y_test, y_pred, zero_division=0)

                mejores.append({
                    "modelo": modelo,
                    "profundidad": profundidad,
                    "min_leaf": min_leaf,
                    "f1": f1
                })

        mejor = max(mejores, key=lambda x: x["f1"])
        modelo = mejor["modelo"]

        parametros = {
            "Profundidad máxima": mejor["profundidad"],
            "Mínimo de casos por hoja": mejor["min_leaf"],
            "Criterio de selección": "Mayor F1-score"
        }

    else:
        st.info(
            "La profundidad controla cuántos niveles tendrá el árbol. "
            "Un árbol más profundo puede encontrar reglas más específicas, "
            "pero también puede memorizar los datos. "
            "El mínimo de casos por hoja evita reglas basadas en muy pocos registros."
        )

        col1, col2 = st.columns(2)

        with col1:
            profundidad = st.selectbox(
                "Profundidad máxima del árbol",
                [2, 3, 4, 5, 6, 7, 8],
                index=2
            )

        with col2:
            min_leaf = st.selectbox(
                "Mínimo de casos por hoja",
                [5, 10, 20, 30, 50],
                index=2
            )

        modelo = DecisionTreeClassifier(
            max_depth=profundidad,
            min_samples_leaf=min_leaf,
            class_weight="balanced",
            random_state=42
        )

        modelo.fit(X_train, y_train)

        parametros = {
            "Profundidad máxima": profundidad,
            "Mínimo de casos por hoja": min_leaf,
            "Criterio de selección": "Manual"
        }

    y_pred = modelo.predict(X_test)

    metricas = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0)
    }

    matriz = confusion_matrix(y_test, y_pred)

    return modelo, metricas, matriz, parametros


def mostrar_arbol_causa_muerte(df):
    st.divider()

    st.subheader("Árbol binario para predecir una causa de muerte")

    if "CAUSA_CIE_4D" not in df.columns:
        st.warning("No se encontró la columna CAUSA_CIE_4D.")
        return

    causas = (
        df["CAUSA_CIE_4D"]
        .dropna()
        .astype(str)
        .value_counts()
        .reset_index()
    )

    causas.columns = ["CAUSA_CIE_4D", "Frecuencia"]

    causas["Etiqueta"] = (
        causas["CAUSA_CIE_4D"] +
        " | n=" +
        causas["Frecuencia"].astype(str)
    )

    causa_etiqueta = st.selectbox(
        "Selecciona la causa de muerte a predecir",
        causas["Etiqueta"]
    )

    causa_objetivo = causa_etiqueta.split(" | n=")[0]

    st.markdown(f"**Causa seleccionada:** {causa_objetivo}")

    n_causa = int((df["CAUSA_CIE_4D"].astype(str) == causa_objetivo).sum())
    n_total = int(df["CAUSA_CIE_4D"].notna().sum())

    st.markdown(
        f"**Casos de la causa seleccionada:** {n_causa} de {n_total} "
        f"({(n_causa / n_total) * 100:.2f}%)"
    )

    if n_causa < 20:
        st.warning(
            "Esta causa tiene pocos casos. El árbol puede ser inestable. "
            "Conviene interpretar las reglas con cautela."
        )

    variables_sugeridas = [
        "EDAD",
        "EDAD_QUINQUENALD",
        "ESTADO_CONYUGALD",
        "ENTIDAD_RESIDENCIAD",
        "ENTIDAD_OCURRENCIAD",
        "MUNICIPIO_RESIDENCIAD",
        "MUNICIPIO_OCURRENCIAD",
        "TAMANIO_LOCALIDADD",
        "OCUPACION_HABITUALD",
        "ESCOLARIDADD",
        "DERECHOHABIENCIAD",
        "SITIO_DEFUNCIOND",
        "ASISTENCIA_MEDICAD",
        "CERTIFICOD",
        "ANIO_DEFUNCION"
    ]

    variables_disponibles = [
        v for v in variables_sugeridas if v in df.columns and v != "CAUSA_CIE_4D"
    ]

    variables_predictoras = st.multiselect(
        "Selecciona variables predictoras",
        options=variables_disponibles,
        default=[
            v for v in [
                "EDAD_QUINQUENALD",
                "ESCOLARIDADD",
                "DERECHOHABIENCIAD",
                "TAMANIO_LOCALIDADD",
                "SITIO_DEFUNCIOND",
                "ASISTENCIA_MEDICAD",
                "ANIO_DEFUNCION"
            ] if v in variables_disponibles
        ]
    )

    if len(variables_predictoras) < 2:
        st.warning("Selecciona al menos dos variables predictoras.")
        return

    modo_parametros = st.radio(
        "Configuración del árbol",
        [
            "Automático recomendado",
            "Manual avanzada"
        ],
        horizontal=True
    )

    X, y = preparar_datos_arbol_causa(
        df,
        causa_objetivo,
        variables_predictoras
    )

    if y.sum() < 5:
        st.warning("Hay muy pocos casos positivos para entrenar el árbol.")
        return

    modelo, metricas, matriz, parametros = entrenar_arbol_binario(
        X,
        y,
        modo_parametros
    )

    if modelo is None:
        st.warning("No fue posible entrenar el árbol con esta causa.")
        return



    st.markdown("### Árbol de decisión")
    profundidad_real = modelo.get_depth()

    fig, ax = plt.subplots(
        figsize=(22, 12),
        dpi=120
    )

    plot_tree(
        modelo,
        feature_names=X.columns,
        class_names=["Otras causas", "Causa seleccionada"],
        filled=True,
        rounded=True,
        fontsize=9,
        proportion=True,
        impurity=False,
        ax=ax
    )

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


    st.markdown("### Métricas de desempeño")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{metricas['Accuracy']:.3f}")

    with col2:
        st.metric("Precision", f"{metricas['Precision']:.3f}")

    with col3:
        st.metric("Recall", f"{metricas['Recall']:.3f}")

    with col4:
        st.metric("F1-score", f"{metricas['F1-score']:.3f}")

    st.caption(
        "Recall indica qué proporción de casos reales de la causa seleccionada fueron detectados. "
        "Precision indica qué proporción de los casos predichos como esa causa realmente lo eran. "
        "F1-score resume precision y recall, útil cuando hay desbalance."
    )

    st.markdown("### Matriz de confusión")

    df_matriz = pd.DataFrame(
        matriz,
        index=["Real: otras causas", "Real: causa seleccionada"],
        columns=["Predicho: otras causas", "Predicho: causa seleccionada"]
    )

    st.dataframe(
        df_matriz,
        use_container_width=True
    )

    st.markdown("### Parámetros usados")

    st.dataframe(
        pd.DataFrame(
            parametros.items(),
            columns=["Parámetro", "Valor"]
        ),
        use_container_width=True
    )

    importancias = pd.DataFrame({
        "Variable": X.columns,
        "Importancia": modelo.feature_importances_
    }).sort_values("Importancia", ascending=False)

    st.markdown("### Variables más importantes")

    st.dataframe(
        importancias.head(20),
        use_container_width=True
    )

    fig_imp = px.bar(
        importancias.head(20),
        x="Importancia",
        y="Variable",
        orientation="h",
        title="Top 20 variables predictoras del árbol"
    )

    fig_imp.update_layout(
        height=600,
        yaxis=dict(autorange="reversed")
    )

    st.plotly_chart(
        fig_imp,
        use_container_width=True
    )