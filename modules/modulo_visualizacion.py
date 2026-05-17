import streamlit as st
import pandas as pd
import plotly.express as px

def generar_descripcion_columnas(df):
    filas = []

    for col in df.columns:
        if col.endswith("D"):
            tipo = "Descripción"
            descripcion = f"Descripción textual asociada a la variable {col[:-1]}."
        else:
            tipo = "Variable"
            col_d = f"{col}D"
            descripcion = (
                f"Variable codificada; su descripción está en {col_d}."
                if col_d in df.columns
                else "Variable sin columna descriptiva asociada."
            )

        filas.append({
            "Columna": col,
            "Tipo": tipo,
            "Descripción": descripcion,
            "NA": int(df[col].isna().sum()),
            "% NA": round(df[col].isna().mean() * 100, 2),
            "Ejemplos": ", ".join(df[col].dropna().astype(str).unique()[:5])
        })

    return pd.DataFrame(filas)


def mostrar_modulo_visualizacion(df):
    st.header("Módulo 1: Visualización de tablas y gráficas")

    st.subheader("DataFrame completo")

    st.dataframe(
        df,
        use_container_width=True,
        height=600
    )

    with st.expander("Resumen del archivo, columnas y valores faltantes"):
        st.markdown(f"**Filas:** {df.shape[0]}")
        st.markdown(f"**Columnas:** {df.shape[1]}")

        st.markdown("**Nombres de columnas:**")
        st.write(list(df.columns))

        st.divider()

        st.markdown("**Descripción de columnas y valores NA:**")

        descripcion_columnas = generar_descripcion_columnas(df)

        st.dataframe(
            descripcion_columnas,
            use_container_width=True,
            height=600
        )

    st.divider()

    st.subheader("Vista rápida de variables")

    columna = st.selectbox(
        "Selecciona una columna para explorar",
        df.columns
    )

    conteo = (
        df[columna]
        .value_counts(dropna=False)
        .reset_index()
    )

    conteo.columns = [columna, "Frecuencia"]

    if pd.api.types.is_numeric_dtype(df[columna]):
        conteo = conteo.sort_values(by=columna)
    else:
        conteo = conteo.sort_values(by="Frecuencia", ascending=False)

    st.dataframe(
        conteo,
        use_container_width=True
    )

    st.subheader("Gráfica de frecuencias")

    conteo_grafica = conteo.copy()

    def acortar_texto(x, n=35):
        x = str(x)
        return x if len(x) <= n else x[:n] + "..."

    conteo_grafica["Etiqueta_completa"] = conteo_grafica[columna].astype(str)
    conteo_grafica["Etiqueta_corta"] = conteo_grafica[columna].apply(acortar_texto)

    fig = px.bar(
        conteo_grafica,
        x="Etiqueta_corta",
        y="Frecuencia",
        title=f"Frecuencia de {columna}",
        hover_data={
            "Etiqueta_completa": True,
            "Etiqueta_corta": False,
            "Frecuencia": True
        },
        labels={
            "Etiqueta_corta": columna,
            "Frecuencia": "Frecuencia"
        }
    )

    if pd.api.types.is_numeric_dtype(df[columna]):
        valores_validos = df[columna].dropna()

        q1 = valores_validos.quantile(0.25)
        q3 = valores_validos.quantile(0.75)
        iqr = q3 - q1

        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr

        rango_sin_outliers = valores_validos[
            (valores_validos >= limite_inferior) &
            (valores_validos <= limite_superior)
        ]

        if not rango_sin_outliers.empty:
            fig.update_xaxes(
                range=[
                    rango_sin_outliers.min(),
                    rango_sin_outliers.max()
                ]
            )

    fig.update_layout(
        height=550,
        xaxis_title=columna,
        yaxis_title="Frecuencia",
        margin=dict(l=40, r=40, t=80, b=80),
        xaxis=dict(
            tickangle=-90,
            automargin=True,
            tickfont=dict(size=15)
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    mostrar_frecuencias_filtradas(df)

def obtener_columna_geografica(tipo_ubicacion):
    if tipo_ubicacion == "Entidad de ocurrencia":
        return {
            "estado": "ENTIDAD_OCURRENCIAD",
            "municipio": "MUNICIPIO_OCURRENCIAD",
            "localidad": "LOCALIDAD_OCURRENCIAD"
        }

    return {
        "estado": "ENTIDAD_RESIDENCIAD",
        "municipio": "MUNICIPIO_RESIDENCIAD",
        "localidad": "LOCALIDAD_RESIDENCIAD"
    }


def filtrar_geografia_visualizacion(df):
    st.subheader("Filtro geográfico")

    nivel = st.radio(
        "Selecciona el nivel de análisis",
        [
            "Todo el país",
            "Estado / municipio / localidad"
        ],
        horizontal=True
    )

    if nivel == "Todo el país":
        return df.copy(), "Todo el país"

    tipo_ubicacion = st.selectbox(
        "Usar ubicación por:",
        [
            "Entidad de ocurrencia",
            "Entidad de residencia"
        ]
    )

    columnas_geo = obtener_columna_geografica(tipo_ubicacion)

    df_filtrado = df.copy()

    estados = sorted(df_filtrado[columnas_geo["estado"]].dropna().astype(str).unique())

    estado_sel = st.selectbox(
        "Selecciona estado",
        ["Todos"] + estados
    )

    if estado_sel != "Todos":
        df_filtrado = df_filtrado[
            df_filtrado[columnas_geo["estado"]].astype(str) == estado_sel
        ]

    municipios = sorted(df_filtrado[columnas_geo["municipio"]].dropna().astype(str).unique())

    municipio_sel = st.selectbox(
        "Selecciona municipio",
        ["Todos"] + municipios
    )

    if municipio_sel != "Todos":
        df_filtrado = df_filtrado[
            df_filtrado[columnas_geo["municipio"]].astype(str) == municipio_sel
        ]

    localidades = sorted(df_filtrado[columnas_geo["localidad"]].dropna().astype(str).unique())

    localidad_sel = st.selectbox(
        "Selecciona localidad",
        ["Todos"] + localidades
    )

    if localidad_sel != "Todos":
        df_filtrado = df_filtrado[
            df_filtrado[columnas_geo["localidad"]].astype(str) == localidad_sel
        ]

    resumen = f"{tipo_ubicacion}"

    if estado_sel != "Todos":
        resumen += f" | Estado: {estado_sel}"

    if municipio_sel != "Todos":
        resumen += f" | Municipio: {municipio_sel}"

    if localidad_sel != "Todos":
        resumen += f" | Localidad: {localidad_sel}"

    return df_filtrado, resumen


def crear_catalogo_variables(df):
    variables = []

    for col in df.columns:
        if col.endswith("D"):
            variables.append(col)

    variables_extra = [
        "EDAD",
        "ANIO_DEFUNCION",
        "ANIO_NACIMIENTO",
        "ANIO_REGISTRO",
        "ANIO_CERTIFICACION"
    ]

    for col in variables_extra:
        if col in df.columns:
            variables.append(col)

    return sorted(list(set(variables)))

def filtrar_outliers_variable(serie):
    serie_num = pd.to_numeric(serie, errors="coerce")

    q1 = serie_num.quantile(0.25)
    q3 = serie_num.quantile(0.75)
    iqr = q3 - q1

    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    return serie_num.where(
        serie_num.between(limite_inferior, limite_superior)
    )


def tabla_frecuencias_variable(df, variable):
    df_temp = df.copy()
    serie = df_temp[variable]

    if pd.api.types.is_numeric_dtype(serie):

        serie_num_original = pd.to_numeric(serie, errors="coerce")
        serie_num = filtrar_outliers_variable(serie_num_original)

        outliers_excluidos = serie_num_original.notna().sum() - serie_num.notna().sum()

        if outliers_excluidos > 0:
            st.warning(
                f"Se excluyeron {outliers_excluidos} valores extremos para calcular los intervalos."
            )



        valores_unicos = serie_num.dropna().nunique()

        if valores_unicos > 20:
            min_val = int(serie_num.min())
            max_val = int(serie_num.max())

            tam_intervalo = st.number_input(
                "Tamaño del intervalo",
                min_value=1,
                max_value=max(1, max_val - min_val),
                value=max(1, round((max_val - min_val) / 10)),
                step=1
            )

            bins = list(range(min_val, max_val + tam_intervalo, tam_intervalo))

            tabla = (
                pd.cut(
                    serie_num,
                    bins=bins,
                    include_lowest=True
                )
                .value_counts(dropna=False)
                .sort_index()
                .reset_index()
            )

            tabla.columns = ["Categoría / rango", "Frecuencia"]

        else:
            tabla = (
                serie_num
                .value_counts(dropna=False)
                .sort_index()
                .reset_index()
            )

            tabla.columns = ["Categoría / rango", "Frecuencia"]

    else:
        tabla = (
            serie
            .astype("object")
            .where(serie.notna(), "SIN DATO")
            .value_counts(dropna=False)
            .reset_index()
        )

        tabla.columns = ["Categoría / rango", "Frecuencia"]

    tabla["Porcentaje"] = (
        tabla["Frecuencia"] / tabla["Frecuencia"].sum() * 100
    ).round(2)

    return tabla


def mostrar_frecuencias_filtradas(df):
    st.divider()

    st.header("Tablas de frecuencia por ubicación")

    st.info(
        "En esta sección puedes analizar frecuencias para todo el país o filtrar por estado, "
        "municipio y localidad. Se priorizan las columnas terminadas en D porque contienen "
        "las descripciones de las variables codificadas."
    )

    df_filtrado, resumen = filtrar_geografia_visualizacion(df)

    st.markdown(f"**Filtro aplicado:** {resumen}")
    st.markdown(f"**Registros incluidos:** {df_filtrado.shape[0]}")

    if df_filtrado.empty:
        st.warning("No hay registros con los filtros seleccionados.")
        return

    variables_disponibles = crear_catalogo_variables(df_filtrado)

    variable = st.selectbox(
        "Selecciona la variable de interés",
        variables_disponibles
    )

    st.markdown(f"**Variable seleccionada:** `{variable}`")

    tabla = tabla_frecuencias_variable(df_filtrado, variable)

    st.subheader("Tabla de frecuencias")

    st.dataframe(
        tabla,
        use_container_width=True
    )

    st.subheader("Gráfica de frecuencias")

    tabla_grafica = tabla.copy()
    tabla_grafica["Etiqueta completa"] = tabla_grafica["Categoría / rango"].astype(str)

    tabla_grafica["Etiqueta corta"] = tabla_grafica["Etiqueta completa"].apply(
        lambda x: x if len(x) <= 25 else x[:25] + "..."
    )

    fig = px.bar(
        tabla_grafica,
        x="Etiqueta corta",
        y="Frecuencia",
        hover_data={
            "Etiqueta completa": True,
            "Etiqueta corta": False,
            "Frecuencia": True,
            "Porcentaje": True
        },
        title=f"Frecuencia de {variable}",
        labels={
            "Etiqueta corta": variable,
            "Frecuencia": "Frecuencia"
        }
    )

    fig.update_layout(
        height=600,
        xaxis=dict(
            tickangle=-90,
            automargin=True,
            tickfont=dict(size=9)
        ),
        margin=dict(l=40, r=40, t=80, b=200)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )