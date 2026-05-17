import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.nonparametric.smoothers_lowess import lowess
from prophet import Prophet
import json
import os


# ============================================================
# UTILIDADES GENERALES
# ============================================================

def filtrar_outliers_iqr(serie):
    serie = pd.to_numeric(serie, errors="coerce").dropna()

    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1

    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    return serie[
        (serie >= limite_inferior) &
        (serie <= limite_superior)
    ]


def calcular_metricas(y_real, y_pred):
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)
    return rmse, r2


def calcular_metricas_error(y_real, y_pred):
    mae = np.mean(np.abs(y_real - y_pred))
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    return mae, rmse


def normalizar_estado(x):
    if pd.isna(x):
        return None

    x = str(x).strip().upper()

    mapa = {
        "AGUASCALIENTES": "Aguascalientes",
        "BAJA CALIFORNIA": "Baja California",
        "BAJA CALIFORNIA SUR": "Baja California Sur",
        "CAMPECHE": "Campeche",
        "COAHUILA": "Coahuila",
        "COAHUILA DE ZARAGOZA": "Coahuila",
        "COLIMA": "Colima",
        "CHIAPAS": "Chiapas",
        "CHIHUAHUA": "Chihuahua",
        "CIUDAD DE MEXICO": "Ciudad de México",
        "CIUDAD DE MÉXICO": "Ciudad de México",
        "DISTRITO FEDERAL": "Ciudad de México",
        "DURANGO": "Durango",
        "GUANAJUATO": "Guanajuato",
        "GUERRERO": "Guerrero",
        "HIDALGO": "Hidalgo",
        "JALISCO": "Jalisco",
        "MEXICO": "México",
        "MÉXICO": "México",
        "ESTADO DE MEXICO": "México",
        "MICHOACAN": "Michoacán",
        "MICHOACÁN": "Michoacán",
        "MICHOACAN DE OCAMPO": "Michoacán",
        "MORELOS": "Morelos",
        "NAYARIT": "Nayarit",
        "NUEVO LEON": "Nuevo León",
        "NUEVO LEÓN": "Nuevo León",
        "OAXACA": "Oaxaca",
        "PUEBLA": "Puebla",
        "QUERETARO": "Querétaro",
        "QUERÉTARO": "Querétaro",
        "QUERETARO DE ARTEAGA": "Querétaro",
        "QUINTANA ROO": "Quintana Roo",
        "SAN LUIS POTOSI": "San Luis Potosí",
        "SAN LUIS POTOSÍ": "San Luis Potosí",
        "SINALOA": "Sinaloa",
        "SONORA": "Sonora",
        "TABASCO": "Tabasco",
        "TAMAULIPAS": "Tamaulipas",
        "TLAXCALA": "Tlaxcala",
        "VERACRUZ": "Veracruz",
        "VERACRUZ DE IGNACIO DE LA LLAVE": "Veracruz",
        "YUCATAN": "Yucatán",
        "YUCATÁN": "Yucatán",
        "ZACATECAS": "Zacatecas"
    }

    return mapa.get(x, None)


def cargar_geojson_mexico():
    ruta_actual = os.path.dirname(__file__)
    ruta_json = os.path.join(ruta_actual, "mexicoHigh.json")

    with open(ruta_json, "r", encoding="utf-8") as f:
        return json.load(f)


def completar_estados_mapa(df_mapa, valor_col="Valor"):
    estados_mexico = [
        "Aguascalientes", "Baja California", "Baja California Sur", "Campeche",
        "Coahuila", "Colima", "Chiapas", "Chihuahua", "Ciudad de México",
        "Durango", "Guanajuato", "Guerrero", "Hidalgo", "Jalisco", "México",
        "Michoacán", "Morelos", "Nayarit", "Nuevo León", "Oaxaca", "Puebla",
        "Querétaro", "Quintana Roo", "San Luis Potosí", "Sinaloa", "Sonora",
        "Tabasco", "Tamaulipas", "Tlaxcala", "Veracruz", "Yucatán", "Zacatecas"
    ]

    if "Mapa" not in df_mapa.columns:
        df_mapa["Mapa"] = "Mapa"

    mapas = df_mapa["Mapa"].unique()
    completos = []

    for nombre_mapa in mapas:
        df_sub = df_mapa[df_mapa["Mapa"] == nombre_mapa].copy()
        faltantes = sorted(set(estados_mexico) - set(df_sub["Estado"]))

        if faltantes:
            df_faltantes = pd.DataFrame({
                "Estado": faltantes,
                valor_col: 0,
                "Mapa": nombre_mapa
            })

            df_sub = pd.concat([df_sub, df_faltantes], ignore_index=True)

        completos.append(df_sub)

    return pd.concat(completos, ignore_index=True)


# ============================================================
# FILTROS Y PREPARACIÓN DE SERIE
# ============================================================

def crear_catalogo_variables_tendencia(df):
    variables = []

    for col in df.columns:
        if col.endswith("D"):
            variables.append(col)

    extras = [
        "EDAD",
        "ANIO_DEFUNCION",
        "MES_DEFUNCION",
        "HORA_DEFUNCION",
        "MINUTOS_DEFUNCION"
    ]

    for col in extras:
        if col in df.columns:
            variables.append(col)

    return sorted(list(set(variables)))


def aplicar_filtros_unificados(df):
    st.subheader("Filtros de análisis")

    df_filtrado = df.copy()

    tipo_ubicacion = st.radio(
        "Ubicación para filtrar",
        [
            "Todo México",
            "Entidad de ocurrencia",
            "Entidad de residencia"
        ],
        horizontal=True,
        key="filtro_tipo_ubicacion_unificado"
    )

    if tipo_ubicacion == "Entidad de ocurrencia":
        col_estado = "ENTIDAD_OCURRENCIAD"
        col_municipio = "MUNICIPIO_OCURRENCIAD"
        col_localidad = "LOCALIDAD_OCURRENCIAD"

    elif tipo_ubicacion == "Entidad de residencia":
        col_estado = "ENTIDAD_RESIDENCIAD"
        col_municipio = "MUNICIPIO_RESIDENCIAD"
        col_localidad = "LOCALIDAD_RESIDENCIAD"

    else:
        col_estado = "ENTIDAD_OCURRENCIAD"
        col_municipio = "MUNICIPIO_OCURRENCIAD"
        col_localidad = "LOCALIDAD_OCURRENCIAD"

    if tipo_ubicacion != "Todo México":
        estados = sorted(df_filtrado[col_estado].dropna().astype(str).unique())

        estados_sel = st.multiselect(
            "Estado(s)",
            options=estados,
            default=[],
            key="filtro_estados_unificado"
        )

        if estados_sel:
            df_filtrado = df_filtrado[
                df_filtrado[col_estado].astype(str).isin(estados_sel)
            ]

        municipios = sorted(df_filtrado[col_municipio].dropna().astype(str).unique())

        municipios_sel = st.multiselect(
            "Municipio(s)",
            options=municipios,
            default=[],
            key="filtro_municipios_unificado"
        )

        if municipios_sel:
            df_filtrado = df_filtrado[
                df_filtrado[col_municipio].astype(str).isin(municipios_sel)
            ]

        localidades = sorted(df_filtrado[col_localidad].dropna().astype(str).unique())

        localidades_sel = st.multiselect(
            "Localidad(es)",
            options=localidades,
            default=[],
            key="filtro_localidades_unificado"
        )

        if localidades_sel:
            df_filtrado = df_filtrado[
                df_filtrado[col_localidad].astype(str).isin(localidades_sel)
            ]

    else:
        estados_sel = []
        municipios_sel = []
        localidades_sel = []

    if "CAUSA_CIE_4D" in df_filtrado.columns:
        causas = sorted(df_filtrado["CAUSA_CIE_4D"].dropna().astype(str).unique())

        causas_sel = st.multiselect(
            "Causa(s) de muerte",
            options=causas,
            default=[],
            help="Si no seleccionas ninguna causa, se incluyen todas.",
            key="filtro_causas_unificado"
        )

        if causas_sel:
            df_filtrado = df_filtrado[
                df_filtrado["CAUSA_CIE_4D"].astype(str).isin(causas_sel)
            ]
    else:
        causas_sel = []

    resumen = {
        "tipo_ubicacion": tipo_ubicacion,
        "col_estado": col_estado,
        "col_municipio": col_municipio,
        "col_localidad": col_localidad,
        "estados": estados_sel,
        "municipios": municipios_sel,
        "localidades": localidades_sel,
        "causas": causas_sel
    }

    return df_filtrado, resumen


def preparar_serie_anual(df, modo_analisis, variable=None, categorias_sel=None):
    df_temp = df.copy()

    df_temp["ANIO_DEFUNCION"] = pd.to_numeric(
        df_temp["ANIO_DEFUNCION"],
        errors="coerce"
    )

    df_temp = df_temp[df_temp["ANIO_DEFUNCION"].notna()].copy()
    df_temp = df_temp[df_temp["ANIO_DEFUNCION"] >= 2000].copy()
    df_temp["ANIO_DEFUNCION"] = df_temp["ANIO_DEFUNCION"].astype(int)

    if modo_analisis == "Defunciones totales":
        df_anual = (
            df_temp
            .groupby("ANIO_DEFUNCION")
            .size()
            .reset_index(name="Defunciones")
            .sort_values("ANIO_DEFUNCION")
        )

        etiqueta = "Defunciones totales"

    elif modo_analisis == "Variable numérica":
        df_temp[variable] = pd.to_numeric(df_temp[variable], errors="coerce")
        df_temp = df_temp[df_temp[variable].notna()].copy()

        valores_validos = filtrar_outliers_iqr(df_temp[variable])
        df_temp = df_temp[df_temp[variable].isin(valores_validos)].copy()

        df_anual = (
            df_temp
            .groupby("ANIO_DEFUNCION")[variable]
            .mean()
            .reset_index(name="Defunciones")
            .sort_values("ANIO_DEFUNCION")
        )

        etiqueta = f"Promedio anual de {variable}"

    else:
        if variable and categorias_sel:
            df_temp = df_temp[
                df_temp[variable].astype(str).isin(categorias_sel)
            ].copy()

        df_anual = (
            df_temp
            .groupby("ANIO_DEFUNCION")
            .size()
            .reset_index(name="Defunciones")
            .sort_values("ANIO_DEFUNCION")
        )

        etiqueta = f"Frecuencia anual de {variable}"

    return df_anual, etiqueta


def preparar_serie_categorias(df, variable, categorias_sel=None):
    df_temp = df.copy()

    df_temp["ANIO_DEFUNCION"] = pd.to_numeric(
        df_temp["ANIO_DEFUNCION"],
        errors="coerce"
    )

    df_temp = df_temp[
        df_temp["ANIO_DEFUNCION"].notna() &
        df_temp[variable].notna()
    ].copy()

    df_temp = df_temp[df_temp["ANIO_DEFUNCION"] >= 2000].copy()
    df_temp["ANIO_DEFUNCION"] = df_temp["ANIO_DEFUNCION"].astype(int)
    df_temp[variable] = df_temp[variable].astype(str)

    if categorias_sel:
        df_temp = df_temp[df_temp[variable].isin(categorias_sel)]

    df_serie = (
        df_temp
        .groupby(["ANIO_DEFUNCION", variable])
        .size()
        .reset_index(name="Valor")
        .rename(columns={variable: "Categoría"})
    )

    return df_serie


# ============================================================
# GRÁFICOS TEMPORALES
# ============================================================

def graficar_serie_simple(df_anual, etiqueta):
    st.subheader("Tendencia anual")

    st.dataframe(
        df_anual,
        use_container_width=True
    )

    fig = px.line(
        df_anual,
        x="ANIO_DEFUNCION",
        y="Defunciones",
        markers=True,
        title=f"Tendencia anual: {etiqueta}",
        labels={
            "ANIO_DEFUNCION": "Año de defunción",
            "Defunciones": etiqueta
        }
    )

    fig.update_layout(
        height=550,
        xaxis=dict(
            tickmode="linear",
            dtick=1
        ),
        yaxis_title=etiqueta,
        xaxis_title="Año de defunción"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def graficar_categorias_temporales(df_filtrado, variable):
    if variable is None:
        return

    if pd.api.types.is_numeric_dtype(df_filtrado[variable]):
        return

    st.subheader("Tendencia por categorías")

    categorias = sorted(df_filtrado[variable].dropna().astype(str).unique())

    categorias_sel = st.multiselect(
        "Categorías a mostrar en la gráfica temporal",
        options=categorias,
        default=categorias[:5],
        key="categorias_temporales_unificadas"
    )

    df_serie = preparar_serie_categorias(
        df_filtrado,
        variable,
        categorias_sel
    )

    if df_serie.empty:
        st.warning("No hay datos suficientes para graficar categorías.")
        return

    tipo_grafico = st.selectbox(
        "Tipo de gráfica por categorías",
        ["Líneas", "Barras"],
        key="tipo_grafico_categorias_unificadas"
    )

    if tipo_grafico == "Líneas":
        fig = px.line(
            df_serie,
            x="ANIO_DEFUNCION",
            y="Valor",
            color="Categoría",
            markers=True,
            title=f"Evolución temporal por categoría: {variable}"
        )
    else:
        fig = px.bar(
            df_serie,
            x="ANIO_DEFUNCION",
            y="Valor",
            color="Categoría",
            title=f"Evolución temporal por categoría: {variable}"
        )

    fig.update_layout(
        height=650,
        xaxis=dict(
            tickmode="linear",
            dtick=1
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        margin=dict(l=40, r=40, t=80, b=220)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def preparar_series_multiples(df, col_grupo, valor_col="Defunciones"):
    df_temp = df.copy()

    df_temp["ANIO_DEFUNCION"] = pd.to_numeric(
        df_temp["ANIO_DEFUNCION"],
        errors="coerce"
    )

    df_temp = df_temp[
        df_temp["ANIO_DEFUNCION"].notna() &
        df_temp[col_grupo].notna()
    ].copy()

    df_temp = df_temp[df_temp["ANIO_DEFUNCION"] >= 2000].copy()

    df_series = (
        df_temp
        .groupby(["ANIO_DEFUNCION", col_grupo])
        .size()
        .reset_index(name=valor_col)
    )

    df_series["ANIO_DEFUNCION"] = df_series["ANIO_DEFUNCION"].astype(int)
    df_series[col_grupo] = df_series[col_grupo].astype(str)

    df_promedio = (
        df_series
        .groupby("ANIO_DEFUNCION")[valor_col]
        .mean()
        .reset_index(name="Promedio")
    )

    promedio_grupal = df_series[valor_col].mean()

    promedios_individuales = (
        df_series
        .groupby(col_grupo)[valor_col]
        .mean()
        .reset_index(name="Promedio_individual")
    )

    promedios_individuales["Comparacion_promedio"] = promedios_individuales[
        "Promedio_individual"
    ].apply(
        lambda x: "Arriba del promedio"
        if x > promedio_grupal
        else "Igual o debajo del promedio"
    )

    return df_series, df_promedio, promedios_individuales, promedio_grupal


def definir_nivel_grafico_multiple(resumen):
    tipo = resumen["tipo_ubicacion"]

    if tipo == "Todo México":
        return "Estado", resumen["col_estado"], "Promedio nacional por año"

    if resumen["municipios"]:
        return "Localidad", resumen["col_localidad"], "Promedio del municipio por año"

    if resumen["estados"]:
        return "Municipio", resumen["col_municipio"], "Promedio del estado por año"

    return "Estado", resumen["col_estado"], "Promedio nacional por año"


def graficar_series_multiples(df, resumen, etiqueta):
    st.subheader("Gráfico múltiple comparativo")

    nombre_nivel, col_grupo, nombre_promedio = definir_nivel_grafico_multiple(resumen)

    df_series, df_promedio, promedios_individuales, promedio_grupal = preparar_series_multiples(
        df,
        col_grupo
    )

    if df_series.empty:
        st.warning("No hay datos suficientes para el gráfico múltiple.")
        return

    st.markdown(f"**Promedio grupal:** {promedio_grupal:.2f} casos por año")

    opcion_curvas = st.selectbox(
        "Curvas a mostrar",
        [
            "Todas",
            "Solo arriba del promedio",
            "Solo igual o debajo del promedio"
        ],
        key="curvas_multiples_unificadas"
    )

    if opcion_curvas == "Solo arriba del promedio":
        grupos_validos = promedios_individuales[
            promedios_individuales["Comparacion_promedio"] == "Arriba del promedio"
        ][col_grupo].tolist()

    elif opcion_curvas == "Solo igual o debajo del promedio":
        grupos_validos = promedios_individuales[
            promedios_individuales["Comparacion_promedio"] == "Igual o debajo del promedio"
        ][col_grupo].tolist()

    else:
        grupos_validos = promedios_individuales[col_grupo].tolist()

    df_series = df_series[df_series[col_grupo].isin(grupos_validos)]

    with st.expander("Promedios individuales usados para clasificar las curvas"):
        st.dataframe(
            promedios_individuales.sort_values(
                "Promedio_individual",
                ascending=False
            ),
            use_container_width=True
        )

    fig = go.Figure()

    grupos = sorted(df_series[col_grupo].dropna().astype(str).unique())

    for grupo in grupos:
        datos_grupo = df_series[df_series[col_grupo] == grupo]

        fig.add_trace(
            go.Scatter(
                x=datos_grupo["ANIO_DEFUNCION"],
                y=datos_grupo["Defunciones"],
                mode="lines+markers",
                name=grupo,
                line=dict(width=1.5),
                marker=dict(size=5),
                hovertemplate=(
                    f"{nombre_nivel}: {grupo}<br>"
                    "Año: %{x}<br>"
                    "Valor: %{y}<extra></extra>"
                )
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df_promedio["ANIO_DEFUNCION"],
            y=df_promedio["Promedio"],
            mode="lines+markers",
            name=nombre_promedio,
            line=dict(width=5, dash="dot", color="black"),
            marker=dict(size=8, color="black")
        )
    )

    fig.update_layout(
        title=f"Gráfico múltiple: {etiqueta} por {nombre_nivel.lower()}",
        height=750,
        xaxis=dict(
            title="Año de defunción",
            tickmode="linear",
            dtick=1
        ),
        yaxis=dict(
            title=etiqueta
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        margin=dict(l=40, r=40, t=80, b=220)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# MODELOS
# ============================================================

def ajuste_polinomial(df_anual, modo_parametros, key_prefix):
    df_fit = df_anual.copy()
    df_fit["ANIO_CENTRADO"] = df_fit["ANIO_DEFUNCION"] - 2000

    x = df_fit["ANIO_CENTRADO"].values
    y = df_fit["Defunciones"].values

    max_grado = min(10, len(df_fit) - 1)

    if modo_parametros == "Automática recomendada":
        resultados = []

        for grado in range(1, max_grado + 1):
            coef = np.polyfit(x, y, grado)
            modelo = np.poly1d(coef)
            y_pred = modelo(x)
            rmse, r2 = calcular_metricas(y, y_pred)

            resultados.append({
                "grado": grado,
                "rmse": rmse,
                "r2": r2,
                "coeficientes": coef,
                "pred": y_pred
            })

        mejor = min(resultados, key=lambda z: z["rmse"])
        grado = mejor["grado"]
        coeficientes = mejor["coeficientes"]
        y_pred = mejor["pred"]
        rmse = mejor["rmse"]
        r2 = mejor["r2"]

    else:
        st.info(
            "El grado indica qué tan flexible será la curva. "
            "Grados altos pueden sobreajustar los datos."
        )

        grado = st.selectbox(
            "Selecciona el grado del polinomio",
            options=list(range(1, max_grado + 1)),
            index=1 if max_grado >= 2 else 0,
            key=f"{key_prefix}_grado_polinomial"
        )

        coeficientes = np.polyfit(x, y, grado)
        modelo = np.poly1d(coeficientes)
        y_pred = modelo(x)
        rmse, r2 = calcular_metricas(y, y_pred)

    df_fit["Prediccion"] = y_pred

    df_parametros = pd.DataFrame({
        "Parámetro": [f"x^{grado - i}" for i in range(len(coeficientes))],
        "Valor": coeficientes
    })

    return df_fit, df_parametros, rmse, r2, f"Polinomial grado {grado}"


def ajuste_lowess(df_anual, modo_parametros, key_prefix):
    df_fit = df_anual.copy()

    x = df_fit["ANIO_DEFUNCION"].values
    y = df_fit["Defunciones"].values

    if modo_parametros == "Automática recomendada":
        frac = 0.35

    else:
        st.info(
            "LOWESS suaviza la tendencia histórica. "
            "No es un modelo formal de predicción futura."
        )

        frac = st.slider(
            "Nivel de suavizado LOWESS",
            min_value=0.1,
            max_value=1.0,
            value=0.35,
            step=0.05,
            key=f"{key_prefix}_lowess_frac"
        )

    suavizado = lowess(
        y,
        x,
        frac=frac,
        return_sorted=False
    )

    rmse, r2 = calcular_metricas(y, suavizado)

    df_fit["Prediccion"] = suavizado

    df_parametros = pd.DataFrame({
        "Parámetro": ["Frac"],
        "Valor": [frac]
    })

    return df_fit, df_parametros, rmse, r2, f"LOWESS frac={frac}"


def ajuste_prophet(df_anual, modo_parametros, key_prefix):
    df_fit = df_anual.copy()

    prophet_df = pd.DataFrame({
        "ds": pd.to_datetime(df_fit["ANIO_DEFUNCION"].astype(str) + "-01-01"),
        "y": df_fit["Defunciones"]
    })

    if modo_parametros == "Automática recomendada":
        changepoint_prior_scale = 0.1
        seasonality_prior_scale = 10.0

    else:
        st.info(
            "Prophet detecta cambios de tendencia. "
            "Valores altos permiten curvas más flexibles."
        )

        changepoint_prior_scale = st.slider(
            "Flexibilidad de tendencia",
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            step=0.01,
            key=f"{key_prefix}_prophet_changepoint"
        )

        seasonality_prior_scale = st.slider(
            "Peso de estacionalidad",
            min_value=1.0,
            max_value=50.0,
            value=10.0,
            step=1.0,
            key=f"{key_prefix}_prophet_seasonality"
        )

    modelo = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False
    )

    modelo.fit(prophet_df)

    forecast = modelo.predict(prophet_df)

    y_pred = forecast["yhat"].values
    rmse, r2 = calcular_metricas(prophet_df["y"], y_pred)

    df_fit["Prediccion"] = y_pred

    df_parametros = pd.DataFrame({
        "Parámetro": [
            "changepoint_prior_scale",
            "seasonality_prior_scale"
        ],
        "Valor": [
            changepoint_prior_scale,
            seasonality_prior_scale
        ]
    })

    return df_fit, df_parametros, rmse, r2, "Prophet"


def ajuste_arbol_regresion(df_anual, modo_parametros, key_prefix):
    df_fit = df_anual.copy()
    df_fit["ANIO_CENTRADO"] = df_fit["ANIO_DEFUNCION"] - 2000

    x = df_fit[["ANIO_CENTRADO"]].values
    y = df_fit["Defunciones"].values

    if modo_parametros == "Automática recomendada":
        resultados = []

        for profundidad in range(1, 6):
            modelo = DecisionTreeRegressor(
                max_depth=profundidad,
                random_state=42
            )

            modelo.fit(x, y)
            y_pred = modelo.predict(x)
            rmse, r2 = calcular_metricas(y, y_pred)

            resultados.append({
                "profundidad": profundidad,
                "rmse": rmse,
                "r2": r2,
                "pred": y_pred
            })

        mejor = min(resultados, key=lambda z: z["rmse"])
        profundidad = mejor["profundidad"]
        y_pred = mejor["pred"]
        rmse = mejor["rmse"]
        r2 = mejor["r2"]

    else:
        st.info(
            "La profundidad controla cuántas divisiones puede hacer el árbol. "
            "No se recomienda para extrapolación futura."
        )

        profundidad = st.selectbox(
            "Profundidad máxima del árbol",
            options=[1, 2, 3, 4, 5],
            index=2,
            key=f"{key_prefix}_arbol_profundidad"
        )

        modelo = DecisionTreeRegressor(
            max_depth=profundidad,
            random_state=42
        )

        modelo.fit(x, y)
        y_pred = modelo.predict(x)
        rmse, r2 = calcular_metricas(y, y_pred)

    df_fit["Prediccion"] = y_pred

    df_parametros = pd.DataFrame({
        "Parámetro": ["Profundidad máxima"],
        "Valor": [profundidad]
    })

    return df_fit, df_parametros, rmse, r2, f"Árbol profundidad {profundidad}"


def mostrar_modelos_tendencia(df_anual, etiqueta, key_prefix="modelo"):
    st.divider()

    st.subheader("Modelos de tendencia")

    if df_anual.shape[0] < 3:
        st.warning("Se requieren al menos 3 años con datos para ajustar modelos.")
        return

    modelo_seleccionado = st.selectbox(
        "Selecciona el modelo de tendencia",
        [
            "Ajuste polinomial",
            "LOWESS",
            "Prophet",
            "Árbol de regresión"
        ],
        key=f"{key_prefix}_modelo_tendencia"
    )

    modo_parametros = st.radio(
        "Configuración del modelo",
        [
            "Automática recomendada",
            "Manual avanzada"
        ],
        horizontal=True,
        key=f"{key_prefix}_modo_parametros"
    )

    st.caption(
        "Para los modelos que usan año como predictor, se usa x = ANIO_DEFUNCION - 2000."
    )

    anio_min = int(df_anual["ANIO_DEFUNCION"].min())
    anio_max = int(df_anual["ANIO_DEFUNCION"].max())

    anio_inicio_modelo = st.number_input(
        "Usar datos desde el año:",
        min_value=anio_min,
        max_value=anio_max,
        value=anio_min,
        step=1,
        key=f"{key_prefix}_anio_inicio_modelo"
    )

    df_anual_modelo = df_anual[
        df_anual["ANIO_DEFUNCION"] >= anio_inicio_modelo
    ].copy()

    if df_anual_modelo.shape[0] < 3:
        st.warning("Con ese año inicial quedan menos de 3 observaciones.")
        return

    if modelo_seleccionado == "Ajuste polinomial":
        resultado = ajuste_polinomial(df_anual_modelo, modo_parametros, key_prefix)

    elif modelo_seleccionado == "LOWESS":
        resultado = ajuste_lowess(df_anual_modelo, modo_parametros, key_prefix)

    elif modelo_seleccionado == "Prophet":
        resultado = ajuste_prophet(df_anual_modelo, modo_parametros, key_prefix)

    else:
        resultado = ajuste_arbol_regresion(df_anual_modelo, modo_parametros, key_prefix)

    df_fit, df_parametros, rmse, r2, nombre_modelo = resultado

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_fit["ANIO_DEFUNCION"],
            y=df_fit["Defunciones"],
            mode="markers+lines",
            name="Observado",
            marker=dict(size=8),
            line=dict(width=2)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_fit["ANIO_DEFUNCION"],
            y=df_fit["Prediccion"],
            mode="lines+markers",
            name=nombre_modelo,
            line=dict(width=4, color="black", dash="dash"),
            marker=dict(size=7, color="black")
        )
    )

    fig.update_layout(
        title=f"{nombre_modelo} | R² = {r2:.4f} | RMSE = {rmse:.2f}",
        height=600,
        xaxis=dict(
            title="Año de defunción",
            tickmode="linear",
            dtick=1
        ),
        yaxis=dict(
            title=etiqueta
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=80, b=160)
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("R²", f"{r2:.4f}")

    with col2:
        st.metric("RMSE", f"{rmse:.2f}")

    st.markdown("**Parámetros del modelo:**")

    st.dataframe(df_parametros, use_container_width=True)

    with st.expander("Ver datos observados y ajustados"):
        st.dataframe(
            df_fit[["ANIO_DEFUNCION", "Defunciones", "Prediccion"]],
            use_container_width=True
        )


# ============================================================
# PREDICCIÓN FUTURA
# ============================================================

def prediccion_futura_prophet(df_anual, anios_futuros):
    df_temp = df_anual.copy().sort_values("ANIO_DEFUNCION")

    prophet_df = pd.DataFrame({
        "ds": pd.to_datetime(df_temp["ANIO_DEFUNCION"].astype(str) + "-01-01"),
        "y": df_temp["Defunciones"]
    })

    modelo = Prophet(
        changepoint_prior_scale=0.1,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False
    )

    modelo.fit(prophet_df)

    futuro = modelo.make_future_dataframe(
        periods=anios_futuros,
        freq="YS"
    )

    forecast = modelo.predict(futuro)
    forecast["ANIO_DEFUNCION"] = forecast["ds"].dt.year

    df_pred_historica = forecast[
        forecast["ANIO_DEFUNCION"].isin(df_temp["ANIO_DEFUNCION"])
    ].copy()

    y_real = df_temp["Defunciones"].values
    y_pred = df_pred_historica["yhat"].values

    mae, rmse = calcular_metricas_error(y_real, y_pred)

    return forecast, mae, rmse


def prediccion_futura_polinomial(df_anual, anios_futuros, grado=2):
    df_temp = df_anual.copy().sort_values("ANIO_DEFUNCION")
    df_temp["ANIO_CENTRADO"] = df_temp["ANIO_DEFUNCION"] - 2000

    x = df_temp["ANIO_CENTRADO"].values
    y = df_temp["Defunciones"].values

    coeficientes = np.polyfit(x, y, grado)
    modelo = np.poly1d(coeficientes)

    y_pred_hist = modelo(x)
    mae, rmse = calcular_metricas_error(y, y_pred_hist)

    ultimo_anio = int(df_temp["ANIO_DEFUNCION"].max())

    anios_futuros_array = np.arange(
        ultimo_anio + 1,
        ultimo_anio + anios_futuros + 1
    )

    x_futuro = anios_futuros_array - 2000
    y_futuro = np.clip(modelo(x_futuro), 0, None)

    df_forecast = pd.DataFrame({
        "ANIO_DEFUNCION": list(df_temp["ANIO_DEFUNCION"]) + list(anios_futuros_array),
        "Observado": list(df_temp["Defunciones"]) + [np.nan] * anios_futuros,
        "Prediccion": list(y_pred_hist) + list(y_futuro),
        "Tipo": ["Histórico"] * len(df_temp) + ["Predicción futura"] * anios_futuros
    })

    df_coef = pd.DataFrame({
        "Parámetro": [f"x^{grado - i}" for i in range(len(coeficientes))],
        "Coeficiente": coeficientes
    })

    return df_forecast, df_coef, mae, rmse


def mostrar_prediccion_futura(df_anual, etiqueta, key_prefix="pred"):
    st.divider()

    st.subheader("Predicción futura con estimación de error")

    st.info(
        "Para predicción futura se recomienda Prophet. "
        "El polinomial se conserva como comparación exploratoria. "
        "LOWESS y árbol se usan para tendencia histórica, no para extrapolar."
    )

    if df_anual.shape[0] < 6:
        st.warning("Se requieren al menos 6 años con datos para predicción futura.")
        return

    anio_min = int(df_anual["ANIO_DEFUNCION"].min())
    anio_max = int(df_anual["ANIO_DEFUNCION"].max())

    if anio_max - 5 < anio_min:
        st.warning("No hay suficientes años para seleccionar una ventana de predicción.")
        return

    anio_inicio_prediccion = st.number_input(
        "Usar datos para predicción desde el año:",
        min_value=anio_min,
        max_value=anio_max - 5,
        value=max(2003, anio_min),
        step=1,
        key=f"{key_prefix}_anio_inicio_prediccion"
    )

    df_pred_base = df_anual[
        df_anual["ANIO_DEFUNCION"] >= anio_inicio_prediccion
    ].copy()

    if df_pred_base.shape[0] < 6:
        st.warning("Con ese año inicial quedan menos de 6 observaciones.")
        return

    anios_futuros = st.number_input(
        "Años a predecir hacia el futuro",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        key=f"{key_prefix}_anios_futuros"
    )

    if anios_futuros > 5:
        st.warning(
            "Predicciones mayores a 5 años deben interpretarse como escenarios exploratorios."
        )

    modelo_prediccion = st.selectbox(
        "Modelo para predicción futura",
        [
            "Prophet recomendado",
            "Polinomial exploratorio"
        ],
        key=f"{key_prefix}_modelo_prediccion"
    )

    if modelo_prediccion == "Prophet recomendado":
        forecast, mae, rmse = prediccion_futura_prophet(
            df_pred_base,
            anios_futuros
        )

        ultimo_anio = int(df_pred_base["ANIO_DEFUNCION"].max())
        df_futuro = forecast[forecast["ANIO_DEFUNCION"] > ultimo_anio].copy()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df_pred_base["ANIO_DEFUNCION"],
                y=df_pred_base["Defunciones"],
                mode="markers+lines",
                name="Observado"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast["ANIO_DEFUNCION"],
                y=forecast["yhat"],
                mode="lines",
                name="Predicción Prophet",
                line=dict(color="black", dash="dash", width=4)
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast["ANIO_DEFUNCION"],
                y=forecast["yhat_upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast["ANIO_DEFUNCION"],
                y=forecast["yhat_lower"],
                mode="lines",
                name="Intervalo de incertidumbre",
                fill="tonexty",
                line=dict(width=0)
            )
        )

        df_tabla = df_futuro[[
            "ANIO_DEFUNCION",
            "yhat",
            "yhat_lower",
            "yhat_upper"
        ]].rename(columns={
            "yhat": "Predicción",
            "yhat_lower": "Límite inferior",
            "yhat_upper": "Límite superior"
        })

    else:
        grado_pred = st.selectbox(
            "Grado del polinomio para predicción",
            [1, 2, 3],
            index=1,
            key=f"{key_prefix}_grado_pred"
        )

        df_forecast, df_coef, mae, rmse = prediccion_futura_polinomial(
            df_pred_base,
            anios_futuros,
            grado=grado_pred
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df_forecast["ANIO_DEFUNCION"],
                y=df_forecast["Observado"],
                mode="markers+lines",
                name="Observado"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_forecast["ANIO_DEFUNCION"],
                y=df_forecast["Prediccion"],
                mode="lines",
                name=f"Predicción polinomial grado {grado_pred}",
                line=dict(color="black", dash="dash", width=4)
            )
        )

        df_tabla = df_forecast[
            df_forecast["Tipo"] == "Predicción futura"
        ][[
            "ANIO_DEFUNCION",
            "Prediccion"
        ]].rename(columns={
            "Prediccion": "Predicción"
        })

        with st.expander("Coeficientes del modelo polinomial"):
            st.dataframe(df_coef, use_container_width=True)

    fig.update_layout(
        title=f"Predicción futura: {etiqueta}",
        height=600,
        xaxis=dict(
            title="Año",
            tickmode="linear",
            dtick=1
        ),
        yaxis=dict(
            title=etiqueta
        ),
        legend=dict(
            orientation="h",
            y=-0.2,
            x=0.5,
            xanchor="center"
        ),
        margin=dict(l=40, r=40, t=80, b=160)
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("MAE histórico", f"{mae:.2f}")

    with col2:
        st.metric("RMSE histórico", f"{rmse:.2f}")

    st.caption(
        "MAE y RMSE indican el error histórico aproximado del modelo."
    )

    st.markdown("**Tabla de predicción futura:**")
    st.dataframe(df_tabla, use_container_width=True)


# ============================================================
# MAPAS
# ============================================================

def preparar_acumulado_estado(df, col_estado, anio_inicio):
    df_temp = df.copy()

    df_temp["ANIO_DEFUNCION"] = pd.to_numeric(
        df_temp["ANIO_DEFUNCION"],
        errors="coerce"
    )

    df_temp = df_temp[
        df_temp["ANIO_DEFUNCION"].notna() &
        df_temp[col_estado].notna()
    ].copy()

    df_temp = df_temp[df_temp["ANIO_DEFUNCION"] >= anio_inicio].copy()
    df_temp["Estado"] = df_temp[col_estado].apply(normalizar_estado)
    df_temp = df_temp[df_temp["Estado"].notna()].copy()

    df_acum = (
        df_temp
        .groupby("Estado")
        .size()
        .reset_index(name="Valor")
    )

    df_acum["Mapa"] = "Acumulado histórico"

    return df_acum


def predecir_estado_polinomial(df_estado, anios_futuros, grado):
    df_estado = df_estado.sort_values("ANIO_DEFUNCION").copy()

    if df_estado.shape[0] < grado + 2:
        return np.nan

    x = df_estado["ANIO_DEFUNCION"].values - 2000
    y = df_estado["Defunciones"].values

    coef = np.polyfit(x, y, grado)
    modelo = np.poly1d(coef)

    ultimo_anio = int(df_estado["ANIO_DEFUNCION"].max())

    anios_pred = np.arange(
        ultimo_anio + 1,
        ultimo_anio + anios_futuros + 1
    )

    x_pred = anios_pred - 2000
    y_pred = np.clip(modelo(x_pred), 0, None)

    return y_pred.sum()


def predecir_estado_prophet(df_estado, anios_futuros):
    df_estado = df_estado.sort_values("ANIO_DEFUNCION").copy()

    if df_estado.shape[0] < 6:
        return np.nan

    prophet_df = pd.DataFrame({
        "ds": pd.to_datetime(df_estado["ANIO_DEFUNCION"].astype(str) + "-01-01"),
        "y": df_estado["Defunciones"]
    })

    try:
        modelo = Prophet(
            changepoint_prior_scale=0.1,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False
        )

        modelo.fit(prophet_df)

        futuro = modelo.make_future_dataframe(
            periods=anios_futuros,
            freq="YS"
        )

        forecast = modelo.predict(futuro)

        ultimo_anio = int(df_estado["ANIO_DEFUNCION"].max())
        forecast["ANIO_DEFUNCION"] = forecast["ds"].dt.year

        futuro_pred = forecast[
            forecast["ANIO_DEFUNCION"] > ultimo_anio
        ].copy()

        y_pred = np.clip(futuro_pred["yhat"].values, 0, None)

        return y_pred.sum()

    except Exception:
        return np.nan


def preparar_prediccion_estado(df, col_estado, anio_inicio, anios_futuros, modelo_mapa, grado):
    df_temp = df.copy()

    df_temp["ANIO_DEFUNCION"] = pd.to_numeric(
        df_temp["ANIO_DEFUNCION"],
        errors="coerce"
    )

    df_temp = df_temp[
        df_temp["ANIO_DEFUNCION"].notna() &
        df_temp[col_estado].notna()
    ].copy()

    df_temp = df_temp[df_temp["ANIO_DEFUNCION"] >= anio_inicio].copy()
    df_temp["Estado"] = df_temp[col_estado].apply(normalizar_estado)
    df_temp = df_temp[df_temp["Estado"].notna()].copy()

    df_anual_estado = (
        df_temp
        .groupby(["Estado", "ANIO_DEFUNCION"])
        .size()
        .reset_index(name="Defunciones")
    )

    resultados = []

    for estado in sorted(df_anual_estado["Estado"].unique()):
        df_estado = df_anual_estado[df_anual_estado["Estado"] == estado].copy()

        if modelo_mapa == "Prophet":
            pred = predecir_estado_prophet(df_estado, anios_futuros)
        else:
            pred = predecir_estado_polinomial(df_estado, anios_futuros, grado)

        resultados.append({
            "Estado": estado,
            "Valor": pred,
            "Mapa": f"Predicción acumulada próximos {anios_futuros} años"
        })

    return pd.DataFrame(resultados)


def mostrar_mapas_acumulado_vs_prediccion(df_filtrado, resumen):
    st.divider()

    st.subheader("Mapas: acumulado histórico vs escenario futuro")

    if resumen["tipo_ubicacion"] == "Todo México":
        col_estado = "ENTIDAD_OCURRENCIAD"
    else:
        col_estado = resumen["col_estado"]

    modelo_mapa = st.selectbox(
        "Modelo para predicción por estado",
        [
            "Prophet",
            "Polinomial"
        ],
        key="mapa_modelo_unificado"
    )

    grado = 2

    if modelo_mapa == "Polinomial":
        grado = st.selectbox(
            "Grado del polinomio para el mapa predictivo",
            [1, 2, 3],
            index=1,
            key="mapa_grado_unificado"
        )

    df_temp = df_filtrado.copy()
    df_temp["ANIO_DEFUNCION"] = pd.to_numeric(
        df_temp["ANIO_DEFUNCION"],
        errors="coerce"
    )

    if df_temp["ANIO_DEFUNCION"].dropna().empty:
        st.warning("No hay años válidos para generar mapas.")
        return

    anio_min = int(df_temp["ANIO_DEFUNCION"].dropna().min())
    anio_max = int(df_temp["ANIO_DEFUNCION"].dropna().max())

    col1, col2 = st.columns(2)

    with col1:
        anio_inicio_mapa = st.number_input(
            "Usar datos desde el año para mapas",
            min_value=anio_min,
            max_value=anio_max,
            value=max(2003, anio_min),
            step=1,
            key="mapa_anio_inicio_unificado"
        )

    with col2:
        anios_futuros = st.number_input(
            "Años futuros a predecir para mapas",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            key="mapa_anios_futuros_unificado"
        )

    df_acumulado = preparar_acumulado_estado(
        df_filtrado,
        col_estado,
        anio_inicio_mapa
    )

    df_prediccion = preparar_prediccion_estado(
        df_filtrado,
        col_estado,
        anio_inicio_mapa,
        anios_futuros,
        modelo_mapa,
        grado
    )

    df_futuro_total = df_acumulado.merge(
        df_prediccion,
        on="Estado",
        how="outer",
        suffixes=("_historico", "_predicho")
    )

    df_futuro_total["Valor_historico"] = df_futuro_total["Valor_historico"].fillna(0)
    df_futuro_total["Valor_predicho"] = df_futuro_total["Valor_predicho"].fillna(0)

    df_futuro_total["Valor"] = (
        df_futuro_total["Valor_historico"] +
        df_futuro_total["Valor_predicho"]
    )

    df_futuro_total["Mapa"] = (
        f"Acumulado histórico + predicción próximos {anios_futuros} años"
    )

    df_futuro_total = df_futuro_total[["Estado", "Valor", "Mapa"]]

    df_mapa = pd.concat(
        [df_acumulado, df_futuro_total],
        ignore_index=True
    )

    df_mapa = completar_estados_mapa(df_mapa)

    geojson_mx = cargar_geojson_mexico()

    fig = px.choropleth(
        df_mapa,
        geojson=geojson_mx,
        locations="Estado",
        featureidkey="properties.name",
        color="Valor",
        facet_col="Mapa",
        color_continuous_scale="Reds",
        title="Acumulado histórico vs escenario futuro",
        labels={
            "Valor": "Casos"
        }
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False
    )

    fig.update_layout(
        height=650,
        margin=dict(l=20, r=20, t=80, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mapa de diferencia futura estimada")

    df_diferencia = df_futuro_total.copy()

    df_diferencia = df_diferencia.merge(
        df_acumulado[["Estado", "Valor"]],
        on="Estado",
        how="left",
        suffixes=("_futuro_total", "_historico")
    )

    df_diferencia["Valor_historico"] = df_diferencia["Valor_historico"].fillna(0)

    df_diferencia["Diferencia"] = (
        df_diferencia["Valor_futuro_total"] -
        df_diferencia["Valor_historico"]
    )

    df_diferencia = df_diferencia[[
        "Estado",
        "Diferencia"
    ]]

    df_diferencia = completar_estados_mapa(
        df_diferencia
        .rename(columns={"Diferencia": "Valor"})
        .assign(Mapa="Diferencia futura estimada")
    ).rename(columns={"Valor": "Diferencia"})

    max_abs = df_diferencia["Diferencia"].abs().max()

    if max_abs == 0 or pd.isna(max_abs):
        max_abs = 1

    fig_diff = px.choropleth(
        df_diferencia,
        geojson=geojson_mx,
        locations="Estado",
        featureidkey="properties.name",
        color="Diferencia",
        color_continuous_scale="RdBu_r",
        range_color=[-max_abs, max_abs],
        title="Diferencia futura estimada por estado",
        labels={
            "Diferencia": "Diferencia estimada"
        }
    )

    fig_diff.update_geos(
        fitbounds="locations",
        visible=False
    )

    fig_diff.update_layout(
        height=650,
        margin=dict(l=20, r=20, t=80, b=20)
    )

    st.plotly_chart(fig_diff, use_container_width=True)

    with st.expander("Ver tablas de mapas"):
        st.markdown("**Mapa acumulado/futuro**")
        st.dataframe(
            df_mapa.sort_values(["Mapa", "Valor"], ascending=[True, False]),
            use_container_width=True
        )

        st.markdown("**Diferencia futura estimada**")
        st.dataframe(
            df_diferencia.sort_values("Diferencia", ascending=False),
            use_container_width=True
        )


# ============================================================
# FUNCIÓN PRINCIPAL DEL MÓDULO
# ============================================================

def mostrar_modulo_prediccion_tendencias(df):
    st.header("Módulo 2: Análisis temporal y predicción")

    columnas_necesarias = [
        "ANIO_DEFUNCION",
        "ENTIDAD_OCURRENCIAD",
        "MUNICIPIO_OCURRENCIAD",
        "LOCALIDAD_OCURRENCIAD",
        "ENTIDAD_RESIDENCIAD",
        "MUNICIPIO_RESIDENCIAD",
        "LOCALIDAD_RESIDENCIAD"
    ]

    columnas_faltantes = [
        col for col in columnas_necesarias if col not in df.columns
    ]

    if columnas_faltantes:
        st.error("Faltan columnas necesarias para este módulo:")
        st.write(columnas_faltantes)
        return

    df_filtrado, resumen = aplicar_filtros_unificados(df)

    st.markdown(f"**Registros incluidos:** {df_filtrado.shape[0]}")

    if df_filtrado.empty:
        st.warning("No hay registros con los filtros seleccionados.")
        return

    st.divider()

    st.subheader("Variable o fenómeno a analizar")

    modo_analisis = st.selectbox(
        "Selecciona qué quieres analizar",
        [
            "Defunciones totales",
            "Causa de muerte específica",
            "Variable categórica",
            "Variable numérica"
        ],
        key="modo_analisis_unificado"
    )

    variable = None
    categorias_sel = None

    if modo_analisis == "Causa de muerte específica":
        variable = "CAUSA_CIE_4D"

        if variable not in df_filtrado.columns:
            st.warning("No se encontró la columna CAUSA_CIE_4D.")
            return

        causas = sorted(df_filtrado[variable].dropna().astype(str).unique())

        categorias_sel = st.multiselect(
            "Selecciona causa(s) de muerte a analizar",
            options=causas,
            default=[],
            help="Si no seleccionas ninguna, se incluyen todas.",
            key="causas_modo_analisis"
        )

        if not categorias_sel:
            modo_analisis_real = "Defunciones totales"
            variable = None
        else:
            modo_analisis_real = "Variable categórica"

    elif modo_analisis == "Variable categórica":
        variables = [
            col for col in crear_catalogo_variables_tendencia(df_filtrado)
            if not pd.api.types.is_numeric_dtype(df_filtrado[col])
        ]

        variable = st.selectbox(
            "Selecciona variable categórica",
            options=variables,
            key="variable_categorica_unificada"
        )

        categorias = sorted(df_filtrado[variable].dropna().astype(str).unique())

        categorias_sel = st.multiselect(
            "Selecciona categoría(s) a modelar",
            options=categorias,
            default=categorias[:3],
            help="Si seleccionas varias categorías, se modela su frecuencia combinada.",
            key="categorias_modelar_unificadas"
        )

        modo_analisis_real = "Variable categórica"

    elif modo_analisis == "Variable numérica":
        variables = [
            col for col in crear_catalogo_variables_tendencia(df_filtrado)
            if pd.api.types.is_numeric_dtype(df_filtrado[col])
        ]

        if not variables:
            st.warning("No hay variables numéricas disponibles.")
            return

        variable = st.selectbox(
            "Selecciona variable numérica",
            options=variables,
            key="variable_numerica_unificada"
        )

        modo_analisis_real = "Variable numérica"

    else:
        modo_analisis_real = "Defunciones totales"

    df_anual, etiqueta = preparar_serie_anual(
        df_filtrado,
        modo_analisis_real,
        variable=variable,
        categorias_sel=categorias_sel
    )

    if df_anual.empty:
        st.warning("No hay datos suficientes para construir la serie anual.")
        return

    graficar_serie_simple(df_anual, etiqueta)

    if variable is not None and modo_analisis in ["Variable categórica", "Causa de muerte específica"]:
        graficar_categorias_temporales(df_filtrado, variable)

    st.divider()

    graficar_series_multiples(
        df_filtrado,
        resumen,
        etiqueta
    )

    mostrar_modelos_tendencia(
        df_anual,
        etiqueta,
        key_prefix="unificado"
    )

    mostrar_prediccion_futura(
        df_anual,
        etiqueta,
        key_prefix="unificado"
    )

    mostrar_mapas_acumulado_vs_prediccion(
        df_filtrado,
        resumen
    )