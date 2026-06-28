"""
Sistema Inteligente de Analitica Predictiva Logistica - Modelo SAT-ML
Trabajo de Grado 3 - Ingenieria de Sistemas
Autor: Wilson Andres Carbajal Barreto

Este demo entrena un modelo de Machine Learning REAL (Random Forest Regressor)
sobre un dataset simulado de operaciones logisticas en los CEDI de Soacha y Tenjo.
Las metricas de error (MAE, RMSE, R2) se calculan en vivo sobre un conjunto de
prueba separado del de entrenamiento (no son valores fijos ni inventados).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

# -----------------------------------------------------------------------
# CONFIGURACION DE LA PAGINA
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="SAT-ML | Analitica Predictiva Logistica",
    page_icon="📦",
    layout="wide"
)

SEMILLA = 42  # Semilla fija -> resultados reproducibles para el jurado

# -----------------------------------------------------------------------
# 1. SIMULACION DE LA BASE DE DATOS HISTORICA (Materia prima del modelo)
# -----------------------------------------------------------------------
@st.cache_data
def generar_datos_historicos(n_dias=180, semilla=SEMILLA):
    """
    Genera un historico sintetico de 180 dias por CEDI con estacionalidad,
    tendencia y ruido aleatorio, simulando demanda real de una cadena de frio.
    Se documenta el supuesto: en produccion estos datos vendrian del ERP/SQL
    transaccional, tal como se describe en el informe tecnico.
    """
    rng = np.random.default_rng(semilla)
    registros = []
    cedis = ["Soacha", "Tenjo"]
    fecha_inicio = datetime.today() - timedelta(days=n_dias)

    for cedi in cedis:
        base_demanda = 280 if cedi == "Soacha" else 190
        for i in range(n_dias):
            fecha = fecha_inicio + timedelta(days=i)
            dia_semana = fecha.weekday()  # 0=Lunes
            estacionalidad = 1.18 if dia_semana in (4, 5) else 1.0  # pico fin de semana
            tendencia = 1 + (i / n_dias) * 0.12  # leve crecimiento en el periodo
            ruido = rng.normal(0, 18)

            demanda_real = max(
                30,
                base_demanda * estacionalidad * tendencia + ruido
            )
            temperatura_promedio = rng.normal(6.5, 1.2)  # cadena de frio (C)
            tiempo_trafico_min = rng.normal(35 if cedi == "Soacha" else 22, 8)
            satisfaccion_sat = np.clip(rng.normal(4.2, 0.45), 1.0, 5.0)
            stock_inicial = demanda_real * rng.uniform(0.85, 1.25)

            registros.append({
                "Fecha": fecha,
                "CEDI": cedi,
                "Dia_Semana": dia_semana,
                "Temperatura_Promedio_C": round(temperatura_promedio, 1),
                "Tiempo_Trafico_Min": round(max(5, tiempo_trafico_min), 1),
                "Stock_Disponible": round(stock_inicial, 0),
                "Indice_Satisfaccion_SAT": round(satisfaccion_sat, 2),
                "Demanda_Real": round(demanda_real, 0),
            })

    df = pd.DataFrame(registros)
    return df


@st.cache_resource
def entrenar_modelo(df):
    """
    Entrena un Random Forest Regressor REAL para predecir la demanda futura
    a partir de variables operativas. Separa train/test (80/20) y calcula
    las metricas de error sobre el conjunto de prueba (datos que el modelo
    nunca vio durante el entrenamiento).
    """
    df_model = df.copy()
    df_model["CEDI_cod"] = df_model["CEDI"].map({"Soacha": 0, "Tenjo": 1})

    features = [
        "CEDI_cod", "Dia_Semana", "Temperatura_Promedio_C",
        "Tiempo_Trafico_Min", "Stock_Disponible", "Indice_Satisfaccion_SAT"
    ]
    X = df_model[features]
    y = df_model["Demanda_Real"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEMILLA
    )

    modelo = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=SEMILLA
    )
    inicio = time.perf_counter()
    modelo.fit(X_train, y_train)
    tiempo_entrenamiento = time.perf_counter() - inicio

    inicio_pred = time.perf_counter()
    y_pred = modelo.predict(X_test)
    tiempo_prediccion = time.perf_counter() - inicio_pred

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100
    precision_aprox = max(0, 100 - mape)  # Precision aproximada = 100% - error % promedio

    importancias = pd.Series(modelo.feature_importances_, index=features).sort_values(ascending=False)

    return {
        "modelo": modelo,
        "features": features,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "precision_aprox": precision_aprox,
        "tiempo_entrenamiento_seg": tiempo_entrenamiento,
        "tiempo_prediccion_seg": tiempo_prediccion,
        "importancias": importancias,
    }


def proyectar_72h(df, modelo_info, cedi_sel):
    """Genera la proyeccion de demanda para las proximas 72 horas (3 dias) usando el modelo entrenado."""
    rng = np.random.default_rng(SEMILLA + 1)
    modelo = modelo_info["modelo"]
    filas = []
    cedis = ["Soacha", "Tenjo"] if cedi_sel == "Ambos" else [cedi_sel]

    for cedi in cedis:
        ult = df[df["CEDI"] == cedi].iloc[-1]
        for h in range(1, 4):
            fecha_f = datetime.today() + timedelta(days=h)
            fila = {
                "CEDI_cod": 0 if cedi == "Soacha" else 1,
                "Dia_Semana": fecha_f.weekday(),
                "Temperatura_Promedio_C": ult["Temperatura_Promedio_C"] + rng.normal(0, 0.4),
                "Tiempo_Trafico_Min": ult["Tiempo_Trafico_Min"] + rng.normal(0, 3),
                "Stock_Disponible": ult["Stock_Disponible"],
                "Indice_Satisfaccion_SAT": ult["Indice_Satisfaccion_SAT"],
            }
            X_f = pd.DataFrame([fila])[modelo_info["features"]]
            demanda_pred = modelo.predict(X_f)[0]
            riesgo = "Alto" if demanda_pred > ult["Stock_Disponible"] else (
                "Medio" if demanda_pred > 0.85 * ult["Stock_Disponible"] else "Bajo"
            )
            filas.append({
                "CEDI": cedi,
                "Fecha_Proyectada": fecha_f.strftime("%Y-%m-%d"),
                "Horizonte": f"+{h*24}h",
                "Stock_Disponible": round(ult["Stock_Disponible"], 0),
                "Demanda_Predicha_ML": round(demanda_pred, 0),
                "Riesgo_Quiebre": riesgo
            })
    return pd.DataFrame(filas)


# -----------------------------------------------------------------------
# CARGA Y ENTRENAMIENTO (se ejecuta una sola vez gracias al cache)
# -----------------------------------------------------------------------
df_historico = generar_datos_historicos()
modelo_info = entrenar_modelo(df_historico)

# -----------------------------------------------------------------------
# 2. MODULO DE AUTENTICACION (control de acceso por rol)
# -----------------------------------------------------------------------
st.sidebar.title("🔐 Acceso CEDI")
st.sidebar.caption("Modulo de autenticacion simulado para el DEMO academico")

if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

usuario = st.sidebar.text_input("Usuario", "w.carbajal")
clave = st.sidebar.text_input("Contraseña", type="password", placeholder="demo123")
rol = st.sidebar.selectbox("Rol", ["Coordinador Logistico", "Administrador CEDI", "Analista BI"])

if st.sidebar.button("Ingresar al sistema", use_container_width=True):
    st.session_state.autenticado = True
    st.session_state.usuario = usuario
    st.session_state.rol = rol

if st.session_state.autenticado:
    st.sidebar.success(f"Bienvenido, {st.session_state.usuario} ({st.session_state.rol})")
else:
    st.sidebar.info("Ingrese cualquier usuario y contraseña para simular el acceso por rol.")

st.sidebar.markdown("---")
cedi_filter = st.sidebar.selectbox("📍 Seleccionar CEDI", ["Ambos", "Soacha", "Tenjo"])

df_vista = df_historico if cedi_filter == "Ambos" else df_historico[df_historico["CEDI"] == cedi_filter]

st.sidebar.markdown("---")
st.sidebar.caption(
    "Proyecto: Sistema Inteligente de Analitica Predictiva Logistica (SAT-ML)\n\n"
    "Trabajo de Grado 3 - Ingenieria de Sistemas"
)

# -----------------------------------------------------------------------
# 3. ENCABEZADO E INDICADORES DEL MODELO (metricas REALES, calculadas en vivo)
# -----------------------------------------------------------------------
st.title("📦 Sistema Predictivo SAT-ML: Optimizacion de Inventarios")
st.markdown(
    "Plataforma de analitica predictiva que integra variables logisticas con el "
    "indice de satisfaccion del cliente (SAT) para anticipar quiebres de stock en "
    "los CEDI de **Soacha** y **Tenjo**."
)

st.markdown("#### 🧪 Validacion del modelo (Random Forest) sobre datos de prueba")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Precision aproximada", f"{modelo_info['precision_aprox']:.1f}%",
          help="100% - Error porcentual absoluto medio (MAPE) sobre el set de prueba (20% de los datos, no usado en entrenamiento).")
c2.metric("Error absoluto medio (MAE)", f"{modelo_info['mae']:.1f} unid.")
c3.metric("R² (bondad de ajuste)", f"{modelo_info['r2']:.3f}")
c4.metric("Tiempo de inferencia", f"{modelo_info['tiempo_prediccion_seg']*1000:.1f} ms")

with st.expander("Ver detalle metodologico de la validacion (para sustentacion)"):
    st.write(
        f"""
        - **Dataset:** {len(df_historico)} registros simulados (180 dias x 2 CEDI), generados con semilla fija ({SEMILLA}) para garantizar reproducibilidad.
        - **Particion:** 80% entrenamiento / 20% prueba (`train_test_split`, `random_state={SEMILLA}`).
        - **Algoritmo:** `RandomForestRegressor` (200 arboles, profundidad maxima 8), libreria Scikit-Learn.
        - **Variables predictoras:** CEDI, dia de la semana, temperatura promedio, tiempo de trafico, stock disponible, indice de satisfaccion SAT.
        - **Variable objetivo:** Demanda real (unidades despachadas).
        - **Metricas obtenidas sobre el set de prueba (no vistas en entrenamiento):**
            - MAE = {modelo_info['mae']:.2f} unidades
            - RMSE = {modelo_info['rmse']:.2f} unidades
            - MAPE = {modelo_info['mape']:.2f}%
            - R² = {modelo_info['r2']:.3f}
        - **Tiempo de entrenamiento:** {modelo_info['tiempo_entrenamiento_seg']*1000:.1f} ms en este servidor.
        """
    )
    st.caption(
        "Nota academica: los datos de entrada son simulados (no provienen de un ERP real), "
        "pero el modelo, el entrenamiento y las metricas son reales y reproducibles: "
        "cualquiera puede ejecutar este mismo codigo y obtener los mismos resultados."
    )

st.markdown("---")

# -----------------------------------------------------------------------
# 4. GRAFICAS: PREDICCION VS REAL Y CORRELACION SAT-RIESGO
# -----------------------------------------------------------------------
colA, colB = st.columns(2)

with colA:
    st.subheader("Predicho vs. Real (set de prueba)")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=modelo_info["y_test"], y=modelo_info["y_pred"],
        mode="markers", name="Predicciones",
        marker=dict(color="#2E75B6", size=9, opacity=0.75)
    ))
    min_v = min(modelo_info["y_test"].min(), modelo_info["y_pred"].min())
    max_v = max(modelo_info["y_test"].max(), modelo_info["y_pred"].max())
    fig1.add_trace(go.Scatter(
        x=[min_v, max_v], y=[min_v, max_v],
        mode="lines", name="Prediccion perfecta",
        line=dict(color="red", dash="dash")
    ))
    fig1.update_layout(
        xaxis_title="Demanda real",
        yaxis_title="Demanda predicha por el modelo",
        height=380, margin=dict(t=20)
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Cada punto es un dia del set de prueba. Mientras mas cerca de la linea roja, mejor la prediccion.")

with colB:
    st.subheader("Importancia de variables en el modelo")
    imp_df = modelo_info["importancias"].reset_index()
    imp_df.columns = ["Variable", "Importancia"]
    fig2 = px.bar(
        imp_df, x="Importancia", y="Variable", orientation="h",
        color="Importancia", color_continuous_scale="Blues"
    )
    fig2.update_layout(height=380, margin=dict(t=20), showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Indica que tan determinante es cada variable para que el Random Forest prediga la demanda.")

st.markdown("---")

# -----------------------------------------------------------------------
# 5. HISTORICO POR CEDI
# -----------------------------------------------------------------------
st.subheader("📊 Historico de demanda y stock por CEDI")
fig3 = px.line(
    df_vista.sort_values("Fecha").tail(60 if cedi_filter != "Ambos" else 120),
    x="Fecha", y=["Stock_Disponible", "Demanda_Real"],
    color="CEDI" if cedi_filter == "Ambos" else None,
    markers=False,
    title="Ultimos dias registrados: Stock disponible vs. Demanda real"
)
fig3.update_layout(height=380, margin=dict(t=40))
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------
# 6. PROYECCION A 72 HORAS Y MATRIZ DE ALERTAS (Output principal del sistema)
# -----------------------------------------------------------------------
st.subheader("🚦 Proyeccion de demanda a 72 horas y alerta semaforica de quiebre de stock")
df_proy = proyectar_72h(df_historico, modelo_info, cedi_filter)


def color_riesgo(val):
    colores = {"Alto": "background-color: #ffcccc", "Medio": "background-color: #fff3cd", "Bajo": "background-color: #ccffcc"}
    return colores.get(val, "")

st.dataframe(
    df_proy.style.applymap(color_riesgo, subset=["Riesgo_Quiebre"]),
    use_container_width=True,
    hide_index=True
)

riesgo_alto = len(df_proy[df_proy["Riesgo_Quiebre"] == "Alto"])
if riesgo_alto > 0:
    st.warning(f"⚠️ El sistema detecto {riesgo_alto} ventana(s) con riesgo ALTO de quiebre de stock en las proximas 72 horas.")
else:
    st.success("✅ No se detectan rutas con riesgo alto de quiebre de stock en las proximas 72 horas.")

st.markdown("---")
st.caption(
    "Demo academico - Trabajo de Grado 3: Ingenieria de Sistemas | "
    "Autor: Wilson Andres Carbajal Barreto | Modelo SAT-ML (Random Forest real, Scikit-Learn)"
)
