"""
Script de validacion standalone: reproduce EXACTAMENTE la misma logica de
entrenamiento que app.py, pero sin Streamlit, para poder ejecutarlo en
cualquier entorno con solo scikit-learn/pandas/numpy y confirmar que las
metricas que se citan en el informe tecnico y en la matriz de evaluacion
son reales y reproducibles.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

SEMILLA = 42

def generar_datos_historicos(n_dias=180, semilla=SEMILLA):
    rng = np.random.default_rng(semilla)
    registros = []
    cedis = ["Soacha", "Tenjo"]
    fecha_inicio = datetime.today() - timedelta(days=n_dias)

    for cedi in cedis:
        base_demanda = 280 if cedi == "Soacha" else 190
        for i in range(n_dias):
            fecha = fecha_inicio + timedelta(days=i)
            dia_semana = fecha.weekday()
            estacionalidad = 1.18 if dia_semana in (4, 5) else 1.0
            tendencia = 1 + (i / n_dias) * 0.12
            ruido = rng.normal(0, 18)
            demanda_real = max(30, base_demanda * estacionalidad * tendencia + ruido)
            temperatura_promedio = rng.normal(6.5, 1.2)
            tiempo_trafico_min = rng.normal(35 if cedi == "Soacha" else 22, 8)
            satisfaccion_sat = np.clip(rng.normal(4.2, 0.45), 1.0, 5.0)
            stock_inicial = demanda_real * rng.uniform(0.85, 1.25)

            registros.append({
                "Fecha": fecha, "CEDI": cedi, "Dia_Semana": dia_semana,
                "Temperatura_Promedio_C": round(temperatura_promedio, 1),
                "Tiempo_Trafico_Min": round(max(5, tiempo_trafico_min), 1),
                "Stock_Disponible": round(stock_inicial, 0),
                "Indice_Satisfaccion_SAT": round(satisfaccion_sat, 2),
                "Demanda_Real": round(demanda_real, 0),
            })
    return pd.DataFrame(registros)


def entrenar_modelo(df):
    df_model = df.copy()
    df_model["CEDI_cod"] = df_model["CEDI"].map({"Soacha": 0, "Tenjo": 1})
    features = ["CEDI_cod", "Dia_Semana", "Temperatura_Promedio_C",
                "Tiempo_Trafico_Min", "Stock_Disponible", "Indice_Satisfaccion_SAT"]
    X = df_model[features]
    y = df_model["Demanda_Real"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEMILLA)
    modelo = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=SEMILLA)

    t0 = time.perf_counter()
    modelo.fit(X_train, y_train)
    t_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = modelo.predict(X_test)
    t_pred = time.perf_counter() - t0

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100
    precision_aprox = max(0, 100 - mape)

    print(f"N registros totales: {len(df)}")
    print(f"N entrenamiento: {len(X_train)} | N prueba: {len(X_test)}")
    print(f"MAE  = {mae:.2f} unidades")
    print(f"RMSE = {rmse:.2f} unidades")
    print(f"MAPE = {mape:.2f} %")
    print(f"Precision aproximada (100-MAPE) = {precision_aprox:.2f} %")
    print(f"R2   = {r2:.4f}")
    print(f"Tiempo entrenamiento = {t_train*1000:.2f} ms")
    print(f"Tiempo inferencia (20% test set) = {t_pred*1000:.3f} ms")
    print(f"Tiempo inferencia promedio por registro = {(t_pred/len(X_test))*1000:.4f} ms")

    importancias = pd.Series(modelo.feature_importances_, index=features).sort_values(ascending=False)
    print("\nImportancia de variables:")
    print(importancias)

    return mae, rmse, mape, r2, precision_aprox


if __name__ == "__main__":
    df = generar_datos_historicos()
    entrenar_modelo(df)
