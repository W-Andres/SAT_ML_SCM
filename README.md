# Sistema Predictivo SAT-ML — Optimización de Inventarios CEDI

Demo funcional desarrollado para el **ACA 3 — Trabajo de Grado 3 (Ingeniería de Sistemas)**.

Autor: **Wilson Andrés Carbajal Barreto**

## ¿Qué hace este demo?

Es un dashboard web (Streamlit) que:

1. Simula un histórico de 180 días de operación logística para los CEDI de **Soacha** y **Tenjo** (demanda, stock, tráfico, temperatura de cadena de frío, satisfacción del cliente).
2. Entrena un modelo de **Machine Learning real** (`RandomForestRegressor` de Scikit-Learn) para predecir la demanda.
3. Calcula métricas de error **reales** (MAE, RMSE, MAPE, R²) sobre un conjunto de prueba separado del de entrenamiento — no son números fijos ni inventados; si cambias los datos, las métricas cambian.
4. Genera una proyección de demanda a 72 horas y una alerta semafórica (Alto/Medio/Bajo) de riesgo de quiebre de stock.
5. Incluye un módulo de autenticación simulado con selección de rol (Coordinador Logístico, Administrador CEDI, Analista BI).

## Cómo ejecutarlo en tu computador (local)

1. Instala Python 3.10 o superior.
2. Abre una terminal en esta carpeta y ejecuta:

```bash
pip install -r requirements.txt
streamlit run app.py
```

3. Se abrirá automáticamente en tu navegador en `http://localhost:8501`.

## Cómo subirlo a GitHub

1. Crea una cuenta gratuita en [github.com](https://github.com) si no tienes una.
2. Crea un repositorio nuevo, por ejemplo `satml-logistica-demo` (puede ser público).
3. Sube estos 3 archivos a la raíz del repositorio: `app.py`, `requirements.txt`, `README.md`.
   - Más fácil: en GitHub, botón **"Add file" → "Upload files"**, arrastra los archivos y da clic en **"Commit changes"**.

## Cómo desplegarlo gratis en Streamlit Community Cloud (para que quede accesible públicamente)

1. Ve a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con tu cuenta de GitHub.
2. Clic en **"New app"**.
3. Selecciona tu repositorio (`satml-logistica-demo`), la rama (`main`) y el archivo principal (`app.py`).
4. Clic en **"Deploy"**. En 1-2 minutos tendrás una URL pública tipo `https://tu-app.streamlit.app` — esa es la que pones en el informe técnico y enseñas en el video de sustentación.

## Estructura del proyecto

```
satml_demo/
├── app.py              # Aplicación principal (dashboard + modelo ML)
├── validar_modelo.py   # Script independiente que reproduce el entrenamiento sin Streamlit (para verificar las métricas)
├── requirements.txt    # Librerías necesarias
└── README.md           # Este archivo
```

## Nota académica sobre los datos

Los datos de entrada son **simulados** (no provienen de un ERP real, como se aclara también en el informe técnico), pero el proceso de Machine Learning —entrenamiento, partición train/test, cálculo de métricas— **es real y reproducible**. Cualquier persona que ejecute este mismo código obtendrá resultados estadísticamente equivalentes. Esto es importante mencionarlo en la sustentación para que el jurado entienda qué parte es simulación de contexto (los datos) y qué parte es el aporte técnico real (el modelo y su validación).
