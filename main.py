
import joblib
import streamlit as st
import pandas as pd

# --- Configuración de la Página ---
# Esto debe ser lo primero que se ejecute en el script.
st.set_page_config(
    page_title="Predictor de Calidad de Concentración de silice",
    page_icon="🧪",
    layout="wide"
)

# --- Carga del Modelo ---
# Usamos @st.cache_resource para que el modelo se cargue solo una vez y se mantenga en memoria,
# lo que hace que la aplicación sea mucho más rápida.
@st.cache_resource
def load_model(model_path):
    """Carga el modelo entrenado desde un archivo .joblib."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo del modelo en {model_path}. Asegúrate de que el archivo del modelo esté en el directorio correcto.")
        return None

# Cargamos nuestro modelo campeón. Streamlit buscará en la ruta 'modelo_xgboost_final.joblib'.
model = load_model('modelo.joblib')

# --- Barra Lateral para las Entradas del Usuario ---
with st.sidebar:
    st.header("⚙️ Parámetros de Entrada")
    st.markdown("""
    Ajusta los deslizadores para que coincidan con los parámetros operativos del proceso de flotación.
    """)

    # Slider para el caudal de Amina
    flowrate1 = st.slider(
        label='Caudal de Amina (m³/s)',
        min_value=241.7,
        max_value=739.3,
        value=488.43, # Valor inicial
        step=1
    )
    st.caption("El flujo de amina tiene una relación directamente proporcional con la flotación de sílice: a mayor dosificación, más partículas de sílice se colectan y flotan, aumentando su recuperación en el concentrado.")

    # Slider para el caudal de aire en la columna de flotación 01
    Flowrate2 = st.slider(
        label='Caudal de aire en la columna de flotación 01 (m³/s)',
        min_value=175.85,
        max_value=372.44,
        value=200.13,
        step=1
    )
    st.caption("La relación entre el flujo de aire y la concentración final de sílice es inversa, porque al aumentar el aire se favorece la flotación y, por tanto, disminuye la sílice en el concentrado.")

    # Slider para el porcentaje de concentración de hierro
    concentration = st.slider(
        label='Concentración de hierro (%)',
        min_value=62.51,
        max_value=68.01,
        value=65.04,
        step=1
    )
    st.caption("El % de hierro es inversamente proporcional al % de sílice; cuando sube la concentración de hierro, baja el silice")

# --- Contenido de la Página Principal ---
st.title("🧪 Predictor de Concentración de Silice")
st.markdown("""
¡Bienvenido! Esta aplicación utiliza un modelo de machine learning para predecir la concentración final de silice en columnas de flotación basándose en parámetros operativos clave.

**Esta herramienta puede ayudar a los ingenieros de procesos y operadores a:**
- **Optimizar** las condiciones de operación para disminuir la concentración final de silice.
- **Predecir** el impacto de los cambios en el proceso antes de implementarlos.
- **Solucionar** problemas potenciales simulando diferentes escenarios.
""")

# --- Lógica de Predicción ---
# Solo intentamos predecir si el modelo se ha cargado correctamente.
if model is not None:
    # El botón principal que el usuario presionará para obtener un resultado.
    if st.button('🚀 Predecir Concentración', type="primary"):
        # Creamos un DataFrame de pandas con las entradas del usuario.
        # ¡Es crucial que los nombres de las columnas coincidan exactamente con los que el modelo espera!
        df_input = pd.DataFrame({
            'Amina Flow': [Flowrate1],
            'Flotation Column 01 Air Flow': [Flowrate2],
            '% Iron Concentrate': [Concentration]
        })

        # Hacemos la predicción
        try:
            concentration_value = model.predict(df_input)
            st.subheader("📈 Resultado de la Predicción")
            # Mostramos el resultado en un cuadro de éxito, formateado a dos decimales.
            st.success(f"**Concentración Predicha:** `{concentration_value[0]:.2f}%`")
            st.info("Este valor representa el porcentaje estimado de la concentración final de silice.")
        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {e}")
else:
    st.warning("El modelo no pudo ser cargado. Por favor, verifica la ruta del archivo del modelo.")

st.divider()

# --- Sección de Explicación ---
with st.expander("ℹ️ Sobre la Aplicación"):
    st.markdown("""
    **¿Cómo funciona?**

    1.  **Datos de Entrada:** Proporcionas los parámetros operativos clave usando los deslizadores en la barra lateral.
    2.  **Predicción:** El modelo de machine learning pre-entrenado recibe estas entradas y las analiza basándose en los patrones que aprendió de datos históricos.
    3.  **Resultado:** La aplicación muestra la concentración final de silice predicho como un porcentaje.

    **Detalles del Modelo:**

    * **Tipo de Modelo:** `Regression Model` (XGBoost Optimizado)
    * **Propósito:** Predecir la concentración final de silice.
    * **Características Usadas:** Caudal de Amina, Caudal de aire en la columan de flotación 01 y % de concentración de Hierro .
    """)
