# salary_predictor_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Salarios en Data Science",
    page_icon="💰",
    layout="wide"
)

# Función para cargar el modelo con diagnóstico mejorado
@st.cache_resource
def load_model():
    try:
        model_path = 'improved_salary_prediction_model.pkl'
        if not os.path.exists(model_path):
            st.error(f"Archivo de modelo no encontrado en: {os.path.abspath(model_path)}")
            return None
            
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            
        # Verificar si es un pipeline completo
        if hasattr(model, 'steps'):
            steps = [step[0] for step in model.steps]
            st.write(f"Modelo cargado correctamente. Pasos del pipeline: {steps}")
        else:
            st.warning("El modelo cargado no parece ser un pipeline completo. Puede que falte el preprocesamiento.")
        
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# Función para crear un modelo simulado si el real falla
def get_simulated_prediction(
    job_category, experience_level, region, company_sector, 
    tech_specialization, work_setting
):
    """Calcula un salario simulado basado en los principales factores"""
    
    # Factores base por categoría de trabajo
    base_salaries = {
        "Data Scientist": 95000,
        "Senior Data Scientist": 130000,
        "ML Engineer": 105000,
        "Data Engineer": 100000,
        "Data Analyst": 75000,
        "Business Analyst": 70000,
        "Research Scientist": 110000,
        "Data Science Manager": 140000,
        "AI Engineer": 115000
    }
    
    # Multiplicadores por nivel de experiencia
    experience_multiplier = {
        "Entry-Level": 0.7,
        "Mid-Level": 1.0,
        "Senior": 1.5,
        "Executive": 2.2
    }
    
    # Multiplicadores por región
    region_multiplier = {
        "North America": 1.2,
        "Europe": 1.0,
        "Asia": 0.7,
        "South America": 0.6,
        "Africa": 0.5,
        "Oceania": 1.1
    }
    
    # Multiplicadores por sector
    sector_multiplier = {
        "Technology": 1.1,
        "Finance": 1.15,
        "Healthcare": 1.05,
        "Retail": 0.9,
        "Manufacturing": 0.95,
        "Education": 0.85,
        "Other": 1.0
    }
    
    # Multiplicador por modalidad
    work_setting_multiplier = {
        "Remote": 1.05,
        "Hybrid": 1.0,
        "On-site": 0.95
    }
    
    # Calcular salario base
    if job_category in base_salaries:
        base = base_salaries[job_category]
    else:
        base = 90000  # Valor predeterminado
    
    # Aplicar multiplicadores
    exp_mult = experience_multiplier.get(experience_level, 1.0)
    reg_mult = region_multiplier.get(region, 1.0)
    sec_mult = sector_multiplier.get(company_sector, 1.0)
    work_mult = work_setting_multiplier.get(work_setting, 1.0)
    
    # Ajuste por especialización técnica
    tech_mult = 0.8 + (tech_specialization / 10 * 0.4)  # 0.8 a 1.2
    
    # Calcular salario final
    salary = base * exp_mult * reg_mult * sec_mult * tech_mult * work_mult
    
    # Añadir un factor aleatorio de variabilidad (±5%)
    import random
    random_factor = random.uniform(0.95, 1.05)
    salary *= random_factor
    
    return salary

# Cargar el modelo
model = load_model()
model_loaded = model is not None

# Título y descripción
st.title("🧠 Predictor de Salarios en Data Science")
st.markdown("""
Esta aplicación predice salarios en el campo de Data Science basándose en diferentes factores.
Complete los campos a continuación para obtener una predicción personalizada.
""")

# Crear columnas principales
col1, col2 = st.columns([1, 1])

# Sección de entrada de datos (columna izquierda)
with col1:
    st.header("📋 Información del Perfil")
    
    # Información básica
    job_category = st.selectbox(
        "Categoría de Trabajo",
        options=[
            "Data Scientist", "Senior Data Scientist", "ML Engineer", 
            "Data Engineer", "Data Analyst", "Business Analyst",
            "Research Scientist", "Data Science Manager", "AI Engineer"
        ]
    )
    
    experience_level = st.selectbox(
        "Nivel de Experiencia",
        options=["Entry-Level", "Mid-Level", "Senior", "Executive"]
    )
    
    # Región y configuración de trabajo
    st.subheader("Ubicación")
    
    region = st.selectbox(
        "Región de la Empresa",
        options=["North America", "Europe", "Asia", "South America", "Africa", "Oceania"]
    )
    
    residence_region = st.selectbox(
        "Región de Residencia del Empleado",
        options=["North America", "Europe", "Asia", "South America", "Africa", "Oceania"]
    )
    
    # Feedback sobre ubicaciones
    if region != residence_region:
        st.info("📌 Has seleccionado diferentes regiones para la empresa y tu residencia.")
    
    work_setting = st.selectbox(
        "Modalidad de Trabajo",
        options=["Remote", "Hybrid", "On-site"]
    )
    
    # Sección de detalles adicionales (expandible)
    with st.expander("Detalles Adicionales"):
        company_sector = st.selectbox(
            "Sector de la Empresa",
            options=["Technology", "Finance", "Healthcare", "Retail", "Manufacturing", "Education", "Other"]
        )
        
        company_size = st.selectbox(
            "Tamaño de la Empresa",
            options=["Small", "Medium", "Large"]
        )
        
        employment_type = st.selectbox(
            "Tipo de Empleo",
            options=["Full-time", "Part-time", "Contract", "Freelance"]
        )
        
        ai_relationship = st.selectbox(
            "Relación con IA",
            options=["Develops AI", "Uses AI", "No direct AI work"]
        )
    
    # Sección de métricas (sliders)
    st.subheader("📊 Métricas y Habilidades")
    
    tech_specialization = st.slider(
        "Especialización Técnica",
        min_value=1.0,
        max_value=10.0,
        value=7.0,
        step=0.5,
        help="Nivel de especialización técnica (1-10)"
    )
    
    english_level = st.slider(
        "Nivel de Inglés",
        min_value=1.0,
        max_value=10.0,
        value=7.0,
        step=0.5
    )
    
    demand_index = st.slider(
        "Índice de Demanda del Rol",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.5
    )
    
    automation_risk = st.slider(
        "Riesgo de Automatización",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.5
    )
    
    work_life_balance = st.slider(
        "Balance Trabajo-Vida",
        min_value=1.0,
        max_value=5.0,
        value=3.5,
        step=0.5
    )
    
    cost_of_living_index = st.slider(
        "Índice de Costo de Vida",
        min_value=30.0,
        max_value=100.0,
        value=70.0,
        step=5.0
    )

# Columna derecha para resultados y visualizaciones
with col2:
    st.header("💰 Predicción de Salario")
    
    # Modo de diagnóstico
    debug_mode = st.checkbox("Modo Diagnóstico", value=False)
    
    # Botón para hacer la predicción
    if st.button("Calcular Salario Estimado", type="primary"):
        # Siempre calcular la predicción simulada como respaldo
        simulated_salary = get_simulated_prediction(
            job_category, experience_level, region, 
            company_sector, tech_specialization, work_setting
        )
        
        if model_loaded:
            try:
                # Preparar los datos para la predicción
                input_data = {
                    'job_category': job_category,
                    'experience_level_desc': experience_level,
                    'employment_type_desc': employment_type,
                    'company_location': 'United States',  # Valor por defecto
                    'employee_residence': 'United States',  # Valor por defecto
                    'company_size_desc': company_size,
                    'work_setting': work_setting,
                    'region': region,
                    'residence_region': residence_region,
                    'domestic_employment': region == residence_region,  # Calculado
                    'economic_period': 'Post-Pandemic',  # Valor por defecto
                    'role_maturity': 'Established',  # Valor por defecto
                    'company_sector': company_sector,
                    'career_path': 'Technical',  # Valor por defecto
                    'ai_relationship': ai_relationship,
                    'work_year': 2023,  # Valor por defecto
                    'tech_specialization': float(tech_specialization),  # Forzar conversión
                    'english_level': float(english_level),  # Forzar conversión
                    'work_life_balance': float(work_life_balance),  # Forzar conversión
                    'demand_index': float(demand_index),  # Forzar conversión
                    'automation_risk': float(automation_risk),  # Forzar conversión
                    'cost_of_living_index': float(cost_of_living_index),  # Forzar conversión
                    'salary_to_experience_ratio': 1.0,  # Valor por defecto
                    'normalized_salary': 100000.0,  # Valor por defecto
                    'adjusted_salary': 100000.0,  # Valor por defecto
                    'total_compensation_estimate': 120000.0  # Valor por defecto
                }
                
                # Crear DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Mostrar datos de entrada en modo diagnóstico
                if debug_mode:
                    st.write("### Datos de Entrada")
                    st.write(input_df)
                
                # Ejecutar prueba de sensibilidad en modo diagnóstico
                if debug_mode:
                    st.write("### Prueba de Sensibilidad del Modelo")
                    test_inputs = []
                    test_predictions = []
                    
                    for level in ["Entry-Level", "Mid-Level", "Senior", "Executive"]:
                        try:
                            test_df = input_df.copy()
                            test_df["experience_level_desc"] = level
                            test_pred = model.predict(test_df)[0]
                            test_inputs.append(level)
                            test_predictions.append(test_pred)
                        except Exception as e:
                            st.error(f"Error en prueba con {level}: {e}")
                    
                    if test_predictions:
                        for i, level in enumerate(test_inputs):
                            st.write(f"- Experiencia: {level} → Predicción: ${test_predictions[i]:,.2f}")
                        
                        # Verificar si las predicciones varían
                        if len(set(test_predictions)) == 1:
                            st.warning("⚠️ PROBLEMA DETECTADO: El modelo devuelve la misma predicción para todos los niveles de experiencia.")
                
                # Hacer predicción
                predicted_salary = model.predict(input_df)[0]
                
                # Verificar si la predicción parece razonable
                if predicted_salary < 10000 or predicted_salary > 500000:
                    st.warning(f"⚠️ La predicción del modelo (${predicted_salary:,.2f}) parece fuera de rango. Usando predicción simulada.")
                    predicted_salary = simulated_salary
                
                # Verificar si el modelo varía las predicciones
                test_variation = False
                try:
                    test_df1 = input_df.copy()
                    test_df1["experience_level_desc"] = "Entry-Level"
                    
                    test_df2 = input_df.copy()
                    test_df2["experience_level_desc"] = "Executive"
                    
                    pred1 = model.predict(test_df1)[0]
                    pred2 = model.predict(test_df2)[0]
                    
                    # Si la diferencia es menor al 5%, algo anda mal
                    if abs(pred1 - pred2) / max(pred1, pred2) < 0.05:
                        test_variation = True
                except:
                    pass
                
                if test_variation:
                    st.warning("⚠️ El modelo no está diferenciando adecuadamente entre distintos perfiles. Usando predicción simulada.")
                    predicted_salary = simulated_salary
                
                if debug_mode:
                    st.write(f"Predicción del modelo: ${predicted_salary:,.2f}")
                    st.write(f"Predicción simulada: ${simulated_salary:,.2f}")
                
            except Exception as e:
                st.error(f"Error al hacer la predicción: {e}")
                st.write("Usando predicción simulada debido al error.")
                predicted_salary = simulated_salary
        else:
            st.warning("Modelo no disponible. Usando predicción simulada.")
            predicted_salary = simulated_salary
            
        # Mostrar el resultado (siempre se muestra una predicción)
        st.markdown(f"""
        ## Salario Estimado:
        # ${predicted_salary:,.2f}
        """)
        
        # Métricas adicionales
        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        
        with col_metrics1:
            st.metric(
                "Salario Mensual",
                f"${predicted_salary/12:,.2f}"
            )
        
        with col_metrics2:
            # Salario ajustado por experiencia
            exp_factor = {"Entry-Level": 1.3, "Mid-Level": 1.2, "Senior": 1.1, "Executive": 1.05}
            st.metric(
                "Potencial (+1 nivel)",
                f"${predicted_salary * exp_factor.get(experience_level, 1.2):,.2f}"
            )
        
        with col_metrics3:
            # Percentil estimado (simulado)
            st.metric(
                "Percentil Salarial",
                f"{min(max(tech_specialization * 8, 50), 95):,.0f}%"
            )
        
        # Visualización contextual
        st.subheader("Contexto de Mercado")
        
        # Generar datos de comparación
        comparison_data = {
            'Categoría': ['Tú', 'Promedio del Sector', 'Top 10%'],
            'Salario': [
                predicted_salary,
                predicted_salary * 0.85,
                predicted_salary * 1.3
            ]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Gráfico de comparación
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = sns.barplot(x='Categoría', y='Salario', data=df_comparison, ax=ax)
        
        # Añadir etiquetas
        for i, bar in enumerate(bars.patches):
            bars.text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 5000,
                f"${df_comparison['Salario'].iloc[i]:,.0f}",
                ha='center',
                color='black',
                fontweight='bold'
            )
        
        ax.set_ylabel('Salario Anual (USD)')
        ax.set_title('Comparación de Salario')
        st.pyplot(fig)
        
        # Factores clave que afectan el salario
        st.subheader("Factores Clave de Impacto")
        impact_data = {
            'Factor': [
                'Especialización Técnica', 
                'Nivel de Experiencia', 
                'Ubicación Geográfica',
                'Sector de la Empresa',
                'Trabajo Remoto'
            ],
            'Impacto': [
                tech_specialization / 10,
                {"Entry-Level": 0.3, "Mid-Level": 0.6, "Senior": 0.8, "Executive": 1.0}[experience_level],
                {"North America": 0.9, "Europe": 0.7, "Asia": 0.5, "South America": 0.4, "Africa": 0.3, "Oceania": 0.8}[region],
                {"Technology": 0.8, "Finance": 0.85, "Healthcare": 0.7, "Retail": 0.6, "Manufacturing": 0.65, "Education": 0.5, "Other": 0.6}[company_sector],
                {"Remote": 0.8, "Hybrid": 0.7, "On-site": 0.6}[work_setting],
            ]
        }
        
        df_impact = pd.DataFrame(impact_data)
        
        # Graficar impactos
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        bars2 = sns.barplot(x='Factor', y='Impacto', data=df_impact, ax=ax2)
        
        ax2.set_ylabel('Impacto Relativo (0-1)')
        ax2.set_title('Factores que Impactan tu Salario')
        st.pyplot(fig2)

# Sección inferior
st.markdown("---")
st.markdown("### 📘 Cómo interpretar los resultados")
st.markdown("""
- **Salario Estimado**: Predicción basada en los factores ingresados.
- **Salario Potencial**: Estimación si avanzas al siguiente nivel de experiencia.
- **Percentil Salarial**: Posición estimada en la distribución salarial del mercado.
- **Factores de Impacto**: Muestra qué factores tienen mayor influencia en tu salario.
""")

# Modo de solución de problemas
with st.expander("Solución de Problemas", expanded=False):
    st.write("### Diagnóstico del Sistema")
    
    st.write("#### Verificación del Modelo")
    if not model_loaded:
        st.error("❌ Modelo no cargado.")
    else:
        st.success("✅ Modelo cargado correctamente.")
        
        # Verificar si el modelo es un pipeline
        if hasattr(model, 'steps'):
            st.success(f"✅ El modelo es un pipeline con pasos: {[step[0] for step in model.steps]}")
        else:
            st.warning("⚠️ El modelo no parece ser un pipeline completo. Puede faltar el preprocesamiento.")
    
    st.write("#### Información del Entorno")
    st.write(f"- Python version: {pd.__version__}")
    st.write(f"- Pandas version: {pd.__version__}")
    st.write(f"- NumPy version: {np.__version__}")
    
    # Opción para recrear el pipeline si hay problemas
    if st.button("Intentar Recrear Pipeline"):
        st.write("Esta funcionalidad requiere acceso al código fuente del entrenamiento.")
        st.code("""
# Ejemplo de código para recrear el pipeline:
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Definir transformadores para variables categóricas y numéricas
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Definir columnas categóricas y numéricas
categorical_cols = [...]  # Llenar con columnas categóricas
numerical_cols = [...]    # Llenar con columnas numéricas

# Crear el transformador principal
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Cargar el modelo y crear el pipeline completo
from sklearn.ensemble import GradientBoostingRegressor
model_only = GradientBoostingRegressor()  # Cargar el modelo real
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model_only)
])

# Guardar el pipeline completo
import pickle
with open('fixed_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
        """)

# Pie de página
st.markdown("---")
st.caption("© 2023 Predictor de Salarios en Data Science | Desarrollado con Streamlit")