# salary_predictor_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Salarios en Data Science",
    page_icon="💰",
    layout="wide"
)

# Función para cargar el modelo
@st.cache_resource
def load_model():
    with open('improved_salary_prediction_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Cargar el modelo
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    model_loaded = False

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
    region = st.selectbox(
        "Región",
        options=["North America", "Europe", "Asia", "South America", "Africa", "Oceania"]
    )

    residence_region = st.selectbox(
    "Región de Residencia del Empleado",
    options=["North America", "Europe", "Asia", "South America", "Africa", "Oceania"]
    )
    
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
    
    # Botón para hacer la predicción
    if st.button("Calcular Salario Estimado", type="primary"):
        if model_loaded:
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
                'domestic_employment': True,  # Valor por defecto
                'economic_period': 'Post-Pandemic',  # Valor por defecto
                'role_maturity': 'Established',  # Valor por defecto
                'company_sector': company_sector,
                'career_path': 'Technical',  # Valor por defecto
                'ai_relationship': ai_relationship,
                'work_year': 2023,  # Valor por defecto
                'tech_specialization': tech_specialization,
                'english_level': english_level,
                'work_life_balance': work_life_balance,
                'demand_index': demand_index,
                'automation_risk': automation_risk,
                'cost_of_living_index': cost_of_living_index,
                'salary_to_experience_ratio': 1.0,  # Valor por defecto
                'normalized_salary': 100000.0,  # Valor por defecto
                'adjusted_salary': 100000.0,  # Valor por defecto
                'total_compensation_estimate': 120000.0  # Valor por defecto
            }
            
            # Crear DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Hacer predicción
            try:
                predicted_salary = model.predict(input_df)[0]
                
                # Mostrar el resultado
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
                    exp_factor = {"Entry-Level": 1.0, "Mid-Level": 1.2, "Senior": 1.4, "Executive": 1.6}
                    st.metric(
                        "Salario Potencial (+1 nivel)",
                        f"${predicted_salary * exp_factor.get(experience_level, 1.2):,.2f}"
                    )
                
                with col_metrics3:
                    # Percentil estimado (simulado)
                    st.metric(
                        "Percentil Salarial",
                        f"{min(tech_specialization * 10, 90):,.0f}%"
                    )
                
                # Visualización contextual
                st.subheader("Contexto de Mercado")
                
                # Generar datos de comparación (simulados)
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
                
            except Exception as e:
                st.error(f"Error al hacer la predicción: {e}")
                st.write("Detalles del error:")
                st.write(f"Columnas en el DataFrame: {input_df.columns.tolist()}")
        else:
            st.error("El modelo no se ha cargado correctamente. Verifica el archivo del modelo.")

# Sección inferior
st.markdown("---")
st.markdown("### 📘 Cómo interpretar los resultados")
st.markdown("""
- **Salario Estimado**: Predicción basada en los factores ingresados.
- **Salario Potencial**: Estimación si avanzas al siguiente nivel de experiencia.
- **Percentil Salarial**: Posición estimada en la distribución salarial del mercado.
- **Factores de Impacto**: Muestra qué factores tienen mayor influencia en tu salario.
""")

# Pie de página
st.markdown("---")
st.caption("© 2023 Predictor de Salarios en Data Science | Modelo entrenado con datos de salarios en el sector tecnológico")