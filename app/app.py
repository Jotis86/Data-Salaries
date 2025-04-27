# salary_predictor_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
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
    try:
        model_path = 'simple_salary_model.pkl'
        if not os.path.exists(model_path):
            st.error(f"Archivo de modelo no encontrado en: {os.path.abspath(model_path)}")
            return None
            
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            
        # Verificar si es un pipeline
        if hasattr(model, 'steps'):
            steps = [step[0] for step in model.steps]
            st.write(f"Modelo cargado correctamente. Pasos: {steps}")
        else:
            st.warning("El modelo cargado no parece ser un pipeline completo.")
        
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# Función para validar que el modelo responde a cambios en experiencia
def test_model_sensitivity(model):
    try:
        test_inputs = {
            'job_category': ['Data Scientist'] * 4,
            'experience_level_desc': ['Entry-Level', 'Mid-Level', 'Senior', 'Executive'],
            'region': ['North America'] * 4,
            'work_setting': ['Remote'] * 4,
            'company_sector': ['Technology'] * 4
        }
        test_df = pd.DataFrame(test_inputs)
        
        predictions = model.predict(test_df)
        
        # Crear DataFrame para mostrar
        results_df = pd.DataFrame({
            'Nivel de Experiencia': test_inputs['experience_level_desc'],
            'Salario Predicho': [f"${p:,.2f}" for p in predictions]
        })
        
        st.write("Prueba de Sensibilidad del Modelo:")
        st.dataframe(results_df)
        
        # Calcular variación
        variation = (max(predictions) - min(predictions)) / np.mean(predictions)
        if variation > 0.3:  # Al menos 30% de variación
            st.success(f"✅ Modelo sensible a nivel de experiencia (variación: {variation:.1%})")
            return True
        else:
            st.error(f"❌ Baja sensibilidad a experiencia (variación: {variation:.1%})")
            return False
    except Exception as e:
        st.error(f"Error en prueba de sensibilidad: {e}")
        return False

# Función para predicciones de respaldo (si el modelo falla)
def get_simulated_prediction(
    job_category, experience_level, region, work_setting, company_sector
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
    
    # Calcular salario final
    salary = base * exp_mult * reg_mult * sec_mult * work_mult
    
    # Añadir variabilidad (±5%)
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
Esta aplicación predice salarios en el sector de Data Science basándose en los principales 
factores que influyen en la compensación. Complete los campos para obtener una estimación personalizada.
""")

# Verificar modelo
if model_loaded:
    if test_model_sensitivity(model):
        use_simulation = False
    else:
        use_simulation = True
        st.warning("⚠️ Utilizando sistema de predicción de respaldo (modelo basado en reglas)")
else:
    use_simulation = True
    st.warning("⚠️ Modelo no disponible. Utilizando sistema de predicción de respaldo.")

# Crear columnas principales
col1, col2 = st.columns([1, 1])

# Sección de entrada de datos (columna izquierda)
with col1:
    st.header("📋 Perfil Profesional")
    
    # VARIABLES CLAVE PARA EL MODELO
    st.subheader("Información Principal")
    
    job_category = st.selectbox(
        "Categoría de Trabajo",
        options=[
            "Data Scientist", "ML Engineer", "Data Engineer", 
            "Data Analyst", "Business Analyst", "Research Scientist", 
            "Data Science Manager", "AI Engineer"
        ]
    )
    
    experience_level = st.selectbox(
        "Nivel de Experiencia",
        options=["Entry-Level", "Mid-Level", "Senior", "Executive"]
    )
    
    region = st.selectbox(
        "Región de la Empresa",
        options=["North America", "Europe", "Asia", "South America", "Africa", "Oceania"]
    )
    
    work_setting = st.selectbox(
        "Modalidad de Trabajo",
        options=["Remote", "Hybrid", "On-site"]
    )
    
    company_sector = st.selectbox(
        "Sector de la Empresa",
        options=["Technology", "Finance", "Healthcare", "Retail", "Manufacturing", "Education", "Other"]
    )
    
    # VARIABLES ADICIONALES (NO USADAS POR EL MODELO)
    st.subheader("Detalles Adicionales")
    
    with st.expander("Información Complementaria", expanded=False):
        company_size = st.selectbox(
            "Tamaño de la Empresa",
            options=["Small", "Medium", "Large"]
        )
        
        employment_type = st.selectbox(
            "Tipo de Empleo",
            options=["Full-time", "Part-time", "Contract", "Freelance"]
        )
        
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

# Columna derecha para resultados y visualizaciones
with col2:
    st.header("💰 Predicción de Salario")
    
    # Botón para hacer la predicción
    if st.button("Calcular Salario Estimado", type="primary"):
        # Siempre calcular la predicción simulada como respaldo
        simulated_salary = get_simulated_prediction(
            job_category, experience_level, region, 
            work_setting, company_sector
        )
        
        if model_loaded and not use_simulation:
            try:
                # Preparar datos para el modelo simple (solo las variables clave)
                input_data = {
                    'job_category': job_category,
                    'experience_level_desc': experience_level,
                    'region': region,
                    'work_setting': work_setting,
                    'company_sector': company_sector
                }
                
                # Crear DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Hacer predicción
                predicted_salary = model.predict(input_df)[0]
                
                # Verificar que la predicción es razonable
                if predicted_salary < 10000 or predicted_salary > 500000:
                    st.warning(f"⚠️ La predicción del modelo (${predicted_salary:,.2f}) parece fuera de rango. Usando predicción simulada.")
                    predicted_salary = simulated_salary
                
            except Exception as e:
                st.error(f"Error al hacer la predicción: {e}")
                st.write("Usando predicción simulada debido al error.")
                predicted_salary = simulated_salary
        else:
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
            # Percentil estimado basado en experiencia y región
            percentiles = {
                "Entry-Level": 30,
                "Mid-Level": 55,
                "Senior": 80,
                "Executive": 90
            }
            percentile_base = percentiles.get(experience_level, 50)
            
            # Ajustar según región
            region_adjustment = {
                "North America": 10, 
                "Europe": 5, 
                "Asia": -5, 
                "South America": -8, 
                "Africa": -10, 
                "Oceania": 0
            }
            final_percentile = min(95, max(5, percentile_base + region_adjustment.get(region, 0)))
            
            st.metric(
                "Percentil Salarial",
                f"{final_percentile}%"
            )
        
        # Visualización contextual
        st.subheader("Contexto de Mercado")
        
        # Generar datos de comparación
        comparison_data = {
            'Categoría': ['Tu Perfil', 'Promedio Sector', 'Top 10%'],
            'Salario': [
                predicted_salary,
                predicted_salary * 0.85,
                predicted_salary * 1.3
            ]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Gráfico de comparación
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['#1f77b4', '#7f7f7f', '#2ca02c']
        bars = sns.barplot(x='Categoría', y='Salario', data=df_comparison, ax=ax, palette=colors)
        
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
        st.subheader("Factores de Impacto")
        impact_data = {
            'Factor': [
                'Nivel de Experiencia', 
                'Región',
                'Sector de la Empresa',
                'Modalidad de Trabajo',
                'Categoría de Trabajo'
            ],
            'Impacto': [
                {"Entry-Level": 0.3, "Mid-Level": 0.6, "Senior": 0.8, "Executive": 1.0}[experience_level],
                {"North America": 0.9, "Europe": 0.7, "Asia": 0.5, "South America": 0.4, "Africa": 0.3, "Oceania": 0.8}[region],
                {"Technology": 0.8, "Finance": 0.85, "Healthcare": 0.7, "Retail": 0.6, "Manufacturing": 0.65, "Education": 0.5, "Other": 0.6}[company_sector],
                {"Remote": 0.7, "Hybrid": 0.6, "On-site": 0.5}[work_setting],
                {"Data Scientist": 0.8, "ML Engineer": 0.85, "Data Engineer": 0.75, "Data Analyst": 0.6, "Business Analyst": 0.55, "Research Scientist": 0.9, "Data Science Manager": 0.95, "AI Engineer": 0.85}.get(job_category, 0.7)
            ]
        }
        
        df_impact = pd.DataFrame(impact_data)
        
        # Ordenar por impacto
        df_impact = df_impact.sort_values('Impacto', ascending=False).reset_index(drop=True)
        
        # Graficar impactos
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        colors2 = sns.color_palette("viridis", len(df_impact))
        bars2 = sns.barplot(x='Factor', y='Impacto', data=df_impact, ax=ax2, palette=colors2)
        
        # Añadir porcentajes
        for i, bar in enumerate(bars2.patches):
            bars2.text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 0.02,
                f"{df_impact['Impacto'].iloc[i]:.0%}",
                ha='center',
                color='black',
                fontweight='bold'
            )
        
        ax2.set_ylabel('Impacto Relativo')
        ax2.set_title('Factores que Influyen en el Salario')
        ax2.set_ylim(0, 1.1)  # Ajustar límite para etiquetas
        st.pyplot(fig2)

# Sección inferior - Interpretación
st.markdown("---")
st.header("📊 Interpretación de Resultados")

# Crear columnas para secciones explicativas
col_exp1, col_exp2, col_exp3 = st.columns(3)

with col_exp1:
    st.subheader("Sobre la Predicción")
    st.markdown("""
    - El **Salario Estimado** se basa en los 5 factores más importantes que influyen en la compensación.
    - El **Salario Mensual** es una simple división por 12 para referencia rápida.
    - El **Potencial** muestra el crecimiento esperado al avanzar al siguiente nivel de experiencia.
    """)

with col_exp2:
    st.subheader("Factores Clave")
    st.markdown("""
    - **Nivel de Experiencia**: El factor individual más determinante.
    - **Región Geográfica**: Norte América y Europa suelen tener salarios más altos.
    - **Sector**: Finanzas y tecnología tienden a pagar mejor que educación o comercio.
    - **Modalidad**: El trabajo remoto puede afectar la compensación en ambas direcciones.
    - **Categoría**: Roles especializados como ML Engineer o Research Scientist suelen tener mejor paga.
    """)

with col_exp3:
    st.subheader("Limitaciones")
    st.markdown("""
    - Las predicciones son **estimaciones** basadas en datos históricos.
    - Factores como el tamaño específico de la empresa, habilidades particulares, o negociaciones individuales pueden modificar significativamente el salario final.
    - Los mercados con alta demanda de talento pueden superar estas estimaciones.
    - El modelo no considera tendencias recientes del mercado laboral posteriores a los datos de entrenamiento.
    """)

# Pie de página
st.markdown("---")
st.caption("© 2023 Predictor de Salarios en Data Science | Desarrollado con Streamlit")

# Diagnóstico avanzado (oculto por defecto)
with st.expander("🛠️ Diagnóstico y Solución de Problemas", expanded=False):
    st.write("### Información del Sistema")
    
    # Estado del modelo
    if model_loaded:
        if use_simulation:
            st.warning("⚠️ Modelo cargado pero usando sistema de respaldo")
        else:
            st.success("✅ Modelo cargado y funcionando correctamente")
    else:
        st.error("❌ Modelo no cargado")
    
    # Información técnica
    st.write("#### Detalles Técnicos")
    st.write(f"- Python: {pd.__version__}")
    st.write(f"- Pandas: {pd.__version__}")
    st.write(f"- NumPy: {np.__version__}")
    st.write(f"- Ruta del modelo: {os.path.abspath('simple_salary_model.pkl')}")
    
    # Prueba manual
    st.write("#### Prueba Manual del Modelo")
    if st.button("Ejecutar Prueba Completa"):
        if model_loaded:
            # Crear datos de prueba variados
            test_data = []
            for job in ["Data Scientist", "ML Engineer", "Data Analyst"]:
                for exp in ["Entry-Level", "Mid-Level", "Senior"]:
                    for reg in ["North America", "Europe", "Asia"]:
                        test_data.append({
                            'job_category': job,
                            'experience_level_desc': exp,
                            'region': reg,
                            'work_setting': 'Remote',
                            'company_sector': 'Technology'
                        })
            
            test_df = pd.DataFrame(test_data)
            
            try:
                predictions = model.predict(test_df)
                
                # Añadir predicciones al DataFrame
                test_df['Salario Predicho'] = [f"${p:,.2f}" for p in predictions]
                
                # Mostrar resultados
                st.dataframe(test_df)
                
                # Análisis de variación
                st.write("#### Análisis de Variación")
                
                # Por experiencia
                exp_df = pd.DataFrame({
                    'Nivel': ["Entry-Level", "Mid-Level", "Senior"],
                    'Salario Promedio': [
                        predictions[test_df['experience_level_desc'] == "Entry-Level"].mean(),
                        predictions[test_df['experience_level_desc'] == "Mid-Level"].mean(),
                        predictions[test_df['experience_level_desc'] == "Senior"].mean()
                    ]
                })
                
                st.write("Por Nivel de Experiencia:")
                st.dataframe(exp_df)
                
                # Por región
                reg_df = pd.DataFrame({
                    'Región': ["North America", "Europe", "Asia"],
                    'Salario Promedio': [
                        predictions[test_df['region'] == "North America"].mean(),
                        predictions[test_df['region'] == "Europe"].mean(),
                        predictions[test_df['region'] == "Asia"].mean()
                    ]
                })
                
                st.write("Por Región:")
                st.dataframe(reg_df)
                
            except Exception as e:
                st.error(f"Error en prueba completa: {e}")
        else:
            st.error("No se puede ejecutar la prueba sin un modelo cargado")