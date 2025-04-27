# salary_predictor_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Data Science Salary Predictor",
    page_icon="💰",
    layout="wide"
)

# Tech stacks by role
TECH_STACKS = {
    "Data Scientist": ["Python", "R", "SQL", "Pandas", "Scikit-learn", "TensorFlow/PyTorch", "Statistics", "Jupyter"],
    "ML Engineer": ["Python", "TensorFlow", "PyTorch", "MLOps", "Docker", "Kubernetes", "AWS/GCP/Azure", "CI/CD"],
    "Data Engineer": ["Python", "SQL", "Spark", "Airflow", "ETL", "Hadoop", "Databases", "Cloud"],
    "Data Analyst": ["SQL", "Python/R", "Excel", "Tableau/PowerBI", "Basic Statistics", "Data Visualization"],
    "Business Analyst": ["Excel", "SQL", "Tableau/PowerBI", "Business Analysis", "Descriptive Statistics"],
    "Research Scientist": ["Python/R", "Advanced Statistics", "Machine Learning", "Academic Papers", "NLP/Computer Vision"],
    "Data Science Manager": ["Team Management", "Python", "Agile Methodologies", "Planning", "Communication"],
    "AI Engineer": ["Python", "TensorFlow/PyTorch", "NLP", "Computer Vision", "MLOps", "Neural Networks"]
}

# Function to load the model
@st.cache_resource
def load_model():
    try:
        model_path = 'simple_salary_model.pkl'
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {os.path.abspath(model_path)}")
            return None
            
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to validate that the model responds to changes in experience
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
        
        # Calculate variation
        variation = (max(predictions) - min(predictions)) / np.mean(predictions)
        if variation > 0.3:  # At least 30% variation
            return True
        else:
            st.error(f"❌ Low model sensitivity (variation: {variation:.1%})")
            return False
    except Exception as e:
        st.error(f"Error in sensitivity test: {e}")
        return False

# Function to adjust base salary using additional factors
def adjust_salary_with_details(base_salary, job_category, tech_specialization, english_level, company_size, employment_type):
    """Adjust the base salary prediction using additional factors"""
    adjusted_salary = base_salary
    
    # Apply Business Analyst rule (15% less than Data Analyst)
    if job_category == "Business Analyst":
        adjusted_salary *= 0.85  # Business Analysts earn 15% less than Data Analysts
    
    # Technical specialization adjustment
    if tech_specialization <= 3:
        adjusted_salary *= 0.85  # -15% for low specialization
    elif tech_specialization <= 6:
        adjusted_salary *= 1.0   # No change for medium specialization
    elif tech_specialization <= 8:
        adjusted_salary *= 1.08  # +8% for high specialization
    else:
        adjusted_salary *= 1.15  # +15% for very high specialization
    
    # English level adjustment
    if english_level <= 4:
        adjusted_salary *= 0.92  # -8% for low English level
    elif english_level <= 7:
        adjusted_salary *= 1.0   # No change for medium English level
    else:
        adjusted_salary *= 1.05  # +5% for high English level
    
    # Company size adjustment
    company_size_adj = {
        "Small": 0.95,    # -5% for small companies
        "Medium": 1.0,    # No change for medium companies
        "Large": 1.08     # +8% for large companies
    }
    adjusted_salary *= company_size_adj.get(company_size, 1.0)
    
    # Employment type adjustment
    employment_type_adj = {
        "Full-time": 1.0,    # Base for full-time
        "Part-time": 0.95,   # -5% for part-time (hourly could be higher but annual lower)
        "Contract": 1.1,     # +10% for contract (higher rate but fewer benefits)
        "Freelance": 1.15    # +15% for freelance (higher rate but less stability)
    }
    adjusted_salary *= employment_type_adj.get(employment_type, 1.0)
    
    return adjusted_salary

# Function to generate salary improvement recommendations
def generate_recommendations(job_category, experience_level, tech_specialization, english_level, region):
    """Generate personalized recommendations to increase salary potential"""
    recommendations = []
    
    # Experience-based recommendations
    if experience_level == "Entry-Level":
        recommendations.append("📈 Aim for specialized certifications in your field to stand out from other entry-level candidates.")
    elif experience_level == "Mid-Level":
        recommendations.append("📈 Focus on leading small projects to demonstrate leadership capabilities.")
    elif experience_level == "Senior":
        recommendations.append("📈 Develop mentorship skills and consider pursuing management opportunities.")
    
    # Tech specialization recommendations
    if tech_specialization < 7:
        recommendations.append("💻 Increasing your technical specialization could boost your salary by up to 15%.")
        
        # Add specific tech recommendations based on role
        if job_category in TECH_STACKS:
            top_techs = TECH_STACKS[job_category][:3]  # Get top 3 skills for the role
            recommendations.append(f"🔧 For your role as {job_category}, focus on mastering: {', '.join(top_techs)}.")
    
    # English level recommendations
    if english_level < 8 and region in ["North America", "Europe"]:
        recommendations.append("🌎 Improving your English skills could increase your salary potential by 5% in international markets.")
    
    # Region-specific recommendations
    if region not in ["North America", "Europe"]:
        recommendations.append("🌍 Consider remote opportunities with companies based in North America or Europe for higher compensation.")
    
    # Add role-specific recommendations
    if job_category == "Data Scientist":
        recommendations.append("🔬 Specializing in causal inference or experimental design can lead to higher-paying positions.")
    elif job_category == "ML Engineer":
        recommendations.append("🚀 MLOps skills and production deployment expertise are highly valued and can command premium salaries.")
    elif job_category == "Data Engineer":
        recommendations.append("⚙️ Cloud certifications (AWS, Azure, GCP) can significantly increase your market value.")
    elif job_category == "Data Analyst":
        recommendations.append("📊 Developing programming skills in Python can help transition to higher-paying Data Scientist roles.")
    elif job_category == "Business Analyst":
        recommendations.append("📊 Developing technical skills like SQL proficiency and data visualization can help bridge the salary gap with Data Analysts.")
    
    return recommendations

# Function to visualize tech stack for a role
def plot_tech_stack(job_category):
    """Create a visual representation of the tech stack for a given role"""
    if job_category not in TECH_STACKS:
        return None
    
    tech_stack = TECH_STACKS[job_category]
    skill_levels = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55][:len(tech_stack)]  # Importance levels
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create horizontal bar chart
    bars = ax.barh(tech_stack, skill_levels, color='skyblue')
    
    # Add a color gradient
    cmap = plt.cm.Blues
    for i, bar in enumerate(bars):
        bar.set_color(cmap(0.5 + skill_levels[i]/2))
    
    # Add value annotations
    for i, v in enumerate(skill_levels):
        ax.text(v + 0.01, i, f"{'★' * int(v * 5)}", va='center')
    
    # Customize chart
    ax.set_title(f'Key Technologies for {job_category}', fontsize=16)
    ax.set_xlabel('Relative Importance', fontsize=12)
    ax.set_xlim(0, 1)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Remove y axis
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    return fig

# Load the model
model = load_model()
model_loaded = model is not None

# Title and description
st.title("🧠 Data Science Salary Predictor")
st.markdown("""
This application predicts salaries in the Data Science field based on key factors.
Complete the form below to get a personalized salary estimate.
""")

# Verify model silently
if model_loaded:
    use_simulation = not test_model_sensitivity(model)
    if use_simulation:
        st.error("Model validation failed. Please try again or contact support.")
        st.stop()
else:
    st.error("Model could not be loaded. Please check the model file.")
    st.stop()

# Create main columns
col1, col2 = st.columns([1, 1])

# Input section (left column)
with col1:
    st.header("📋 Professional Profile")
    
    # KEY VARIABLES FOR THE MODEL
    st.subheader("Primary Information")
    st.info("⚠️ These factors directly affect the model's prediction")
    
    job_category = st.selectbox(
        "Job Category",
        options=[
            "Data Scientist", "ML Engineer", "Data Engineer", 
            "Data Analyst", "Business Analyst", "Research Scientist", 
            "Data Science Manager", "AI Engineer"
        ]
    )
    
    experience_level = st.selectbox(
        "Experience Level",
        options=["Entry-Level", "Mid-Level", "Senior", "Executive"]
    )
    
    region = st.selectbox(
        "Company Region",
        options=["North America", "Europe", "Asia", "South America", "Africa", "Oceania"]
    )
    
    work_setting = st.selectbox(
        "Work Setting",
        options=["Remote", "Hybrid", "On-site"]
    )
    
    company_sector = st.selectbox(
        "Company Sector",
        options=["Technology", "Finance", "Healthcare", "Retail", "Manufacturing", "Education", "Other"]
    )
    
    # ADDITIONAL VARIABLES (USED FOR SALARY ADJUSTMENT)
    st.subheader("Additional Details")
    st.info("✓ These factors will be used to refine the prediction and provide better insights")
    
    company_size = st.selectbox(
        "Company Size",
        options=["Small", "Medium", "Large"]
    )
    
    employment_type = st.selectbox(
        "Employment Type",
        options=["Full-time", "Part-time", "Contract", "Freelance"]
    )
    
    tech_specialization = st.slider(
        "Technical Specialization",
        min_value=1.0,
        max_value=10.0,
        value=7.0,
        step=0.5,
        help="Level of technical specialization (1-10)"
    )
    
    english_level = st.slider(
        "English Proficiency",
        min_value=1.0,
        max_value=10.0,
        value=7.0,
        step=0.5
    )

# Results section (right column)
with col2:
    st.header("💰 Salary Prediction")
    
    # Prediction button
    if st.button("Calculate Estimated Salary", type="primary"):
        try:
            # Prepare data for the model (key variables only)
            input_data = {
                'job_category': job_category,
                'experience_level_desc': experience_level,
                'region': region,
                'work_setting': work_setting,
                'company_sector': company_sector
            }
            
            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            base_predicted_salary = model.predict(input_df)[0]
            
            # Verify that the prediction is reasonable
            if base_predicted_salary < 10000:
                st.warning(f"⚠️ Model prediction (${base_predicted_salary:,.2f}) seems too low. Adjusting to minimum threshold.")
                base_predicted_salary = 50000  # Minimum reasonable salary
            elif base_predicted_salary > 500000:
                st.warning(f"⚠️ Model prediction (${base_predicted_salary:,.2f}) seems too high. Adjusting to maximum threshold.")
                base_predicted_salary = 300000  # Maximum reasonable salary
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.error("Unable to generate prediction. Please try again or contact support.")
            st.stop()  # Stop execution
        
        # Adjust salary using additional factors
        adjusted_salary = adjust_salary_with_details(
            base_predicted_salary, 
            job_category,
            tech_specialization, 
            english_level, 
            company_size, 
            employment_type
        )
        
        # Show both base and adjusted predictions
        col_preds1, col_preds2 = st.columns(2)
        
        with col_preds1:
            st.subheader("Base Prediction")
            st.markdown(f"### ${base_predicted_salary:,.2f}")
            st.caption("Based on primary factors only")
        
        with col_preds2:
            st.subheader("Adjusted Prediction")
            st.markdown(f"### ${adjusted_salary:,.2f}")
            st.caption("Refined with additional factors")
        
        # Show the adjustment percentage
        adjustment_pct = (adjusted_salary / base_predicted_salary - 1) * 100
        adjustment_text = "increase" if adjustment_pct >= 0 else "decrease"
        
        st.info(f"Your profile details resulted in a {abs(adjustment_pct):.1f}% {adjustment_text} from the base prediction.")
        
        if job_category == "Business Analyst":
            st.info("Note: Business Analyst salaries are typically 15% lower than comparable Data Analyst positions.")
        
        # Additional metrics calculated with rules
        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        
        with col_metrics1:
            st.metric(
                "Monthly Salary",
                f"${adjusted_salary/12:,.2f}"
            )
        
        with col_metrics2:
            # Salary adjusted by experience
            exp_factor = {"Entry-Level": 1.3, "Mid-Level": 1.2, "Senior": 1.1, "Executive": 1.05}
            st.metric(
                "Potential (+1 level)",
                f"${adjusted_salary * exp_factor.get(experience_level, 1.2):,.2f}"
            )
        
        with col_metrics3:
            # Estimated percentile based on experience, region and specialization
            percentiles = {
                "Entry-Level": 30,
                "Mid-Level": 55,
                "Senior": 80,
                "Executive": 90
            }
            percentile_base = percentiles.get(experience_level, 50)
            
            # Adjust by region
            region_adjustment = {
                "North America": 10, 
                "Europe": 5, 
                "Asia": -5, 
                "South America": -8, 
                "Africa": -10, 
                "Oceania": 0
            }
            
            # Adjust by technical specialization
            tech_adjustment = (tech_specialization - 5) * 2
            
            final_percentile = min(95, max(5, percentile_base + region_adjustment.get(region, 0) + tech_adjustment))
            
            st.metric(
                "Salary Percentile",
                f"{int(final_percentile)}%"
            )
        
        # Visualization: Market Context
        st.subheader("Market Context")
        
        # Generate comparison data based on rules
        comparison_data = {
            'Category': ['Your Profile', 'Industry Average', 'Top 10%'],
            'Salary': [
                adjusted_salary,
                adjusted_salary * 0.85,
                adjusted_salary * 1.3
            ]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Comparison chart
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['#1f77b4', '#7f7f7f', '#2ca02c']
        bars = sns.barplot(x='Category', y='Salary', data=df_comparison, ax=ax, palette=colors)
        
        # Add labels
        for i, bar in enumerate(bars.patches):
            bars.text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 5000,
                f"${df_comparison['Salary'].iloc[i]:,.0f}",
                ha='center',
                color='black',
                fontweight='bold'
            )
        
        ax.set_ylabel('Annual Salary (USD)')
        ax.set_title('Salary Comparison')
        st.pyplot(fig)
        
        # Visualization: Tech Stack
        st.subheader(f"Technology Stack for {job_category}")
        
        tech_fig = plot_tech_stack(job_category)
        if tech_fig:
            st.pyplot(tech_fig)
        else:
            st.write("No technology stack information available for this role.")
        
        # Salary growth trajectory
        st.subheader("Salary Growth Trajectory")
        
        # Create salary projection data
        years = np.arange(0, 6)
        growth_rates = {
            "Entry-Level": 0.12,  # 12% annual growth
            "Mid-Level": 0.08,    # 8% annual growth
            "Senior": 0.05,       # 5% annual growth
            "Executive": 0.04     # 4% annual growth
        }
        
        growth_rate = growth_rates.get(experience_level, 0.07)
        
        # Calculate projected salaries
        projected_salaries = [adjusted_salary * (1 + growth_rate) ** year for year in years]
        
        # Create projection dataframe
        projection_df = pd.DataFrame({
            'Year': [f"Current"] + [f"Year {y}" for y in years[1:]],
            'Salary': projected_salaries
        })
        
        # Plot projection
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        sns.lineplot(x='Year', y='Salary', data=projection_df, marker='o', linewidth=2, ax=ax3)
        
        # Add value labels
        for i, val in enumerate(projected_salaries):
            ax3.text(i, val + 5000, f"${val:,.0f}", ha='center')
        
        # Customize chart
        ax3.set_title('Projected Salary Growth (Based on Industry Averages)')
        ax3.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig3)
        
        # Impact factors (calculated with rules)
        st.subheader("Impact Factors")
        impact_data = {
            'Factor': [
                'Experience Level', 
                'Region',
                'Company Sector',
                'Work Setting',
                'Job Category'
            ],
            'Impact': [
                {"Entry-Level": 0.3, "Mid-Level": 0.6, "Senior": 0.8, "Executive": 1.0}[experience_level],
                {"North America": 0.9, "Europe": 0.7, "Asia": 0.5, "South America": 0.4, "Africa": 0.3, "Oceania": 0.8}[region],
                {"Technology": 0.8, "Finance": 0.85, "Healthcare": 0.7, "Retail": 0.6, "Manufacturing": 0.65, "Education": 0.5, "Other": 0.6}[company_sector],
                {"Remote": 0.7, "Hybrid": 0.6, "On-site": 0.5}[work_setting],
                {"Data Scientist": 0.8, "ML Engineer": 0.85, "Data Engineer": 0.75, "Data Analyst": 0.6, "Business Analyst": 0.5, "Research Scientist": 0.9, "Data Science Manager": 0.95, "AI Engineer": 0.85}.get(job_category, 0.7)
            ]
        }
        
        df_impact = pd.DataFrame(impact_data)
        
        # Sort by impact
        df_impact = df_impact.sort_values('Impact', ascending=False).reset_index(drop=True)
        
        # Plot impacts
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        colors2 = sns.color_palette("viridis", len(df_impact))
        bars2 = sns.barplot(x='Factor', y='Impact', data=df_impact, ax=ax2, palette=colors2)
        
        # Add percentages
        for i, bar in enumerate(bars2.patches):
            bars2.text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 0.02,
                f"{df_impact['Impact'].iloc[i]:.0%}",
                ha='center',
                color='black',
                fontweight='bold'
            )
        
        ax2.set_ylabel('Relative Impact')
        ax2.set_title('Factors Influencing Your Salary')
        ax2.set_ylim(0, 1.1)  # Adjust limit for labels
        st.pyplot(fig2)
        
        # Personalized recommendations
        st.subheader("💡 Salary Growth Recommendations")
        
        recommendations = generate_recommendations(
            job_category, 
            experience_level, 
            tech_specialization, 
            english_level, 
            region
        )
        
        for rec in recommendations:
            st.markdown(f"- {rec}")

# Bottom section - Interpretation
st.markdown("---")
st.header("📊 Interpreting Results")

# Create columns for explanation sections
col_exp1, col_exp2, col_exp3 = st.columns(3)

with col_exp1:
    st.subheader("About the Prediction")
    st.markdown("""
    - The **Base Prediction** uses only the 5 main factors.
    - The **Adjusted Prediction** incorporates your additional details.
    - **Monthly Salary** is simply divided by 12 for quick reference.
    - **Potential** shows expected growth at the next experience level.
    """)

with col_exp2:
    st.subheader("Key Factors")
    st.markdown("""
    - **Experience Level**: The single most determinant factor.
    - **Geographic Region**: North America and Europe typically offer higher salaries.
    - **Sector**: Finance and technology tend to pay better than education or retail.
    - **Work Setting**: Remote work can affect compensation in both directions.
    - **Job Category**: Specialized roles like ML Engineer or Research Scientist often command higher pay.
    """)

with col_exp3:
    st.subheader("Limitations")
    st.markdown("""
    - Predictions are **estimates** based on historical data.
    - Factors like specific company size, particular skills, or individual negotiation can significantly modify the final salary.
    - High-demand talent markets may exceed these estimates.
    - The model doesn't account for recent labor market trends beyond the training data.
    """)

# Footer
st.markdown("---")
st.caption("© 2023 Data Science Salary Predictor | Developed with Streamlit")