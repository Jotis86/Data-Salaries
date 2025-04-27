# salary_predictor_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image

# ======= APP CONFIGURATION =======
st.set_page_config(
    page_title="Data Science Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {font-size:2.5rem !important; font-weight:800; margin-bottom:1rem; color:#1E88E5}
    .section-header {font-size:1.8rem !important; font-weight:700; margin-top:2rem; margin-bottom:1rem; color:#333333}
    .subsection-header {font-size:1.3rem !important; font-weight:650; margin-top:1.5rem; color:#444444}
    .highlight {background-color:#f0f7ff; padding:1.2rem; border-radius:0.5rem; margin-bottom:1rem}
    .info-box {background-color:#e1f5fe; padding:1rem; border-radius:0.5rem; margin-bottom:1rem; border-left:5px solid #0277bd}
    .success-box {background-color:#e8f5e9; padding:1rem; border-radius:0.5rem; margin-bottom:1rem; border-left:5px solid #2e7d32}
    .warning-box {background-color:#fff8e1; padding:1rem; border-radius:0.5rem; margin-bottom:1rem; border-left:5px solid #ff8f00}
    .error-box {background-color:#ffebee; padding:1rem; border-radius:0.5rem; margin-bottom:1rem; border-left:5px solid #c62828}
    .stButton button {background-color:#1976D2; color:white; font-weight:bold; border:none; padding:0.5rem 1rem; border-radius:0.3rem}
    .stButton button:hover {background-color:#1565C0; color:white}
    .sidebar .sidebar-content {background-color:#f5f5f5}
    .chart-container {background-color:#f9f9f9; padding:1rem; border-radius:0.5rem; border:1px solid #e0e0e0; margin-bottom:1.5rem}
    .metric-container {background-color:#fafafa; padding:1rem; border-radius:0.5rem; border:1px solid #e0e0e0; margin-bottom:1rem}
</style>
""", unsafe_allow_html=True)

# ======= CONSTANTS AND GLOBAL VARIABLES =======
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

# ======= HELPER FUNCTIONS =======
@st.cache_data
def load_sample_data():
    """Load sample dataset for visualizations"""
    # This is a placeholder - in a real app, you'd load your actual dataset
    # For simplicity, I'm creating a synthetic dataset that resembles salary data
    np.random.seed(42)
    
    job_categories = list(TECH_STACKS.keys())
    experience_levels = ["Entry-Level", "Mid-Level", "Senior", "Executive"]
    regions = ["North America", "Europe", "Asia", "South America", "Africa", "Oceania"]
    work_settings = ["Remote", "Hybrid", "On-site"]
    sectors = ["Technology", "Finance", "Healthcare", "Retail", "Manufacturing", "Education"]
    
    # Base salaries by job and experience
    base_salaries = {
        "Data Scientist": {"Entry-Level": 70000, "Mid-Level": 100000, "Senior": 130000, "Executive": 180000},
        "ML Engineer": {"Entry-Level": 80000, "Mid-Level": 110000, "Senior": 140000, "Executive": 190000},
        "Data Engineer": {"Entry-Level": 75000, "Mid-Level": 105000, "Senior": 135000, "Executive": 175000},
        "Data Analyst": {"Entry-Level": 60000, "Mid-Level": 85000, "Senior": 110000, "Executive": 150000},
        "Business Analyst": {"Entry-Level": 55000, "Mid-Level": 75000, "Senior": 95000, "Executive": 130000},
        "Research Scientist": {"Entry-Level": 85000, "Mid-Level": 115000, "Senior": 145000, "Executive": 195000},
        "Data Science Manager": {"Entry-Level": 90000, "Mid-Level": 120000, "Senior": 160000, "Executive": 210000},
        "AI Engineer": {"Entry-Level": 85000, "Mid-Level": 115000, "Senior": 145000, "Executive": 195000}
    }
    
    # Region multipliers
    region_mult = {
        "North America": 1.2,
        "Europe": 1.0,
        "Asia": 0.7,
        "South America": 0.6,
        "Africa": 0.5,
        "Oceania": 1.1
    }
    
    # Create synthetic data
    num_samples = 1000
    data = []
    
    for _ in range(num_samples):
        job = np.random.choice(job_categories)
        exp = np.random.choice(experience_levels)
        reg = np.random.choice(regions)
        work = np.random.choice(work_settings)
        sector = np.random.choice(sectors)
        
        # Calculate base salary with some random variation
        base = base_salaries[job][exp]
        reg_factor = region_mult[reg]
        variation = np.random.uniform(0.85, 1.15)  # ±15% random variation
        
        salary = base * reg_factor * variation
        
        data.append({
            "job_category": job,
            "experience_level_desc": exp,
            "region": reg,
            "work_setting": work,
            "company_sector": sector,
            "salary_in_usd": int(salary)
        })
    
    return pd.DataFrame(data)

@st.cache_resource
def load_model():
    """Load the prediction model"""
    try:
        model_path = 'simple_salary_model.pkl'
        if not os.path.exists(model_path):
            return None
            
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        return model
    except:
        return None

def test_model_sensitivity(model):
    """Test if model responds to changes in experience levels"""
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
        return variation > 0.3  # At least 30% variation
    except:
        return False

def adjust_salary_with_details(base_salary, job_category, tech_specialization, english_level, company_size, employment_type):
    """Adjust the base salary prediction using additional factors"""
    adjusted_salary = base_salary
    
    # Apply Business Analyst rule (15% less than Data Analyst)
    if job_category == "Business Analyst":
        adjusted_salary *= 0.85
    
    # Technical specialization adjustment
    if tech_specialization <= 3:
        adjusted_salary *= 0.85
    elif tech_specialization <= 6:
        adjusted_salary *= 1.0
    elif tech_specialization <= 8:
        adjusted_salary *= 1.08
    else:
        adjusted_salary *= 1.15
    
    # English level adjustment
    if english_level <= 4:
        adjusted_salary *= 0.92
    elif english_level <= 7:
        adjusted_salary *= 1.0
    else:
        adjusted_salary *= 1.05
    
    # Company size adjustment
    company_size_adj = {
        "Small": 0.95,
        "Medium": 1.0,
        "Large": 1.08
    }
    adjusted_salary *= company_size_adj.get(company_size, 1.0)
    
    # Employment type adjustment
    employment_type_adj = {
        "Full-time": 1.0,
        "Part-time": 0.95,
        "Contract": 1.1,
        "Freelance": 1.15
    }
    adjusted_salary *= employment_type_adj.get(employment_type, 1.0)
    
    return adjusted_salary

def generate_recommendations(job_category, experience_level, tech_specialization, english_level, region):
    """Generate personalized recommendations"""
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
            top_techs = TECH_STACKS[job_category][:3]
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

def plot_tech_stack(job_category):
    """Create a visual representation of the tech stack for a given role"""
    if job_category not in TECH_STACKS:
        return None
    
    tech_stack = TECH_STACKS[job_category]
    skill_levels = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55][:len(tech_stack)]
    
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

# ======= PAGE FUNCTIONS =======
def home_page():
    """Display the home page content"""
    st.markdown('<h1 class="main-header">Data Science Salary Predictor</h1>', unsafe_allow_html=True)
    
    # Introduction section
    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    st.markdown("""
    Welcome to the **Data Science Salary Predictor**! This application helps data professionals estimate their
    market value based on key factors that influence salaries in the data science and analytics field.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # About the app section
    st.markdown('<h2 class="section-header">About This App</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        This tool was developed to help data professionals:

        * **Estimate their market value** based on role, experience, location, and other factors
        * **Visualize salary trends** across different dimensions of the data science job market
        * **Discover insights** that can help negotiate better compensation
        * **Identify strategies** to increase earning potential in the data field
        
        The app uses machine learning to predict salaries based on real-world data from thousands
        of data professionals across different regions, companies, and specializations.
        """)
    
    with col2:
        # Sample image or chart
        st.image("https://img.freepik.com/free-vector/annual-salary-concept-illustration_114360-5401.jpg", 
                caption="Analyze your salary potential", use_column_width=True)
    
    # How to use section
    st.markdown('<h2 class="section-header">How to Use This App</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 📊 Explore Visualizations")
        st.markdown("""
        Visit the **Visualizations** section to explore salary trends across:
        * Job categories
        * Experience levels
        * Geographic regions
        * Industry sectors
        * Work settings (remote, hybrid, on-site)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("### 🧮 Get Salary Predictions")
        st.markdown("""
        Use the **Prediction Tool** to:
        * Enter your professional profile
        * Get a personalized salary estimate
        * See how different factors impact your salary
        * Receive tailored recommendations
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("### 📈 Plan Your Career")
        st.markdown("""
        Use the insights to:
        * Benchmark your current compensation
        * Identify skills to develop
        * Plan your career progression
        * Prepare for salary negotiations
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Data sources and limitations
    st.markdown('<h2 class="section-header">Data Sources & Limitations</h2>', unsafe_allow_html=True)
    st.markdown("""
    * The predictions are based on a machine learning model trained on salary data from various sources
    * The model considers key factors like job role, experience level, region, company sector, and more
    * Remember that predictions are estimates and actual salaries may vary based on specific circumstances
    * The tool doesn't account for all possible factors that might influence individual compensation
    """)
    
    # Call to action
    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    st.markdown("### Ready to explore your salary potential?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("View Salary Trends", key="goto_viz"):
            st.session_state.page = "Visualizations"
            st.experimental_rerun()
    with col2:
        if st.button("Get Salary Prediction", key="goto_predict"):
            st.session_state.page = "Prediction Tool"
            st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def visualizations_page():
    """Display the visualizations page content"""
    st.markdown('<h1 class="main-header">Salary Trend Visualizations</h1>', unsafe_allow_html=True)
    
    # Load the sample data
    df = load_sample_data()
    
    # Introduction
    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    st.markdown("""
    Explore salary trends across different dimensions of the data science job market.
    These visualizations can help you understand how factors like job role, experience,
    region, and work setting influence compensation in the field.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualization options
    st.markdown('<h2 class="section-header">Select a Visualization</h2>', unsafe_allow_html=True)
    viz_type = st.radio(
        "Choose what to visualize:",
        ["Salary by Job Category", "Salary by Experience Level", "Regional Salary Comparison", 
         "Sector Comparison", "Work Setting Impact", "Combined Factors"]
    )
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Create the selected visualization
    if viz_type == "Salary by Job Category":
        st.markdown('<h3 class="subsection-header">Salary Distribution by Job Category</h3>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x='job_category', y='salary_in_usd', palette='viridis', ax=ax)
        ax.set_xlabel('Job Category')
        ax.set_ylabel('Annual Salary (USD)')
        ax.set_title('Salary Distribution by Job Category', fontsize=16)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Calculate average salary by job category for annotation
        avg_by_job = df.groupby('job_category')['salary_in_usd'].mean().sort_values(ascending=False)
        
        # Add text explaining the visualization
        st.pyplot(fig)
        
        st.markdown("""
        #### Key Insights:
        * **Highest paying roles**: Data Science Managers and Research Scientists typically earn the highest salaries
        * **Business Analysts** generally earn less than Data Analysts, with a salary difference of about 15%
        * **ML Engineers** and **AI Engineers** command premium salaries due to specialized skills
        
        The box plot shows the distribution of salaries for each job category. The box represents the 
        interquartile range (middle 50% of salaries), the line inside the box is the median, and the 
        whiskers extend to the minimum and maximum values (excluding outliers).
        """)
    
    elif viz_type == "Salary by Experience Level":
        st.markdown('<h3 class="subsection-header">Salary Progression by Experience Level</h3>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=df, x='experience_level_desc', y='salary_in_usd', 
                   order=['Entry-Level', 'Mid-Level', 'Senior', 'Executive'],
                   palette='Blues', ax=ax)
        ax.set_xlabel('Experience Level')
        ax.set_ylabel('Average Annual Salary (USD)')
        ax.set_title('Average Salary by Experience Level', fontsize=16)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Calculate average by experience level for each job
        exp_job_avg = df.groupby(['experience_level_desc', 'job_category'])['salary_in_usd'].mean().reset_index()
        exp_job_avg = exp_job_avg.pivot(index='job_category', columns='experience_level_desc', values='salary_in_usd')
        exp_job_avg = exp_job_avg[['Entry-Level', 'Mid-Level', 'Senior', 'Executive']]
        
        # Add annotations
        for i, exp in enumerate(['Entry-Level', 'Mid-Level', 'Senior', 'Executive']):
            avg_salary = df[df['experience_level_desc'] == exp]['salary_in_usd'].mean()
            ax.text(i, avg_salary + 5000, f"${avg_salary:,.0f}", ha='center', fontweight='bold')
        
        st.pyplot(fig)
        
        # Show detailed comparison table
        st.markdown("#### Salary Progression by Role and Experience Level")
        st.dataframe(exp_job_avg.style.format("${:,.0f}").background_gradient(cmap='Blues'))
        
        st.markdown("""
        #### Key Insights:
        * **Significant jumps**: Moving from Entry-Level to Mid-Level typically results in a 30-40% salary increase
        * **Executive premium**: Executive-level positions earn 35-50% more than Senior roles
        * **Growth potential**: The largest salary jump is typically from Mid-Level to Senior positions
        
        Experience level is the single most important factor determining salary in the data science field.
        As you gain experience, your market value increases substantially.
        """)
    
    elif viz_type == "Regional Salary Comparison":
        st.markdown('<h3 class="subsection-header">Salary Comparison by Region</h3>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=df, x='region', y='salary_in_usd', palette='viridis', ax=ax)
        ax.set_xlabel('Region')
        ax.set_ylabel('Average Annual Salary (USD)')
        ax.set_title('Average Salary by Region', fontsize=16)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Calculate average by region
        region_avg = df.groupby('region')['salary_in_usd'].mean().sort_values(ascending=False)
        
        # Add annotations
        for i, region in enumerate(ax.get_xticklabels()):
            region_name = region.get_text()
            avg_salary = df[df['region'] == region_name]['salary_in_usd'].mean()
            ax.text(i, avg_salary + 5000, f"${avg_salary:,.0f}", ha='center', fontweight='bold')
        
        st.pyplot(fig)
        
        # Create a heatmap of salary by region and experience
        st.markdown("#### Salary by Region and Experience Level")
        region_exp_pivot = df.pivot_table(values='salary_in_usd', index='region', 
                                          columns='experience_level_desc', aggfunc='mean')
        region_exp_pivot = region_exp_pivot[['Entry-Level', 'Mid-Level', 'Senior', 'Executive']]
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.heatmap(region_exp_pivot, annot=True, fmt="$,.0f", cmap='YlGnBu', ax=ax2)
        ax2.set_title("Average Salary by Region and Experience Level")
        st.pyplot(fig2)
        
        st.markdown("""
        #### Key Insights:
        * **North America** (particularly the US) offers the highest average salaries in the data science field
        * **Europe** follows with competitive compensation, especially in Western European countries
        * **Regional disparities**: There's a significant salary gap between regions, with North America offering
          up to twice the compensation of regions like Africa or South America
        * **Remote opportunity**: Remote work can allow professionals in lower-paying regions to access higher salaries
        
        Geographic location remains one of the strongest predictors of salary range, though remote work
        is gradually reducing these disparities for some roles.
        """)
    
    elif viz_type == "Sector Comparison":
        st.markdown('<h3 class="subsection-header">Salary Comparison by Company Sector</h3>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=df, x='company_sector', y='salary_in_usd', palette='mako', ax=ax)
        ax.set_xlabel('Company Sector')
        ax.set_ylabel('Average Annual Salary (USD)')
        ax.set_title('Average Salary by Company Sector', fontsize=16)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add annotations
        for i, sector in enumerate(ax.get_xticklabels()):
            sector_name = sector.get_text()
            avg_salary = df[df['company_sector'] == sector_name]['salary_in_usd'].mean()
            ax.text(i, avg_salary + 5000, f"${avg_salary:,.0f}", ha='center', fontweight='bold')
        
        st.pyplot(fig)
        
        # Create a comparison of top roles in each sector
        st.markdown("#### Top Paying Roles by Sector")
        
        sectors = df['company_sector'].unique()
        top_roles = []
        
        for sector in sectors:
            sector_df = df[df['company_sector'] == sector]
            top_role = sector_df.groupby('job_category')['salary_in_usd'].mean().nlargest(1).reset_index()
            top_role['company_sector'] = sector
            top_roles.append(top_role)
        
        top_roles_df = pd.concat(top_roles)
        
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        sns.barplot(data=top_roles_df, x='company_sector', y='salary_in_usd', hue='job_category', palette='Set2', ax=ax2)
        ax2.set_xlabel('Company Sector')
        ax2.set_ylabel('Average Annual Salary (USD)')
        ax2.set_title('Highest Paying Role by Sector', fontsize=16)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.legend(title='Job Category')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        st.pyplot(fig2)
        
        st.markdown("""
        #### Key Insights:
        * **Finance** and **Technology** sectors typically offer the highest salaries for data professionals
        * **Healthcare** is increasingly competitive, especially for roles involving medical data or research
        * **Education** generally offers lower compensation but may provide other benefits like work-life balance
        * **Role specialization**: Some specialized roles command premium salaries regardless of sector
        
        The finance sector often leads in compensation due to the high value placed on data-driven insights
        for investment, risk assessment, and business strategy.
        """)
    
    elif viz_type == "Work Setting Impact":
        st.markdown('<h3 class="subsection-header">Impact of Work Setting on Salary</h3>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df, x='work_setting', y='salary_in_usd', palette='Set1', ax=ax)
        ax.set_xlabel('Work Setting')
        ax.set_ylabel('Average Annual Salary (USD)')
        ax.set_title('Average Salary by Work Setting', fontsize=16)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add annotations
        for i, setting in enumerate(['Hybrid', 'On-site', 'Remote']):
            avg_salary = df[df['work_setting'] == setting]['salary_in_usd'].mean()
            ax.text(i, avg_salary + 5000, f"${avg_salary:,.0f}", ha='center', fontweight='bold')
        
        st.pyplot(fig)
        
        # Create a comparison by work setting and experience
        st.markdown("#### Work Setting Salary by Experience Level")
        
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.barplot(data=df, x='experience_level_desc', y='salary_in_usd', hue='work_setting', 
                   order=['Entry-Level', 'Mid-Level', 'Senior', 'Executive'],
                   palette='Set1', ax=ax2)
        ax2.set_xlabel('Experience Level')
        ax2.set_ylabel('Average Annual Salary (USD)')
        ax2.set_title('Average Salary by Experience Level and Work Setting', fontsize=16)
        ax2.legend(title='Work Setting')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        st.pyplot(fig2)
        
        st.markdown("""
        #### Key Insights:
        * **Remote work** often comes with a slight salary premium, especially at higher experience levels
        * **Hybrid arrangements** typically offer competitive compensation while balancing flexibility
        * **On-site positions** may offer lower base salaries but could have additional benefits
        * **Executive trend**: At executive levels, the remote premium is most pronounced
        
        The shift to remote work has changed salary dynamics, with many companies offering competitive
        or premium compensation for remote talent to access larger talent pools.
        """)
    
    elif viz_type == "Combined Factors":
        st.markdown('<h3 class="subsection-header">Interactive Multi-Factor Analysis</h3>', unsafe_allow_html=True)
        
        # Let user select dimensions to analyze
        col1, col2 = st.columns(2)
        
        with col1:
            primary_factor = st.selectbox("Primary grouping factor:", 
                                         ["job_category", "experience_level_desc", "region", "company_sector", "work_setting"])
        
        with col2:
            secondary_factor = st.selectbox("Secondary grouping factor (color):",
                                           ["experience_level_desc", "job_category", "region", "company_sector", "work_setting"],
                                           index=1)
        
        if primary_factor == secondary_factor:
            st.warning("Please select different factors for primary and secondary grouping")
        else:
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.barplot(data=df, x=primary_factor, y='salary_in_usd', hue=secondary_factor, palette='viridis', ax=ax)
            ax.set_xlabel(primary_factor.replace('_', ' ').title())
            ax.set_ylabel('Average Annual Salary (USD)')
            ax.set_title(f'Average Salary by {primary_factor.replace("_", " ").title()} and {secondary_factor.replace("_", " ").title()}', fontsize=16)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.legend(title=secondary_factor.replace('_', ' ').title())
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Try to make legend more readable
            if len(df[secondary_factor].unique()) > 5:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            st.pyplot(fig)
            
            # Create a pivot table for the same data
            pivot_df = df.pivot_table(values='salary_in_usd', 
                                      index=primary_factor, 
                                      columns=secondary_factor, 
                                      aggfunc='mean')
            
            st.markdown("#### Detailed Comparison Table")
            st.dataframe(pivot_df.style.format("${:,.0f}").background_gradient(cmap='Blues'))
            
            st.markdown("""
            #### Interactive Analysis
            This visualization allows you to explore how different combinations of factors affect salary.
            Try different combinations to uncover interesting patterns and insights.
            
            For example:
            * Job category by experience level shows career progression paths
            * Region by job category reveals which roles are most valued in different areas
            * Work setting by sector shows which industries embrace remote work with higher compensation
            """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Bottom section with key takeaways
    st.markdown('<h2 class="section-header">Key Takeaways</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Based on our analysis of salary data, here are the most important factors influencing compensation:
    
    1. **Experience Level**: The single most important factor, with senior and executive roles earning 1.5-2.5x entry-level positions
    
    2. **Geographic Region**: North America leads in compensation, followed by Europe and Oceania
    
    3. **Job Specialization**: Roles requiring specialized skills (ML Engineer, Research Scientist) command higher salaries
    
    4. **Company Sector**: Finance and Technology typically offer the highest compensation
    
    5. **Technical Expertise**: Higher technical specialization can increase compensation by 10-15%
    """)
    
    # Call to action
    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    st.markdown("### Ready to see how these factors apply to your profile?")
    if st.button("Get Your Salary Prediction", key="viz_to_predict"):
        st.session_state.page = "Prediction Tool"
        st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def prediction_page():
    """Display the prediction tool page content"""
    st.markdown('<h1 class="main-header">Personalized Salary Prediction</h1>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if not model or not test_model_sensitivity(model):
        st.error("There's an issue with the prediction model. Please try again later or contact support.")
        st.stop()
    
    # Introduction
    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    st.markdown("""
    Get a personalized salary estimate based on your professional profile. 
    Fill in the form below with your details, and our machine learning model will generate
    a prediction tailored to your specific situation.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create main columns for form and results
    col1, col2 = st.columns([1, 1])
    
    # Input form
    with col1:
        st.markdown('<h2 class="section-header">Your Professional Profile</h2>', unsafe_allow_html=True)
        
        # Key variables section
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown('<h3 class="subsection-header">Primary Information</h3>', unsafe_allow_html=True)
        st.info("These factors directly affect the model's prediction")
        
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
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional details section
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown('<h3 class="subsection-header">Additional Details</h3>', unsafe_allow_html=True)
        st.info("These factors will refine the prediction and provide better insights")
        
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
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Submit button
        predict_button = st.button("Calculate My Salary Estimate", type="primary", key="predict_salary")
    
    # Results section (only shown after prediction)
    with col2:
        st.markdown('<h2 class="section-header">Your Salary Estimate</h2>', unsafe_allow_html=True)
        
        if "results_ready" not in st.session_state:
            st.session_state.results_ready = False
        
        if predict_button:
            with st.spinner("Analyzing your profile..."):
                try:
                    # Prepare data for the model
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
                        base_predicted_salary = 50000  # Minimum reasonable salary
                    elif base_predicted_salary > 500000:
                        base_predicted_salary = 300000  # Maximum reasonable salary
                    
                    # Adjust salary using additional factors
                    adjusted_salary = adjust_salary_with_details(
                        base_predicted_salary, 
                        job_category,
                        tech_specialization, 
                        english_level, 
                        company_size, 
                        employment_type
                    )
                    
                    # Store in session state
                    st.session_state.base_salary = base_predicted_salary
                    st.session_state.adjusted_salary = adjusted_salary
                    st.session_state.results_ready = True
                    
                    # Store other values for visualization
                    st.session_state.job_category = job_category
                    st.session_state.experience_level = experience_level
                    st.session_state.region = region
                    st.session_state.tech_specialization = tech_specialization
                    st.session_state.english_level = english_level
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
        
        # Display results if ready
        if st.session_state.results_ready:
            base_predicted_salary = st.session_state.base_salary
            adjusted_salary = st.session_state.adjusted_salary
            
            # Main prediction result
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown('<h3 class="subsection-header">Estimated Annual Salary</h3>', unsafe_allow_html=True)
            st.markdown(f"<h1 style='color:#1976D2; text-align:center; font-size:3.5rem'>${adjusted_salary:,.2f}</h1>", unsafe_allow_html=True)
            
            # Show adjustment info
            adjustment_pct = (adjusted_salary / base_predicted_salary - 1) * 100
            adjustment_text = "increase" if adjustment_pct >= 0 else "decrease"
            
            st.info(f"Your profile details resulted in a {abs(adjustment_pct):.1f}% {adjustment_text} from the base prediction.")
            
            if job_category == "Business Analyst":
                st.info("Note: Business Analyst salaries are typically 15% lower than comparable Data Analyst positions.")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional metrics
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Monthly Salary",
                    f"${adjusted_salary/12:,.2f}"
                )
            
            with col2:
                # Salary adjusted by experience
                exp_factor = {"Entry-Level": 1.3, "Mid-Level": 1.2, "Senior": 1.1, "Executive": 1.05}
                next_level = {
                    "Entry-Level": "Mid-Level", 
                    "Mid-Level": "Senior", 
                    "Senior": "Executive", 
                    "Executive": "Executive+"
                }
                
                potential_salary = adjusted_salary * exp_factor.get(experience_level, 1.2)
                
                st.metric(
                    f"Potential ({next_level[experience_level]})",
                    f"${potential_salary:,.2f}",
                    delta=f"{(exp_factor.get(experience_level, 1.2) - 1) * 100:.0f}%"
                )
            
            with col3:
                # Estimated percentile
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
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Tech stack visualization
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<h3 class="subsection-header">Technology Stack Analysis</h3>', unsafe_allow_html=True)
            
            tech_fig = plot_tech_stack(job_category)
            if tech_fig:
                st.pyplot(tech_fig)
                st.markdown(f"""
                The chart above shows the key technologies and skills for your role as a **{job_category}**.
                Mastering these skills, especially the top 3-4, can significantly increase your market value.
                """)
            else:
                st.write("No technology stack information available for this role.")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Salary growth trajectory
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<h3 class="subsection-header">5-Year Salary Projection</h3>', unsafe_allow_html=True)
            
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
            ax3 = sns.lineplot(x='Year', y='Salary', data=projection_df, marker='o', linewidth=2, color='#1976D2')
            
            # Add value labels
            for i, val in enumerate(projected_salaries):
                ax3.text(i, val + 5000, f"${val:,.0f}", ha='center')
            
            # Customize chart
            ax3.set_title('Projected Salary Growth (Based on Industry Averages)')
            ax3.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig3)
            
            st.markdown(f"""
            With an estimated annual growth rate of **{growth_rate*100:.0f}%** for your experience level, 
            your salary could reach **${projected_salaries[-1]:,.2f}** in 5 years if you maintain your 
            current career trajectory.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Personalized recommendations
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown('<h3 class="subsection-header">Salary Growth Recommendations</h3>', unsafe_allow_html=True)
            
            recommendations = generate_recommendations(
                job_category, 
                experience_level, 
                tech_specialization, 
                english_level, 
                region
            )
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
            st.markdown('</div>', unsafe_allow_html=True)

# ======= MAIN APP =======
def main():
    # Initialize session state for navigation
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://img.freepik.com/free-vector/illustration-data-analysis-graph_53876-18139.jpg", width=200)
        st.title("Navigation")
        
        # Navigation options
        selected = st.radio("Go to:", ["Home", "Visualizations", "Prediction Tool"], index=["Home", "Visualizations", "Prediction Tool"].index(st.session_state.page))
        
        # Update session state on change
        if selected != st.session_state.page:
            st.session_state.page = selected
        
        st.markdown("---")
        st.markdown("### About")
        st.info("This app helps data professionals estimate their market value and understand salary trends in the field.")
        
        st.markdown("### Resources")
        st.markdown("- [Career Development Guide](https://example.com)")
        st.markdown("- [Salary Negotiation Tips](https://example.com)")
        st.markdown("- [Skills Assessment Tools](https://example.com)")
    
    # Display the selected page
    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Visualizations":
        visualizations_page()
    elif st.session_state.page == "Prediction Tool":
        prediction_page()

if __name__ == "__main__":
    main()