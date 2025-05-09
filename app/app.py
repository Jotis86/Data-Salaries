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
    """Load the prediction model with improved error handling"""
    # Try multiple approaches to locate the model file
    possible_paths = [
        "simple_salary_model.pkl",  # Direct in current directory
        os.path.join(os.path.dirname(__file__), "simple_salary_model.pkl"),  # Using __file__
        os.path.join(".", "simple_salary_model.pkl"),  # Explicit current directory
        os.path.abspath("simple_salary_model.pkl")  # Absolute path
    ]
    
    # Try each path
    for path in possible_paths:
        try:
            with open(path, 'rb') as file:
                model = pickle.load(file)
            return model
        except Exception:
            continue
    
    # If we get here, all attempts failed
    st.error("Could not load the model file from any location.")
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
    # Custom CSS for enhanced styling with theme-responsive colors
    st.markdown("""
    <style>
        /* Hero section styling */
        .hero-container {
            background: linear-gradient(135deg, #1E88E5 0%, #0D47A1 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        .hero-title {
            font-size: 2.8rem !important;
            font-weight: 800;
            margin-bottom: 1rem;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        .hero-text {
            font-size: 1.2rem;
            opacity: 0.9;
            max-width: 800px;
            line-height: 1.6;
        }
        
        /* Section headers - theme responsive */
        .custom-header {
            font-size: 1.8rem !important;
            font-weight: 700;
            margin-top: 2rem;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #E5E7EB;
        }
        
        /* Light theme - dark text for headers */
        .light-mode .custom-header {
            color: #1E3A8A;
        }
        
        /* Dark theme - lighter text for headers */
        .dark-mode .custom-header {
            color: #90CAF9;
        }
        
        /* Card styling - theme responsive */
        .card-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 2rem;
        }
        
        /* Light theme card */
        .light-mode .info-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            padding: 1.5rem;
            transition: transform 0.2s, box-shadow 0.2s;
            flex: 1;
            min-width: 250px;
            border-top: 5px solid #3B82F6;
        }
        
        /* Dark theme card */
        .dark-mode .info-card {
            background-color: #1E1E1E;
            border-radius: 8px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
            padding: 1.5rem;
            transition: transform 0.2s, box-shadow 0.2s;
            flex: 1;
            min-width: 250px;
            border-top: 5px solid #3B82F6;
        }
        
        .info-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
        }
        
        .card-icon {
            font-size: 2rem;
            margin-bottom: 1rem;
            color: #3B82F6;
        }
        
        /* Card title - theme responsive */
        .light-mode .card-title {
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 1rem;
            color: #1E3A8A;
        }
        
        .dark-mode .card-title {
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 1rem;
            color: #90CAF9;
        }
        
        /* List styling - theme responsive */
        .styled-list {
            list-style-type: none;
            padding-left: 0;
        }
        
        .light-mode .styled-list li {
            padding-left: 1.5rem;
            position: relative;
            margin-bottom: 0.7rem;
            line-height: 1.5;
            color: #333333;
        }
        
        .dark-mode .styled-list li {
            padding-left: 1.5rem;
            position: relative;
            margin-bottom: 0.7rem;
            line-height: 1.5;
            color: #e0e0e0;
        }
        
        .styled-list li:before {
            content: "•";
            color: #3B82F6;
            font-weight: bold;
            position: absolute;
            left: 0;
        }
        
        /* Limitations section - theme responsive */
        .light-mode .limitations-container {
            background-color: #E8EAF6;
            border-radius: 8px;
            padding: 1.5rem;
            border-left: 4px solid #5C6BC0;
            margin: 1.5rem 0;
        }
        
        .dark-mode .limitations-container {
            background-color: #1A237E;
            background-opacity: 0.2;
            border-radius: 8px;
            padding: 1.5rem;
            border-left: 4px solid #7986CB;
            margin: 1.5rem 0;
        }
        
        /* Limitations title - theme responsive */
        .light-mode .limitations-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #303F9F;
        }
        
        .dark-mode .limitations-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #C5CAE9;
        }
        
        /* Default text colors - theme responsive */
        .light-mode p, .light-mode li {
            color: #333333;
        }
        
        .dark-mode p, .dark-mode li {
            color: #e0e0e0;
        }
        
        /* Only hero section text should be white in both themes */
        .hero-container h1, .hero-container p {
            color: white !important;
        }
        
        /* Theme detection */
        .stApp[data-theme="light"] .light-mode { display: block; }
        .stApp[data-theme="light"] .dark-mode { display: none; }
        .stApp[data-theme="dark"] .light-mode { display: none; }
        .stApp[data-theme="dark"] .dark-mode { display: block; }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero section (same for both themes)
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">Data Science Salary Predictor</h1>
        <p class="hero-text">
            Welcome to the Data Science Salary Predictor! This application helps data professionals 
            estimate their market value based on key factors that influence salaries in the 
            data science and analytics field.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # About the app section - for both light and dark themes
    st.markdown('<h2 class="custom-header">About This App</h2>', unsafe_allow_html=True)
    
    # Light theme version
    st.markdown("""
    <div class="light-mode">
        <div class="info-card">
            <div class="card-title">Why Use This Tool?</div>
            <ul class="styled-list">
                <li><strong>Estimate your market value</strong> based on role, experience, location, and other factors</li>
                <li><strong>Visualize salary trends</strong> across different dimensions of the data science job market</li>
                <li><strong>Discover insights</strong> that can help negotiate better compensation</li>
                <li><strong>Identify strategies</strong> to increase earning potential in the data field</li>
            </ul>
            <p>The app uses machine learning to predict salaries based on real-world data from thousands
            of data professionals across different regions, companies, and specializations.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Dark theme version
    st.markdown("""
    <div class="dark-mode">
        <div class="info-card">
            <div class="card-title">Why Use This Tool?</div>
            <ul class="styled-list">
                <li><strong>Estimate your market value</strong> based on role, experience, location, and other factors</li>
                <li><strong>Visualize salary trends</strong> across different dimensions of the data science job market</li>
                <li><strong>Discover insights</strong> that can help negotiate better compensation</li>
                <li><strong>Identify strategies</strong> to increase earning potential in the data field</li>
            </ul>
            <p>The app uses machine learning to predict salaries based on real-world data from thousands
            of data professionals across different regions, companies, and specializations.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # How to use section - for both themes
    st.markdown('<h2 class="custom-header">How to Use This App</h2>', unsafe_allow_html=True)
    
    # Light theme version
    st.markdown("""
    <div class="light-mode">
        <div class="info-card">
            <div class="card-title">Application Features</div>
            <ul class="styled-list">
                <li><strong>Visualizations Section:</strong> Explore salary trends across job categories, experience levels, geographic regions, industry sectors, and work settings</li>
                <li><strong>Prediction Tool:</strong> Enter your professional profile to get a personalized salary estimate and see how different factors impact your compensation</li>
                <li><strong>Career Planning:</strong> Use insights to benchmark your current compensation, identify skills to develop, and prepare for salary negotiations</li>
                <li><strong>Detailed Analysis:</strong> View comprehensive charts and analytics about the data science job market</li>
            </ul>
            <p>Navigate between sections using the sidebar menu to make the most of all available features.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Dark theme version
    st.markdown("""
    <div class="dark-mode">
        <div class="info-card">
            <div class="card-title">Application Features</div>
            <ul class="styled-list">
                <li><strong>Visualizations Section:</strong> Explore salary trends across job categories, experience levels, geographic regions, industry sectors, and work settings</li>
                <li><strong>Prediction Tool:</strong> Enter your professional profile to get a personalized salary estimate and see how different factors impact your compensation</li>
                <li><strong>Career Planning:</strong> Use insights to benchmark your current compensation, identify skills to develop, and prepare for salary negotiations</li>
                <li><strong>Detailed Analysis:</strong> View comprehensive charts and analytics about the data science job market</li>
            </ul>
            <p>Navigate between sections using the sidebar menu to make the most of all available features.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Data sources and limitations - for both themes
    st.markdown('<h2 class="custom-header">Data Sources & Limitations</h2>', unsafe_allow_html=True)
    
    # Light theme version
    st.markdown("""
    <div class="light-mode">
        <div class="limitations-container">
            <h3 class="limitations-title">Important Notes About This Tool</h3>
            <ul class="styled-list">
                <li>The predictions are based on a machine learning model trained on salary data from various sources</li>
                <li>The model considers key factors like job role, experience level, region, company sector, and more</li>
                <li>Remember that predictions are estimates and actual salaries may vary based on specific circumstances</li>
                <li>The tool doesn't account for all possible factors that might influence individual compensation</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Dark theme version
    st.markdown("""
    <div class="dark-mode">
        <div class="limitations-container">
            <h3 class="limitations-title">Important Notes About This Tool</h3>
            <ul class="styled-list">
                <li>The predictions are based on a machine learning model trained on salary data from various sources</li>
                <li>The model considers key factors like job role, experience level, region, company sector, and more</li>
                <li>Remember that predictions are estimates and actual salaries may vary based on specific circumstances</li>
                <li>The tool doesn't account for all possible factors that might influence individual compensation</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    

def visualizations_page():
    """Display the visualizations page content"""
    st.markdown('<h1 class="main-header">Data Science Salary Insights</h1>', unsafe_allow_html=True)
    
    # Load the real data instead of sample data
    @st.cache_data
    def load_real_data():
        try:
            # Try to load the data file from the same directory as the app
            data_path = os.path.join(os.path.dirname(__file__), "clean_salary_data.csv")
            return pd.read_csv(data_path)
        except:
            st.error("Could not load clean_salary_data.csv. Using sample data instead.")
            return load_sample_data()
    
    df = load_real_data()
    
    # Introduction with animated number counter
    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Data Professionals", f"{len(df):,}", delta="analyzed")
    with col2:
        st.metric("Average Salary", f"${df['salary_in_usd'].mean():,.0f}", delta=f"{df['salary_in_usd'].mean()/1000:.0f}K")
    with col3:
        st.metric("Salary Range", f"${df['salary_in_usd'].max():,.0f}", delta=f"{df['salary_in_usd'].max() - df['salary_in_usd'].min():,.0f} spread")
    
    st.markdown("""
    Explore comprehensive salary insights across the data science ecosystem. Discover how factors like job role, 
    experience level, location, and skill specialization influence compensation in today's market.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualization selection with improved UI
    st.markdown('<h2 class="section-header">Select Visualization</h2>', unsafe_allow_html=True)
    
    # Create a more visual selector with icons
    viz_cols = st.columns(4)
    with viz_cols[0]:
        tab1 = st.button("📊 Salary Landscape", use_container_width=True)
    with viz_cols[1]:
        tab2 = st.button("🚀 Career Progression", use_container_width=True)
    with viz_cols[2]:
        tab3 = st.button("🌎 Geographic Impact", use_container_width=True)
    with viz_cols[3]:
        tab4 = st.button("🔬 Specialization Effect", use_container_width=True)
    
    viz_cols2 = st.columns(4)
    with viz_cols2[0]:
        tab5 = st.button("⚙️ AI Impact Analysis", use_container_width=True)
    with viz_cols2[1]:
        tab6 = st.button("💼 Work Setting Trends", use_container_width=True)
    with viz_cols2[2]:
        tab7 = st.button("⚖️ Work-Life Balance", use_container_width=True)
    with viz_cols2[3]:
        tab8 = st.button("🔮 Future Outlook", use_container_width=True)
    
    # Tab handling with session state
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Salary Landscape"
    
    if tab1: st.session_state.active_tab = "Salary Landscape"
    elif tab2: st.session_state.active_tab = "Career Progression"
    elif tab3: st.session_state.active_tab = "Geographic Impact"
    elif tab4: st.session_state.active_tab = "Specialization Effect"
    elif tab5: st.session_state.active_tab = "AI Impact Analysis"
    elif tab6: st.session_state.active_tab = "Work Setting Trends"
    elif tab7: st.session_state.active_tab = "Work-Life Balance"
    elif tab8: st.session_state.active_tab = "Future Outlook"
    
    # Display which tab is active
    st.markdown(f"<h3 class='subsection-header'>{st.session_state.active_tab}</h3>", unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # 1. SALARY LANDSCAPE
    if st.session_state.active_tab == "Salary Landscape":
        # Enhanced horizontal lollipop chart for job categories using full width
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Calculate key statistics
        job_stats = df.groupby('job_category')['salary_in_usd'].agg(['mean', 'median', 'std']).reset_index()
        job_stats = job_stats.sort_values('median', ascending=True)
        
        # Plot horizontal lines (stems)
        ax.hlines(y=job_stats['job_category'], xmin=0, xmax=job_stats['median'], 
                color='skyblue', alpha=0.7, linewidth=5)
        
        # Plot median points
        scatter = ax.scatter(job_stats['median'], job_stats['job_category'], s=job_stats['mean']/500, 
                color='#3498db', alpha=0.8, zorder=10)
        
        # Add smaller mean indicators
        ax.scatter(job_stats['mean'], job_stats['job_category'], s=60, 
                marker='D', color='#e74c3c', alpha=0.8, zorder=5)
        
        # Add salary labels
        #for i, row in job_stats.iterrows():
            #ax.text(row['median'] + 5000, i, f"${row['median']:,.0f}", 
                    #va='center', ha='left', fontweight='bold', color='white')
            
        # Customize graph
        ax.set_xlabel('Median Annual Salary (USD)', fontsize=12, fontweight='bold')
        ax.set_title('Salary Distribution by Role', fontsize=18, fontweight='bold', pad=20)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set background color to match Streamlit's darker theme
        fig.patch.set_facecolor('#2c3e50')
        ax.set_facecolor('#2c3e50')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#3498db')
        
        # Add legend proxy artists
        median_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=10, label='Median Salary')
        mean_marker = plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#e74c3c', markersize=8, label='Mean Salary')
        count_info = plt.Line2D([0], [0], marker='o', color='w', alpha=0, markersize=0, 
                            label=f'Total roles analyzed: {len(df)}')
        ax.legend(handles=[median_marker, mean_marker, count_info], loc='lower right', 
                frameon=True, facecolor='#2c3e50', edgecolor='#3498db')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Key metrics table in a single column
        job_metrics = df.groupby('job_category')['salary_in_usd'].agg(['count', 'mean', 'median', 'min', 'max']).reset_index()
        job_metrics = job_metrics.sort_values('median', ascending=False)
        
        # Rename columns for display
        job_metrics.columns = ['Job Category', 'Count', 'Mean', 'Median', 'Minimum', 'Maximum']
        
        # Format metrics as currency
        for col in ['Mean', 'Median', 'Minimum', 'Maximum']:
            job_metrics[col] = job_metrics[col].apply(lambda x: f"${x:,.0f}")
        
        # Create a styled table
        st.markdown("### Salary Statistics by Role")
        st.dataframe(job_metrics.set_index('Job Category'), use_container_width=True)
        
        st.markdown("""
        #### Key Insights
        * **Data Science Leaders** (Managers, Architects, and Research Scientists) command the highest salaries
        * **Technical specialization** is rewarded: ML Engineers earn more than general Data Scientists
        * **Business-focused roles** (Business Analysts) typically earn less than technical roles
        * **Considerable variability** exists within each role, indicating other factors impact compensation
        """)
    
    # 2. CAREER PROGRESSION
    elif st.session_state.active_tab == "Career Progression":
        # Create a simpler but visually appealing chart showing salary growth by experience level
        
        # Prepare the data for a grouped bar chart
        exp_level_order = ['Entry-Level', 'Mid-Level', 'Senior', 'Executive']
        
        # Calculate median salaries by job category and experience level for all jobs
        all_exp_data = df.groupby(['job_category', 'experience_level_desc'])['salary_in_usd'].median().reset_index()
        
        # Get top 12 job categories by median salary
        top_jobs = df.groupby('job_category')['salary_in_usd'].median().nlargest(12).index.tolist()
        
        # Split into two groups for better visibility
        first_group = top_jobs[:6]
        second_group = top_jobs[6:]
        
        # Function to create chart for a group of job categories
        def create_career_chart(job_group, chart_title):
            # Filter data for this group
            filtered_df = df[df['job_category'].isin(job_group)]
            exp_data = filtered_df.groupby(['job_category', 'experience_level_desc'])['salary_in_usd'].median().reset_index()
            
            # Create a more intuitive and cleaner bar chart
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Set up the positions for grouped bars
            job_categories = sorted(job_group)
            x = np.arange(len(job_categories))
            width = 0.2  # width of bars
            
            # Define an attractive color palette
            colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
            
            # Create bars for each experience level
            for i, level in enumerate(exp_level_order):
                level_data = exp_data[exp_data['experience_level_desc'] == level]
                
                # Create a dictionary mapping job categories to salaries
                salary_dict = dict(zip(level_data['job_category'], level_data['salary_in_usd']))
                
                # Get salaries in the correct order
                values = [salary_dict.get(job, 0) for job in job_categories]
                
                # Create the bars
                bars = ax.bar(x + (i - 1.5) * width, values, width, label=level, color=colors[i], alpha=0.85)
            
            # Add growth arrows and percentages between Executive and Entry-Level for visualization
            for j, job in enumerate(job_categories):
                job_data = exp_data[exp_data['job_category'] == job]
                if len(job_data) >= 2:
                    entry_salary = job_data[job_data['experience_level_desc'] == 'Entry-Level']['salary_in_usd'].values
                    exec_salary = job_data[job_data['experience_level_desc'] == 'Executive']['salary_in_usd'].values
                    
                    if len(entry_salary) > 0 and len(exec_salary) > 0:
                        growth_pct = (exec_salary[0] / entry_salary[0] - 1) * 100
                        
                        # Add a vertical arrow showing growth
                        ax.annotate(
                            f"+{growth_pct:.0f}%", 
                            xy=(j, entry_salary[0] + (exec_salary[0] - entry_salary[0])/2),
                            xytext=(j + 0.25, entry_salary[0] + (exec_salary[0] - entry_salary[0])/2),
                            arrowprops=dict(arrowstyle='<->', color='#f1c40f', lw=2),
                            fontsize=10, fontweight='bold', color='#f1c40f'
                        )
            
            # Customize the chart
            ax.set_xticks(x)
            ax.set_xticklabels(job_categories, rotation=30, ha='right')
            ax.set_ylabel('Median Salary (USD)', fontsize=14, fontweight='bold')
            ax.set_title(chart_title, fontsize=20, fontweight='bold', pad=20)
            
            # Add a legend with clear labels
            ax.legend(
                title='Experience Level', 
                title_fontsize=12, 
                fontsize=10, 
                loc='upper left', 
                frameon=True, 
                facecolor='#2c3e50', 
                edgecolor='#3498db'
            )
            
            # Set background color and style
            fig.patch.set_facecolor('#2c3e50')
            ax.set_facecolor('#2c3e50')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            for spine in ax.spines.values():
                spine.set_color('#3498db')
            
            plt.tight_layout()
            return fig
        
        # Create and display the first chart (top 6 roles)
        st.markdown("### Top Paying Roles")
        fig1 = create_career_chart(first_group, 'Career Progression: Top 6 Paying Roles')
        st.pyplot(fig1)
        
        # Create and display the second chart (next 6 roles)
        st.markdown("### More Popular Roles")
        fig2 = create_career_chart(second_group, 'Career Progression: Additional Roles')
        st.pyplot(fig2)
        
        # Create a simple table showing the growth multiplier
        st.markdown("### Salary Growth Multipliers")
        
        # Calculate growth multipliers for all job categories
        multiplier_data = df.groupby(['job_category', 'experience_level_desc'])['salary_in_usd'].median().reset_index()
        multiplier_pivot = multiplier_data.pivot(index='job_category', columns='experience_level_desc', values='salary_in_usd')
        
        # Ensure correct column order
        if all(col in multiplier_pivot.columns for col in exp_level_order):
            multiplier_pivot = multiplier_pivot[exp_level_order]
        
        # Calculate multipliers based on Entry-Level
        multiplier_result = pd.DataFrame(index=multiplier_pivot.index)
        
        for level in exp_level_order:
            if level in multiplier_pivot.columns:
                if 'Entry-Level' in multiplier_pivot.columns:
                    multiplier_result[f"{level} Multiplier"] = multiplier_pivot[level] / multiplier_pivot['Entry-Level']
                else:
                    multiplier_result[f"{level} Multiplier"] = multiplier_pivot[level] / multiplier_pivot.iloc[:, 0]
        
        # Format multipliers as nicely formatted values
        formatted_result = multiplier_result.applymap(lambda x: f"{x:.1f}x")
        
        # Display as a styled table
        st.dataframe(formatted_result, use_container_width=True)
        
        st.markdown("""
        #### Key Insights
        * **Career Ladder Value**: Moving from Entry-Level to Executive typically results in a 2.5-3.5x salary increase
        * **Mid-Career Jump**: The largest percentage increase usually occurs when moving from Mid-Level to Senior roles
        * **Management Premium**: Data Science Manager and Director roles show the highest executive-level compensation
        * **Technical Growth**: Even technical individual contributor roles can see substantial growth with experience
        """)
    
    # 3. GEOGRAPHIC IMPACT
    elif st.session_state.active_tab == "Geographic Impact":
        # Create a horizontal bar chart showing salary by region
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate statistics by region
        region_stats = df.groupby('region')['salary_in_usd'].agg(['count', 'mean', 'median']).reset_index()
        region_stats = region_stats.sort_values('median', ascending=True)
        
        # Create custom colormap based on salary values
        norm = plt.Normalize(region_stats['median'].min(), region_stats['median'].max())
        colors = plt.cm.plasma(norm(region_stats['median']))
        
        # Create horizontal bars
        bars = ax.barh(region_stats['region'], region_stats['median'], 
                height=0.6, color=colors, alpha=0.8, edgecolor='none')
        
        # Add count indicators as scatter points
        sizes = region_stats['count'] / region_stats['count'].max() * 1000
        ax.scatter(region_stats['median'] * 0.1, region_stats['region'], s=sizes, 
                  color='white', alpha=0.5, zorder=10)
        
        # Add salary labels
        for i, bar in enumerate(bars):
            ax.text(bar.get_width() + 5000, bar.get_y() + bar.get_height()/2, 
                   f"${region_stats.iloc[i]['median']:,.0f}", 
                   va='center', ha='left', fontweight='bold', color='white')
            
            # Add count labels near the scatter points
            ax.text(region_stats.iloc[i]['median'] * 0.1, i, 
                   f"{region_stats.iloc[i]['count']} jobs", 
                   va='center', ha='center', fontsize=8, 
                   fontweight='bold', color='black', zorder=11)
        
        # Customize graph
        ax.set_xlabel('Median Annual Salary (USD)', fontsize=12, fontweight='bold')
        ax.set_title('Global Salary Landscape by Region', fontsize=18, fontweight='bold', pad=20)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set background color to match Streamlit's darker theme
        fig.patch.set_facecolor('#2c3e50')
        ax.set_facecolor('#2c3e50')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#3498db')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Create a heatmap showing the cost-adjusted salaries
        # Calculate cost-adjusted values
        region_data = df.groupby(['region', 'experience_level_desc'])['salary_in_usd'].median().reset_index()
        region_pivot = region_data.pivot(index='region', columns='experience_level_desc', values='salary_in_usd')
        
        # Ensure column order
        if all(col in region_pivot.columns for col in ['Entry-Level', 'Mid-Level', 'Senior', 'Executive']):
            region_pivot = region_pivot[['Entry-Level', 'Mid-Level', 'Senior', 'Executive']]
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(region_pivot, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=.5, ax=ax)
        
        # Customize heatmap
        ax.set_title('Regional Salary by Experience Level', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Experience Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Region', fontsize=12, fontweight='bold')
        
        # Set background color
        fig.patch.set_facecolor('#2c3e50')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white') 
        ax.title.set_color('white')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Cost of living adjustment visualization
        if 'cost_of_living_index' in df.columns:
            # Calculate adjusted salaries
            adjustment_data = df.groupby('region')[['salary_in_usd', 'cost_of_living_index']].median().reset_index()
            adjustment_data['adjusted_salary'] = adjustment_data['salary_in_usd'] / adjustment_data['cost_of_living_index']
            adjustment_data = adjustment_data.sort_values('adjusted_salary', ascending=False)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = np.arange(len(adjustment_data))
            width = 0.4
            
            # Plot bars
            raw_bars = ax.bar(x - width/2, adjustment_data['salary_in_usd'], width, label='Raw Salary', color='#3498db')
            adj_bars = ax.bar(x + width/2, adjustment_data['adjusted_salary']*100, width, label='Cost-Adjusted (× 100)', color='#e74c3c')
            
            # Add labels
            ax.set_xticks(x)
            ax.set_xticklabels(adjustment_data['region'], rotation=45, ha='right')
            ax.set_xlabel('Region', fontsize=12, fontweight='bold')
            ax.set_ylabel('Salary (USD)', fontsize=12, fontweight='bold')
            ax.set_title('Cost of Living Adjusted Salaries by Region', fontsize=16, fontweight='bold', pad=20)
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            ax.legend()
            
            # Set background color to match Streamlit's darker theme
            fig.patch.set_facecolor('#2c3e50')
            ax.set_facecolor('#2c3e50')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#3498db')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("""
        #### Key Insights
        * **North America** continues to lead in absolute salary figures, especially at executive levels
        * **Oceania** shows competitive compensation when adjusted for cost of living
        * **Remote opportunity**: Remote work allows professionals in lower-paying regions to access global compensation
        * **Regional arbitrage**: Working remotely for North American companies while living in lower-cost regions provides maximum financial benefit
        """)
    
    # 4. SPECIALIZATION EFFECT
    elif st.session_state.active_tab == "Specialization Effect":
        if 'tech_specialization' in df.columns:
            # Create a scatter plot showing relationship between tech specialization and salary
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Prepare data
            scatter_data = df.groupby('job_category').agg({
                'tech_specialization': 'mean',
                'salary_in_usd': 'median'
            }).reset_index()
            
            # Calculate demand index if it doesn't exist
            if 'demand_index' not in scatter_data.columns:
                # Create simplified demand index based on job count
                job_counts = df['job_category'].value_counts().reset_index()
                job_counts.columns = ['job_category', 'count']
                scatter_data = scatter_data.merge(job_counts, on='job_category')
                scatter_data['demand_index'] = scatter_data['count'] / scatter_data['count'].max() * 10
            
            # Create scatter plot with sized and colored points
            scatter = ax.scatter(scatter_data['tech_specialization'], scatter_data['salary_in_usd'], 
                            s=scatter_data['demand_index']*40, 
                            c=scatter_data['demand_index'], cmap='viridis',
                            alpha=0.7, edgecolors='white', linewidths=1)
            
            # Add job category labels
            for i, row in scatter_data.iterrows():
                ax.annotate(row['job_category'], 
                        (row['tech_specialization']+0.05, row['salary_in_usd']), 
                        fontsize=9, color='white')
            
            # Add trend line
            z = np.polyfit(scatter_data['tech_specialization'], scatter_data['salary_in_usd'], 1)
            p = np.poly1d(z)
            ax.plot(scatter_data['tech_specialization'], p(scatter_data['tech_specialization']), 
                    linestyle='--', color='#FF5555', alpha=0.8, linewidth=2)
            
            # Add legend for the size
            handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=4)
            legend = ax.legend(handles, labels, loc="upper left", title="Demand Index")
            plt.setp(legend.get_title(), color='white')
            
            # Customize graph
            ax.set_title('Relationship Between Technical Specialization and Salary', fontsize=18, fontweight='bold', pad=20)
            ax.set_xlabel('Technical Specialization Level', fontsize=14, fontweight='bold')
            ax.set_ylabel('Median Salary (USD)', fontsize=14, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Demand Index', fontsize=12, fontweight='bold')
            cbar.ax.yaxis.label.set_color('white')
            cbar.ax.tick_params(colors='white')
            
            # Set background color to match Streamlit's darker theme
            fig.patch.set_facecolor('#2c3e50')
            ax.set_facecolor('#2c3e50')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#3498db')
            
            # Add explanatory text
            ax.text(
                0.5, -0.15, 
                "This chart shows how technical specialization relates to salary across different roles.\nThe size and color of the dots indicate the relative demand for each job category.",
                transform=ax.transAxes, 
                ha='center', 
                fontsize=11, 
                color='white', 
                alpha=0.8
            )
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Create a tech stack specialization visualization
            if 'tech_stack' in df.columns:
                # Get the most common tech stacks
                all_tech = []
                for techs in df['tech_stack'].str.split(','):
                    if isinstance(techs, list):
                        all_tech.extend([t.strip() for t in techs])
                
                tech_counts = pd.Series(all_tech).value_counts().head(12)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Create horizontal bar chart for tech skills
                bars = ax.barh(tech_counts.index, tech_counts.values, 
                            color=plt.cm.viridis(np.linspace(0, 1, len(tech_counts))), 
                            alpha=0.8, edgecolor='none')
                
                # Add value annotations
                for i, bar in enumerate(bars):
                    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                        f"{tech_counts.values[i]}", 
                        va='center', fontweight='bold', color='white')
                
                # Customize graph
                ax.set_xlabel('Frequency in Job Profiles', fontsize=12, fontweight='bold')
                ax.set_title('Most In-Demand Technical Skills', fontsize=18, fontweight='bold', pad=20)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='x', linestyle='--', alpha=0.3)
                
                # Set background color to match Streamlit's darker theme
                fig.patch.set_facecolor('#2c3e50')
                ax.set_facecolor('#2c3e50')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.title.set_color('white')
                for spine in ax.spines.values():
                    spine.set_color('#3498db')
                
                plt.tight_layout()
                st.pyplot(fig)
        
        st.markdown("""
        #### Key Insights
        * **Specialization and salary**: There's a clear correlation between technical specialization level and salary
        * **High-demand roles**: Roles with higher technical specialization tend to have higher demand
        * **Sweet spot**: Roles that combine technical specialization with management skills (like directors) achieve maximum salary benefit
        * **Key technologies**: Python, SQL, and Cloud platforms are the most valued and in-demand technical skills
        * **Compound effect**: Higher technical specialization attracts greater demand, which further drives up salaries
        """)
    
    # 5. AI IMPACT ANALYSIS
    elif st.session_state.active_tab == "AI Impact Analysis":
        if 'ai_relationship' in df.columns and 'automation_risk' in df.columns:
            # Create visualization showing AI impact on roles and salaries
            # First, analyse by AI relationship
            ai_impact = df.groupby('ai_relationship')['salary_in_usd'].agg(['count', 'mean', 'median']).reset_index()
            ai_impact = ai_impact.sort_values('median', ascending=False)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create bars with custom colors
            bars = ax.bar(ai_impact['ai_relationship'], ai_impact['median'], 
                        color=plt.cm.plasma(np.linspace(0, 1, len(ai_impact))), 
                        alpha=0.8, width=0.6)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5000,
                    f'${height:,.0f}',
                    ha='center', va='bottom', fontweight='bold', color='white')
                
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right', fontweight='bold')
            
            # Customize graph
            ax.set_xlabel('Relationship to AI', fontsize=12, fontweight='bold')
            ax.set_ylabel('Median Salary (USD)', fontsize=12, fontweight='bold')
            ax.set_title('Impact of AI Relationship on Compensation', fontsize=18, fontweight='bold', pad=20)
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Set background color to match Streamlit's darker theme
            fig.patch.set_facecolor('#2c3e50')
            ax.set_facecolor('#2c3e50')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#3498db')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Create a demand vs automation risk scatter plot (replacing the old horizontal bars)
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Group by job category
            bubble_data = df.groupby('job_category').agg({
                'demand_index': 'mean',
                'automation_risk': 'mean',
                'salary_in_usd': 'median',
                'tech_specialization': 'mean'
            }).reset_index()
            
            # Create scatter with bubbles
            scatter = ax.scatter(
                bubble_data['demand_index'], 
                bubble_data['automation_risk'],
                s=bubble_data['salary_in_usd'] / 1000,
                c=bubble_data['tech_specialization'],
                cmap='viridis',
                alpha=0.7,
                edgecolors='white',
                linewidth=1
            )
            
            # Add bubble labels
            for i, row in bubble_data.iterrows():
                ax.annotate(
                    row['job_category'], 
                    (row['demand_index'], row['automation_risk']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    color='white',
                    bbox=dict(boxstyle="round,pad=0.3", fc="#2c3e50", ec="#3498db", alpha=0.7)
                )
            
            # Divide chart into quadrants
            x_middle = bubble_data['demand_index'].median()
            y_middle = bubble_data['automation_risk'].median()
            
            ax.axhline(y=y_middle, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=x_middle, color='gray', linestyle='--', alpha=0.5)
            
            # Label quadrants
            ax.text(
                bubble_data['demand_index'].min() + 0.5, 
                bubble_data['automation_risk'].max() - 0.1, 
                "High Risk /\nLow Demand", 
                ha='left', 
                va='top',
                fontsize=11,
                color='white',
                bbox=dict(boxstyle="round,pad=0.4", fc="darkred", ec="#3498db", alpha=0.7)
            )
            
            ax.text(
                bubble_data['demand_index'].max() - 0.5, 
                bubble_data['automation_risk'].max() - 0.1, 
                "High Risk /\nHigh Demand", 
                ha='right', 
                va='top',
                fontsize=11,
                color='white',
                bbox=dict(boxstyle="round,pad=0.4", fc="darkorange", ec="#3498db", alpha=0.7)
            )
            
            ax.text(
                bubble_data['demand_index'].min() + 0.5, 
                bubble_data['automation_risk'].min() + 0.1, 
                "Low Risk /\nLow Demand", 
                ha='left', 
                va='bottom',
                fontsize=11,
                color='white',
                bbox=dict(boxstyle="round,pad=0.4", fc="darkblue", ec="#3498db", alpha=0.7)
            )
            
            ax.text(
                bubble_data['demand_index'].max() - 0.5, 
                bubble_data['automation_risk'].min() + 0.1, 
                "Low Risk /\nHigh Demand", 
                ha='right', 
                va='bottom',
                fontsize=11,
                color='white',
                bbox=dict(boxstyle="round,pad=0.4", fc="darkgreen", ec="#3498db", alpha=0.7)
            )
            
            # Create custom legend for bubble sizes
            handles, _ = scatter.legend_elements(prop="sizes", alpha=0.6, num=4)
            
            # Generate labels directly
            salary_values = [100, 150, 200, 250]  # Representative values in thousands
            custom_labels = [f"${val}K" for val in salary_values]
            
            legend1 = ax.legend(handles, custom_labels, loc="upper right", title="Median Salary")
            plt.setp(legend1.get_title(), color='white')
            
            # Customize graph
            ax.set_title('Relationship Between Demand and Automation Risk', fontsize=18, fontweight='bold', pad=20)
            ax.set_xlabel('Demand Index', fontsize=14, fontweight='bold')
            ax.set_ylabel('Automation Risk', fontsize=14, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Technical Specialization', fontsize=12, fontweight='bold')
            cbar.ax.yaxis.label.set_color('white')
            cbar.ax.tick_params(colors='white')
            
            # Set background color to match Streamlit's darker theme
            fig.patch.set_facecolor('#2c3e50')
            ax.set_facecolor('#2c3e50')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#3498db')
            
            # Add explanatory text
            ax.text(
                0.5, -0.15, 
                "This chart shows the relationship between demand and automation risk for different roles.\nBubble size represents salary level and color indicates technical specialization.",
                transform=ax.transAxes, 
                ha='center', 
                fontsize=11, 
                color='white', 
                alpha=0.8
            )
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("""
        #### Key Insights
        * **AI creators** command the highest salaries, followed by those who apply or implement AI
        * **Quadrant analysis**: The most desirable roles are in the "Low Risk / High Demand" quadrant
        * **Specialized skills**: Roles with higher technical specialization (brighter colors) tend to face lower automation risk
        * **Salary protection**: Larger bubbles in high-risk areas indicate "risk premiums" for certain roles
        * **Future-proof roles**: Data Scientists and ML Engineers combine high demand with relatively low automation risk
        """)
    
    # 6. WORK SETTING TRENDS
    elif st.session_state.active_tab == "Work Setting Trends":
        # Create visualization comparing work settings across dimensions
        work_settings = df.groupby('work_setting')['salary_in_usd'].agg(['count', 'mean', 'median']).reset_index()
        
        # Create a more complex visualization that shows trends over time if work_year exists
        if 'work_year' in df.columns:
            # Create a line chart showing trends by work setting over time
            work_year_data = df.groupby(['work_year', 'work_setting'])['salary_in_usd'].median().reset_index()
            work_year_pivot = work_year_data.pivot(index='work_year', columns='work_setting', values='salary_in_usd')
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot lines with markers
            for setting in work_year_pivot.columns:
                ax.plot(work_year_pivot.index, work_year_pivot[setting], marker='o', linewidth=3, 
                       label=setting, alpha=0.8, markersize=10)
            
            # Add markers at each data point
            for setting in work_year_pivot.columns:
                for year in work_year_pivot.index:
                    if not pd.isna(work_year_pivot.loc[year, setting]):
                        ax.text(year, work_year_pivot.loc[year, setting] + 5000, 
                               f"${work_year_pivot.loc[year, setting]:,.0f}", 
                               ha='center', va='bottom', fontsize=9, fontweight='bold', 
                               color='white', alpha=0.9)
            
            # Customize graph
            ax.set_xlabel('Year', fontsize=12, fontweight='bold')
            ax.set_ylabel('Median Salary (USD)', fontsize=12, fontweight='bold')
            ax.set_title('Salary Trends by Work Setting Over Time', fontsize=18, fontweight='bold', pad=20)
            ax.grid(linestyle='--', alpha=0.3)
            ax.legend(title='Work Setting', title_fontsize=12)
            
            # Set background color to match Streamlit's darker theme
            fig.patch.set_facecolor('#2c3e50')
            ax.set_facecolor('#2c3e50')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#3498db')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Create a stacked bar chart showing work setting by experience level
        work_exp_data = df.groupby(['experience_level_desc', 'work_setting'])['salary_in_usd'].median().reset_index()
        work_exp_pivot = work_exp_data.pivot(index='experience_level_desc', columns='work_setting', values='salary_in_usd')
        
        # Ensure correct experience level order
        if all(level in work_exp_pivot.index for level in ['Entry-Level', 'Mid-Level', 'Senior', 'Executive']):
            work_exp_pivot = work_exp_pivot.reindex(['Entry-Level', 'Mid-Level', 'Senior', 'Executive'])
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot stacked bars
        work_exp_pivot.plot(kind='bar', stacked=False, ax=ax, colormap='viridis', width=0.7)
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='${:,.0f}', padding=5, fontweight='bold')
        
        # Customize graph
        ax.set_xlabel('Experience Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Median Salary (USD)', fontsize=12, fontweight='bold')
        ax.set_title('Work Setting Compensation by Experience Level', fontsize=18, fontweight='bold', pad=20)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend(title='Work Setting')
        
        # Set background color to match Streamlit's darker theme
        fig.patch.set_facecolor('#2c3e50')
        ax.set_facecolor('#2c3e50')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#3498db')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Distribution of work settings across different job categories
        work_job_data = df.groupby(['job_category', 'work_setting']).size().reset_index(name='count')
        work_job_pivot = work_job_data.pivot(index='job_category', columns='work_setting', values='count')
        work_job_pivot = work_job_pivot.fillna(0)
        
        # Convert to percentages
        work_job_pct = work_job_pivot.div(work_job_pivot.sum(axis=1), axis=0) * 100
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create a colorful heatmap
        sns.heatmap(work_job_pct, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax, linewidths=.5)
        
        # Customize heatmap
        ax.set_title('Work Setting Distribution by Job Category (%)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Work Setting', fontsize=12, fontweight='bold')
        ax.set_ylabel('Job Category', fontsize=12, fontweight='bold')
        
        # Set background color to match Streamlit's darker theme
        fig.patch.set_facecolor('#2c3e50')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        #### Key Insights
        * **Remote premium** exists for most job categories, especially at senior and executive levels
        * **Hybrid flexibility**: Hybrid work settings show balanced compensation while providing flexibility
        * **Changing landscape**: The trend toward remote work has accelerated, with increasing salary parity
        * **Role differences**: Technical roles (ML Engineers, Data Engineers) are more likely to have remote options
        * **Experience impact**: Remote work compensation premium increases with experience level
        """)
    
    # 7. WORK-LIFE BALANCE
    elif st.session_state.active_tab == "Work-Life Balance":
        if 'work_life_balance' in df.columns:
            # Create a bubble chart showing salary, work-life balance, and job category
            # Increase figure size to accommodate all elements
            fig, ax = plt.subplots(figsize=(14, 10))

            # Prepare data
            wlb_data = df.groupby('job_category')[['salary_in_usd', 'work_life_balance']].mean().reset_index()

            # Count number of professionals in each category
            job_counts = df['job_category'].value_counts()
            wlb_data['count'] = wlb_data['job_category'].map(job_counts)

            # Create custom color palette
            colors = plt.cm.viridis(np.linspace(0, 1, len(wlb_data)))

            # Create bubble chart
            scatter = ax.scatter(wlb_data['work_life_balance'], wlb_data['salary_in_usd'], 
                                s=wlb_data['count']*2, c=colors, alpha=0.7, edgecolors='white')

            # Add labels for each bubble - improved positioning
            for i, row in wlb_data.iterrows():
                # Use different offsets based on position to prevent overlap
                if row['work_life_balance'] < wlb_data['work_life_balance'].median():
                    x_offset = -5
                    ha = 'right'
                else:
                    x_offset = 5
                    ha = 'left'
                    
                ax.annotate(row['job_category'], 
                        (row['work_life_balance'], row['salary_in_usd']),
                        xytext=(x_offset, 5), textcoords="offset points",
                        fontsize=9, fontweight='bold', color='white',
                        ha=ha)

            # Add a best fit line
            z = np.polyfit(wlb_data['work_life_balance'], wlb_data['salary_in_usd'], 1)
            p = np.poly1d(z)
            ax.plot(wlb_data['work_life_balance'], p(wlb_data['work_life_balance']), 
                "--", color='#e74c3c', linewidth=2)

            # Highlight quadrants - using actual min/max of displayed data
            median_salary = wlb_data['salary_in_usd'].median()
            median_wlb = wlb_data['work_life_balance'].median()

            ax.axhline(y=median_salary, color='white', linestyle='--', alpha=0.5)
            ax.axvline(x=median_wlb, color='white', linestyle='--', alpha=0.5)

            # Calculate actual plot boundaries with more space
            min_wlb = wlb_data['work_life_balance'].min() * 0.92
            max_wlb = wlb_data['work_life_balance'].max() * 1.08
            min_salary = wlb_data['salary_in_usd'].min() * 0.92
            max_salary = wlb_data['salary_in_usd'].max() * 1.08

            # Set axis limits explicitly to ensure all content fits
            ax.set_xlim(min_wlb, max_wlb)
            ax.set_ylim(min_salary, max_salary)

            # Add labels for quadrants - moved further toward corners to reduce overlap
            # High Stress, High Pay (bottom left)
            ax.text(min_wlb + (max_wlb - min_wlb) * 0.02, 
                max_salary - (max_salary - min_salary) * 0.02,
                "High Stress\nHigh Pay", 
                ha='left', va='top',
                color='white', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc='darkred', alpha=0.5))

            # Low Stress, High Pay (bottom right)
            ax.text(max_wlb - (max_wlb - min_wlb) * 0.02, 
                max_salary - (max_salary - min_salary) * 0.02,
                "Low Stress\nHigh Pay", 
                ha='right', va='top',
                color='white', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc='darkgreen', alpha=0.5))

            # High Stress, Low Pay (top left)
            ax.text(min_wlb + (max_wlb - min_wlb) * 0.02, 
                min_salary + (max_salary - min_salary) * 0.02,
                "High Stress\nLow Pay", 
                ha='left', va='bottom',
                color='white', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc='darkorange', alpha=0.5))

            # Low Stress, Low Pay (top right)
            ax.text(max_wlb - (max_wlb - min_wlb) * 0.02, 
                min_salary + (max_salary - min_salary) * 0.02,
                "Low Stress\nLow Pay", 
                ha='right', va='bottom',
                color='white', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc='darkblue', alpha=0.5))

            # Customize graph
            ax.set_xlabel('Work-Life Balance Score', fontsize=12, fontweight='bold')
            ax.set_ylabel('Average Salary (USD)', fontsize=12, fontweight='bold')
            ax.set_title('Salary vs. Work-Life Balance by Role', fontsize=18, fontweight='bold', pad=20)
            ax.grid(linestyle='--', alpha=0.3)

            # Move the legend to a better position to avoid overlap
            handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=4)
            legend = ax.legend(handles, labels, loc="center right", title="Number of Professionals", 
                            bbox_to_anchor=(0.95, 0.5), framealpha=0.8)
            plt.setp(legend.get_title(), color='white')

            # Set background color to match Streamlit's darker theme
            fig.patch.set_facecolor('#2c3e50')
            ax.set_facecolor('#2c3e50')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#3498db')

            # Increase margins to give more space 
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

            st.pyplot(fig)
            
            # Create a chart comparing total compensation to base salary across roles
            if 'total_compensation_estimate' in df.columns:
                # Calculate the compensation ratio
                total_comp_data = df.groupby('job_category')[['salary_in_usd', 'total_compensation_estimate']].mean().reset_index()
                total_comp_data['comp_ratio'] = total_comp_data['total_compensation_estimate'] / total_comp_data['salary_in_usd']
                total_comp_data['bonus_value'] = total_comp_data['total_compensation_estimate'] - total_comp_data['salary_in_usd']
                total_comp_data = total_comp_data.sort_values('comp_ratio', ascending=False)
                
                # Limit to top 12 roles for better readability if there are many
                if len(total_comp_data) > 12:
                    total_comp_data = total_comp_data.head(12)
                
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Create the bars
                x = np.arange(len(total_comp_data))
                width = 0.35
                
                bars1 = ax.bar(x, total_comp_data['salary_in_usd'], width, label='Base Salary', color='#3498db')
                bars2 = ax.bar(x, total_comp_data['bonus_value'], width, bottom=total_comp_data['salary_in_usd'], 
                            label='Bonuses & Benefits', color='#f39c12')
                
                # Add ratio labels
                for i, row in total_comp_data.iterrows():
                    ax.text(i, row['total_compensation_estimate'] + 5000, 
                        f"{row['comp_ratio']:.2f}x", ha='center', fontweight='bold', color='white')
                
                # Customize graph
                ax.set_xlabel('Job Category', fontsize=12, fontweight='bold')
                ax.set_ylabel('Compensation (USD)', fontsize=12, fontweight='bold')
                ax.set_title('Total Compensation Structure by Role', fontsize=18, fontweight='bold', pad=20)
                ax.set_xticks(x)
                ax.set_xticklabels(total_comp_data['job_category'], rotation=45, ha='right')
                ax.legend()
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                
                # Set background color to match Streamlit's darker theme
                fig.patch.set_facecolor('#2c3e50')
                ax.set_facecolor('#2c3e50')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                for spine in ax.spines.values():
                    spine.set_color('#3498db')
                
                # Set margins explicitly
                plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)
                
                st.pyplot(fig)
        
        st.markdown("""
        #### Key Insights
        * **Inverse relationship**: Higher salaries often correlate with lower work-life balance scores
        * **Executive tradeoff**: Management roles typically come with both higher compensation and higher stress
        * **Hidden benefits**: Total compensation can be up to 1.5x base salary when including bonuses and benefits
        * **Optimal roles**: Some specialized technical roles offer both good work-life balance and competitive pay
        * **Sector impact**: Certain sectors (Education, Non-profit) prioritize work-life balance over maximum compensation
        """)
    
    # 8. FUTURE OUTLOOK
    elif st.session_state.active_tab == "Future Outlook":
        if 'demand_index' in df.columns:
            # Create visualization showing demand trends
            demand_data = df.groupby('job_category')[['demand_index', 'salary_in_usd', 'automation_risk']].mean().reset_index()
            demand_data = demand_data.sort_values('demand_index', ascending=False)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create horizontal bars with gradient colors based on automation risk
            colors = plt.cm.RdYlGn_r(demand_data['automation_risk'])
            
            # Create bars for demand index
            bars = ax.barh(demand_data['job_category'], demand_data['demand_index'], 
                          color=colors, alpha=0.8, height=0.6)
            
            # Add labels
            for i, bar in enumerate(bars):
                # Add demand index value
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                       f"{demand_data.iloc[i]['demand_index']:.1f}", 
                       va='center', fontweight='bold', color='white')
                
                # Add salary as a secondary indicator
                ax.text(0.1, bar.get_y() + bar.get_height()/2, 
                       f"${demand_data.iloc[i]['salary_in_usd']:,.0f}", 
                       va='center', ha='left', fontsize=8, 
                       fontweight='bold', color='black')
            
            # Customize graph
            ax.set_xlabel('Demand Index (Higher = More In Demand)', fontsize=12, fontweight='bold')
            ax.set_title('Future Demand Outlook by Role', fontsize=18, fontweight='bold', pad=20)
            ax.grid(axis='x', linestyle='--', alpha=0.3)
            
            # Add a colorbar for automation risk
            sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(0, 1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Automation Risk', fontweight='bold', color='white')
            cbar.ax.tick_params(colors='white')
            
            # Set background color to match Streamlit's darker theme
            fig.patch.set_facecolor('#2c3e50')
            ax.set_facecolor('#2c3e50')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#3498db')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        if 'career_path' in df.columns:
            # Create a visualization showing career path progression
            career_paths = df['career_path'].value_counts().head(8)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create bars with custom colors
            bars = ax.bar(career_paths.index, career_paths.values, 
                         color=plt.cm.viridis(np.linspace(0, 1, len(career_paths))), 
                         alpha=0.8, width=0.7)
            
            # Add count labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                       f'{height:,.0f}',
                       ha='center', va='bottom', fontweight='bold', color='white')
            
            # Customize graph
            ax.set_xlabel('Career Path', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Professionals', fontsize=12, fontweight='bold')
            ax.set_title('Popular Career Progression Paths', fontsize=18, fontweight='bold', pad=20)
            ax.set_xticklabels(career_paths.index, rotation=45, ha='right')
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Set background color to match Streamlit's darker theme
            fig.patch.set_facecolor('#2c3e50')
            ax.set_facecolor('#2c3e50')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#3498db')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Create a comparison of salaries by industry sector and experience
            if 'company_sector' in df.columns:
                # Get top sectors by count
                top_sectors = df['company_sector'].value_counts().head(5).index.tolist()
                sector_exp_data = df[df['company_sector'].isin(top_sectors)]
                
                # Group by sector and experience
                sector_data = sector_exp_data.groupby(['company_sector', 'experience_level_desc'])['salary_in_usd'].median().reset_index()
                sector_pivot = sector_data.pivot(index='company_sector', columns='experience_level_desc', values='salary_in_usd')
                
                # Ensure column order
                if all(col in sector_pivot.columns for col in ['Entry-Level', 'Mid-Level', 'Senior', 'Executive']):
                    sector_pivot = sector_pivot[['Entry-Level', 'Mid-Level', 'Senior', 'Executive']]
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Create a grouped bar chart
                sector_pivot.plot(kind='bar', ax=ax, colormap='viridis')
                
                # Add value labels
                for container in ax.containers:
                    ax.bar_label(container, fmt='${:,.0f}', fontsize=8, fontweight='bold')
                
                # Customize graph
                ax.set_xlabel('Industry Sector', fontsize=12, fontweight='bold')
                ax.set_ylabel('Median Salary (USD)', fontsize=12, fontweight='bold')
                ax.set_title('Salary Progression by Top Industry Sectors', fontsize=18, fontweight='bold', pad=20)
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                ax.legend(title='Experience Level')
                
                # Set background color to match Streamlit's darker theme
                fig.patch.set_facecolor('#2c3e50')
                ax.set_facecolor('#2c3e50')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                for spine in ax.spines.values():
                    spine.set_color('#3498db')
                
                plt.tight_layout()
                st.pyplot(fig)
        
        st.markdown("""
        #### Key Insights
        * **Highest demand**: Machine Learning and AI-focused roles show the strongest future demand
        * **Emerging roles**: Roles combining domain expertise with data skills are rapidly growing
        * **Career transitions**: Analytics-to-Engineering and Specialist-to-Management are common progression paths
        * **Industry growth**: Finance and Healthcare are projected to have the largest data science job growth
        * **Resilient roles**: Positions requiring both technical skills and strategic thinking show the lowest automation risk
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Bottom section with key takeaways from the data
    st.markdown('<h2 class="section-header">Key Salary Determinants</h2>', unsafe_allow_html=True)
    
    # Create 3-column layout for key takeaways
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Primary Factors
        
        1. **Experience Level**
           * 40-50% increase from Entry to Mid-Level
           * 30-40% increase from Mid to Senior 
           * 35-55% increase from Senior to Executive
        
        2. **Job Specialization**
           * Technical specialization correlates with 10-15% higher salary
           * AI/ML-specific roles command 15-25% premium
        
        3. **Geographic Region**
           * North America offers highest absolute salaries
           * Remote work enables global compensation arbitrage
        """)
        
    with col2:
        st.markdown("""
        ### Secondary Factors
        
        1. **Work Setting**
           * Remote work often carries 5-10% premium
           * Setting impact varies by experience level
        
        2. **Company Sector**
           * Finance & Technology lead in compensation
           * Sector impact strongest at executive level
        
        3. **Company Size**
           * Large companies offer 10-15% higher base pay
           * Startups may compensate with equity instead
        """)
        
    with col3:
        st.markdown("""
        ### Emerging Trends
        
        1. **AI Impact**
           * AI creators & implementers command highest premiums
           * Automation risk correlates with compensation adjustments
        
        2. **Work-Life Balance**
           * Slight inverse relationship with compensation
           * Total benefits package can add 20-50% to base salary
        
        3. **Remote Revolution**
           * Geographic salary differences becoming less pronounced
           * Remote work increasingly normalized across all roles
        """)
    

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
    
    # Initialize session state variables
    if "results_ready" not in st.session_state:
        st.session_state.results_ready = False
    if "calculation_complete" not in st.session_state:
        st.session_state.calculation_complete = False
    
    # Create a tab-based interface for better organization
    tab1, tab2 = st.tabs(["📝 Input Your Details", "💰 View Results"])
    
    # Input form tab
    with tab1:
        # Key variables section
        st.markdown('<h2 class="section-header">Your Professional Profile</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
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
                options=["Europe", "North America", "Asia", "South America", "Africa", "Oceania"]
            )
            
            work_setting = st.selectbox(
                "Work Setting",
                options=["Remote", "Hybrid", "On-site"]
            )
            
            company_sector = st.selectbox(
                "Company Sector",
                options=["Finance", "Technology", "Healthcare", "Retail", "Manufacturing", "Education", "Other"]
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Additional details section
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown('<h3 class="subsection-header">Additional Details</h3>', unsafe_allow_html=True)
            st.info("These factors will refine the prediction")
            
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
        
        # Submit button - centered and full width
        st.markdown('<div style="text-align: center; margin-top: 20px;">', unsafe_allow_html=True)
        predict_button = st.button("Calculate My Salary Estimate", type="primary", key="predict_salary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process the calculation when the button is clicked
        if predict_button:
            # Show spinner in this tab
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
                    st.session_state.calculation_complete = True
                    
                    # Store other values for visualization
                    st.session_state.job_category = job_category
                    st.session_state.experience_level = experience_level
                    st.session_state.region = region
                    st.session_state.tech_specialization = tech_specialization
                    st.session_state.english_level = english_level
                    
                    # Display success message immediately
                    st.success("✅ Your salary has been calculated successfully! Click on the 'View Results' tab to see your personalized salary prediction.")
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
        
        # Show success message if calculation was previously completed
        elif st.session_state.calculation_complete:
            st.success("✅ Your salary has been calculated successfully! Click on the 'View Results' tab to see your personalized salary prediction.")
    
    # Results tab
    with tab2:
        # Display results if ready
        if st.session_state.results_ready:
            base_predicted_salary = st.session_state.base_salary
            adjusted_salary = st.session_state.adjusted_salary
            job_category = st.session_state.job_category
            experience_level = st.session_state.experience_level
            region = st.session_state.region
            tech_specialization = st.session_state.tech_specialization
            english_level = st.session_state.english_level
            
            # Main prediction result - full width
            st.markdown('<div class="metric-container" style="text-align: center; padding: 20px;">', unsafe_allow_html=True)
            st.markdown('<h3 class="subsection-header">Estimated Annual Salary</h3>', unsafe_allow_html=True)
            st.markdown(f"<h1 style='color:#1976D2; text-align:center; font-size:4rem'>${adjusted_salary:,.2f}</h1>", unsafe_allow_html=True)
            
            # Show adjustment info
            adjustment_pct = (adjusted_salary / base_predicted_salary - 1) * 100
            adjustment_text = "increase" if adjustment_pct >= 0 else "decrease"
            
            st.info(f"Your profile details resulted in a {abs(adjustment_pct):.1f}% {adjustment_text} from the base prediction.")
            
            if job_category == "Business Analyst":
                st.info("Note: Business Analyst salaries are typically 15% lower than comparable Data Analyst positions.")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional metrics - 3 columns
            st.markdown('<h3 class="subsection-header">Salary Breakdown</h3>', unsafe_allow_html=True)
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
            
            # Rest of the code remains the same...
            # Two visualizations side by side
            st.markdown('<h3 class="subsection-header">Career Analytics</h3>', unsafe_allow_html=True)
            vis_col1, vis_col2 = st.columns(2)
            
            # Tech stack visualization
            with vis_col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<h4 class="subsection-header">Technology Stack Analysis</h4>', unsafe_allow_html=True)
                
                tech_fig = plot_tech_stack(job_category)
                if tech_fig:
                    st.pyplot(tech_fig)
                    st.markdown(f"""
                    The chart shows key technologies for your role as a **{job_category}**.
                    Mastering these top skills can increase your market value.
                    """)
                else:
                    st.write("No technology stack information available for this role.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Salary growth trajectory
            with vis_col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<h4 class="subsection-header">5-Year Salary Projection</h4>', unsafe_allow_html=True)
                
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
                fig3, ax3 = plt.subplots(figsize=(8, 4))
                ax3 = sns.lineplot(x='Year', y='Salary', data=projection_df, marker='o', linewidth=2, color='#1976D2')
                
                # Add value labels
                for i, val in enumerate(projected_salaries):
                    ax3.text(i, val + 5000, f"${val:,.0f}", ha='center', fontsize=8)
                
                # Customize chart
                ax3.set_title('Projected Salary Growth')
                ax3.grid(True, linestyle='--', alpha=0.7)
                
                # Set background colors
                fig3.patch.set_facecolor('#2c3e50')
                ax3.set_facecolor('#2c3e50')
                ax3.tick_params(colors='white')
                ax3.xaxis.label.set_color('white')
                ax3.yaxis.label.set_color('white')
                ax3.title.set_color('white')
                
                st.pyplot(fig3)
                
                st.markdown(f"""
                With an annual growth rate of **{growth_rate*100:.0f}%**, your salary could 
                reach **${projected_salaries[-1]:,.0f}** in 5 years.
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Personalized recommendations - full width
            st.markdown('<h3 class="subsection-header">Career Growth Strategy</h3>', unsafe_allow_html=True)
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            
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
        
        else:
            # Message when no results are available
            st.info("Complete the form in the 'Input Your Details' tab and click 'Calculate My Salary Estimate' to see your personalized salary prediction.")




# ======= MAIN APP =======
def main():
    # Apply unified CSS with WHITE text for better visibility on dark backgrounds
    st.markdown("""
    <style>
        /* Base text and background colors for dark theme compatibility */
        body {
            color: #ffffff !important;
            background-color: #1e1e1e !important;
        }
        
        /* Force ALL text elements to have white color for visibility */
        p, h1, h2, h3, h4, h5, h6, li, span, div, label, .stMarkdown, 
        .stText, [data-testid="stVerticalBlock"] p {
            color: #ffffff !important;
            font-weight: 500 !important;
        }
        
        /* Fix sidebar styling */
        [data-testid="stSidebar"], .css-1d391kg, .css-1lcbmhc {
            background-color: #2c3e50 !important;
        }
        
        /* Make sidebar headers bold and very visible */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] .stTitle {
            color: #3498db !important;
            font-weight: 700 !important;
        }
        
        /* Make sidebar paragraphs visible */
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
            color: #ffffff !important;
            font-weight: 500 !important;
        }
        
        /* Prominent radio buttons */
        .stRadio > div {
            padding: 10px 0 !important;
        }
        .stRadio label {
            background-color: transparent !important;
            color: #ffffff !important;
            font-weight: 500 !important;
            padding: 8px !important;
            border-radius: 4px !important;
        }
        .stRadio label:hover {
            background-color: rgba(52, 152, 219, 0.3) !important;
        }
        .stRadio label[data-baseweb="radio"] input:checked + span {
            background-color: #3498db !important;
            border-color: #3498db !important;
        }
        
        /* Style headings throughout the app */
        .main-header {
            color: #3498db !important;
            font-size: 2.5rem !important;
            font-weight: 800 !important;
        }
        .section-header {
            color: #ffffff !important;
            font-size: 1.8rem !important;
            font-weight: 700 !important;
        }
        .subsection-header {
            color: #ecf0f1 !important;
            font-size: 1.3rem !important;
            font-weight: 650 !important;
        }
        
        /* Container backgrounds with dark theme */
        .chart-container, .metric-container, .info-card, .info-box, 
        .success-box, .warning-box, .error-box, .limitations-container {
            background-color: #2c3e50 !important;
            color: #ffffff !important;
            border: 1px solid #34495e !important;
            border-radius: 8px !important;
            padding: 15px !important;
        }
        
        /* Ensure list items are visible */
        .styled-list li {
            color: #ffffff !important;
            font-weight: 500 !important;
            margin-bottom: 10px !important;
        }
        .styled-list li:before {
            color: #3498db !important;
        }
        
        /* Alert boxes */
        .stAlert {
            background-color: #2c3e50 !important;
            border: 1px solid #3498db !important;
        }
        .stAlert p, .stAlert div {
            color: #ffffff !important;
            font-weight: 500 !important;
        }
        
        /* Hero section */
        .hero-container {
            background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%) !important;
            padding: 2rem !important;
        }
        .hero-container h1, .hero-title {
            color: white !important;
            font-weight: 800 !important;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.4) !important;
        }
        .hero-container p, .hero-text {
            color: white !important;
            font-weight: 500 !important;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.4) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state for navigation
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://img.freepik.com/free-vector/illustration-data-analysis-graph_53876-18139.jpg", width=200)
        st.title("Navigation")
        
        # FIX FOR DOUBLE-CLICK ISSUE:
        # Use a callback approach to update immediately on first click
        def navigate_to(page_name):
            st.session_state.page = page_name
            st.rerun()
        
        # Use buttons stacked in a single column for navigation
        if st.button("Home", key="home_btn", use_container_width=True):
            navigate_to("Home")
            
        if st.button("Visualizations", key="viz_btn", use_container_width=True):
            navigate_to("Visualizations")
            
        if st.button("Prediction Tool", key="pred_btn", use_container_width=True):
            navigate_to("Prediction Tool")
        
        # Visual indicator of current page - highlight current page with a colored label
        st.markdown(
            f"""
            <div style="margin-top: 10px; text-align: center;">
                <span style="background-color: #3498db; color: white; padding: 4px 12px; border-radius: 15px; font-size: 0.8em;">
                    {st.session_state.page}
                </span>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Add visual footer instead of plain about text
        st.markdown("---")
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%); 
                        border-radius: 10px; padding: 15px; margin-top: 20px; 
                        border: 1px solid #3498db; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="color: white; margin-bottom: 10px; font-size: 1.2em; text-align: center;">
                    <span style="margin-right: 8px;">💰</span> Salary Predictor
                </h3>
                <p style="color: white; font-size: 0.9em; margin-bottom: 10px; text-align: center;">
                    Estimate your data science market value
                </p>
                <div style="display: flex; justify-content: center; margin-top: 15px;">
                    <span style="color: #3498db; background: rgba(255,255,255,0.2); 
                                 margin: 0 5px; padding: 5px 10px; border-radius: 5px; font-size: 1em;">
                        📊
                    </span>
                    <span style="color: #3498db; background: rgba(255,255,255,0.2); 
                                 margin: 0 5px; padding: 5px 10px; border-radius: 5px; font-size: 1em;">
                        🔍
                    </span>
                    <span style="color: #3498db; background: rgba(255,255,255,0.2); 
                                 margin: 0 5px; padding: 5px 10px; border-radius: 5px; font-size: 1em;">
                        💻
                    </span>
                </div>
                <p style="color: #ecf0f1; font-size: 0.7em; margin-top: 15px; text-align: center; opacity: 0.8;">
                    v1.0.0 • Data Science Analytics Tool
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Display the selected page
    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Visualizations":
        visualizations_page()
    elif st.session_state.page == "Prediction Tool":
        prediction_page()

if __name__ == "__main__":
    main()