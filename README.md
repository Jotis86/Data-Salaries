# 📊 Salary Analysis and Prediction for Data Professionals

## 📝 Project Description
This data analysis project develops a predictive model to estimate salaries in the field of Data Science and Artificial Intelligence. Using advanced machine learning techniques, the system analyzes factors such as experience, geographic location, technical specialization, and work modality to generate personalized salary estimates.

## 🚀 Key Features
- 🎯 **Personalized Salary Prediction**: The model predicts salaries in USD based on multiple professional profile variables.
- 🌍 **Comparative Analysis**: Compare salaries across different regions, experience levels, and specialties.
- 📈 **Interactive Visualization**: Dashboards and graphs to display salary trends and projections.
- 🎓 **Career Recommendations**: Personalized suggestions to improve salary prospects.
- 💻 **User-Friendly Web Interface**: A Streamlit application that makes interaction with the model easy and intuitive.

## 🛠️ Methodology
1. **Data Cleaning and Preprocessing**
   - Transformation of categorical variables
   - Creation of derived features (work-life balance index, automation risk index)
   - Salary normalization by cost of living and experience level
   - Segmentation by region and industry sector

2. **Exploratory Data Analysis**
   - Identification of salary trends by specialty
   - Correlation analysis between technical variables and compensation
   - Regional and experience-level difference analysis
   - Evaluation of the impact of work modalities (remote, hybrid, on-site)

3. **Model Development**
   - Evaluation of various algorithms (Linear Regression, Random Forest, Gradient Boosting)
   - Hyperparameter optimization using grid search and cross-validation
   - Feature selection to improve model accuracy
   - Performance evaluation using RMSE, MAE, and R² Score

4. **Application Deployment**
   - Development of an interactive interface with Streamlit
   - Integration of the optimized predictive model
   - Generation of personalized career recommendations
   - Visualization of salary projections

## 📊 Results and Performance
The final model achieved:

- **RMSE**: \$48,450.43
- **MAE**: \$37,201.02
- **R² Score**: 0.4054

These indicators show there is room for improvement, particularly through better feature selection and potentially incorporating more region-specific training data.

## 🔮 Future Improvements
- Implementation of more advanced feature selection techniques
- Incorporation of temporal data for long-term trend analysis
- More precise segmentation by specific industries
- Integration of emerging skills data and their impact on compensation
- Model enhancement to reduce RMSE and increase R² score

## 🧰 Technologies Used
- 🐍 **Python**: Core for data analysis and model development
- 📊 **Pandas/NumPy**: Data manipulation and processing
- 🤖 **Scikit-learn**: Machine learning algorithm implementation
- 📈 **Matplotlib/Seaborn**: Data visualization
- 🌐 **Streamlit**: Development of the interactive web application
- 📦 **Pickle**: Model serialization

## 🏁 Conclusions
This project demonstrates how advanced data analysis can provide valuable insights into salary trends in the Data Science field. The developed model serves as a tool for professionals seeking salary references and for organizations aiming to establish competitive compensation structures.