# Salary Prediction Project

This project focuses on analyzing and predicting salaries based on demographic and professional information using machine learning.

##   Structure

- `salary_prediction_EDA.ipynb`:  
  This notebook performs **Exploratory Data Analysis (EDA)** on the dataset.  
  It includes data cleaning, handling missing values, detecting outliers, and visualizing relationships between salary and features like gender, education level, job title, etc.

- `salary_prediction.ipynb`:  
  This notebook is responsible for **training and evaluating three regression models**:
  - Linear Regression  
  - Decision Tree Regressor  
  - Random Forest Regressor  
  
  It also computes various performance metrics (MAE, MAPE, MSE, RMSE, R²), compares the models, and saves them in the `models/` directory using `joblib`.

##  Requirements

Make sure to install the following Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
