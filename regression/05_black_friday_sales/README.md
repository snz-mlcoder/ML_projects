## Black Friday Sales Prediction
This project is a data science pipeline to predict the purchase amount of customers during Black Friday sales. It includes data cleaning, exploratory data analysis (EDA), feature engineering, and a comparative analysis of several machine learning regression models.

## Project Files
black_friday_sales.csv: The raw dataset.

black_friday_sales.ipynb: A self-contained Python script to run the entire pipeline in a single notebook cell.

best_black_friday_model.joblib: The final trained model, saved for future use.

README.md: Project overview and instructions.

## Key Findings & Results
The analysis revealed that product categories, age, and occupation are key predictors of purchase amount. We evaluated several models, with the LGBM Regressor outperforming others by achieving the lowest Root Mean Square Error (RMSE).

## Installation & Usage
To run the project, ensure you have Python 3.x installed along with the following libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn lightgbm joblib
```
