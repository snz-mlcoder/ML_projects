
## 🧹 Data Preprocessing Steps

The dataset originally contained raw laptop specifications and required several preprocessing steps:


- **Removed units** like `GB`, `kg`, and symbols using regular expressions.
- **Extracted binary features** such as `TouchScreen` and `IPS` from the display information.
- **Calculated pixel density (PPI)** from screen resolution and size.
- **Parsed brand names** from CPU and GPU strings (e.g., `"Intel Core i7"` → `"Intel"`).
- **Grouped rare categories** into `"Other"` (e.g., uncommon OS types).
- **Handled missing values** by dropping or replacing with generalized values.
- Final cleaned dataset saved to: `data/cleaned_laptop_data.csv`

🔍 All preprocessing steps are included in the notebook: [`laptop_price_prediction.ipynb`](laptop_price_prediction.ipynb)
