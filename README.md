# Corporate Bankruptcy Prediction

📈 **Project Overview**

This project aims to predict corporate bankruptcy using various machine learning models. The dataset consists of financial indicators from multiple companies, and the primary goal is to identify key factors contributing to bankruptcy. This analysis helps in understanding financial health and risk management.

📂 **Dataset**

The dataset used in this project is sourced from Jennie’s GitHub repository. It contains the following key features:
	•	Financial Ratios: Various metrics like liquidity, profitability, leverage, etc.
	•	Target Variable: Bankruptcy, indicating whether a company has declared bankruptcy (1) or not (0).

**Data Source**: [Company Bankruptcy CSV](https://github.com/jtmtran/Corporate-Bankruptcy-Prediction/blob/540549b59b7b8a69deef83e9324302c364a5b91c/Company%20Bankruptcy.csv)

🛠️ Tools & Libraries

	•	Python
	•	Pandas & NumPy: Data manipulation
	•	Matplotlib & Seaborn: Data visualization
	•	Scikit-learn: Machine learning models
	•	XGBoost: Advanced boosting techniques

🔍 Analysis Process

	1.	Data Import & Cleaning:
 		Loaded the dataset from GitHub and handled potential encoding issues.
		Checked for missing values and performed necessary preprocessing.
	2.	Exploratory Data Analysis (EDA):
		Visualized the distribution of key financial ratios.
		Identified correlations between features and the target variable.
	3.	Model Building:
		Implemented various models: Logistic Regression, Decision Tree, Random Forest, and XGBoost.
		Evaluated model performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
	4.	Results & Recommendations:
		Compared the performance of different models.
		Provided insights on the most significant predictors of bankruptcy.

📝 Results Summary

	•	Best Model: XGBoost achieved the highest accuracy and F1-score.
	•	Key Predictors: Liquidity ratios and profitability indicators were found to be the most significant features in predicting bankruptcy.

📊 Visualizations

Several visualizations were created to support the analysis:

	•	Correlation heatmap to identify feature relationships.
	•	ROC curve to compare model performance.
	•	Feature importance plot from the XGBoost model.

 ## 🚀 How to Reproduce

Follow these steps to get the project running on your machine:
Prerequisites

	•	Python 3.10 or higher
	•	Jupyter Notebook or Jupyter Lab installed
	•	Required packages listed in requirements.txt

1. **Clone the Repository**:
 ```
   git clone https://github.com/jtmtran/Corporate-Bankruptcy-Prediction.git
   cd Corporate-Bankruptcy-Prediction
```

2. **Create a Virtual Environment (Optional but Recommended)**:
```
# Create a virtual environment
python -m venv env

# Activate the virtual environment
# On Windows:
env\\Scripts\\activate
# On macOS/Linux:
source env/bin/activate
```
3. **Install Dependencies:**

```
pip install -r requirements.txt
```

4. **Download the Dataset:**
The dataset is hosted on GitHub and will be automatically fetched when running the notebook. However, you can manually download it if needed:
[Company Bankruptcy CSV](https://raw.githubusercontent.com/jtmtran/Corporate-Bankruptcy-Prediction/refs/heads/main/Company%20Bankruptcy.csv_)

5. **Run the Notebook:**

Open the Google Colab Notebook and upload the file in sequence:
[Google Colab Notebook] Corporate_Bankruptcy_Prediction_final.ipynb


🔍 **Example Code & Outputs**

1. Data Loading
   
3. Correlation Heatmap
   
4. Model Evaluation
   
5. Confusion Matrix

📬 Contact

For any questions, please reach out to Jennie Tran via [LinkedIn](www.linkedin.com/in/jennietmtran).
