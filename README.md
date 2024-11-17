# Corporate Bankruptcy Prediction

📈 **Project Overview**

This project aims to predict corporate bankruptcy using various machine learning models. The dataset consists of financial indicators from multiple companies, and the primary goal is to identify key factors contributing to bankruptcy. This analysis helps in understanding financial health and risk management.

📂 **Dataset**

The dataset used in this project is sourced from Jennie’s GitHub repository. It contains the following key features:
	•	Financial Ratios: Various metrics like liquidity, profitability, leverage, etc.
	•	Target Variable: Bankruptcy, indicating whether a company has declared bankruptcy (1) or not (0).

Data Source: Company Bankruptcy CSV

🛠️ Tools & Libraries

	•	Python
	•	Pandas & NumPy: Data manipulation
	•	Matplotlib & Seaborn: Data visualization
	•	Scikit-learn: Machine learning models
	•	XGBoost: Advanced boosting techniques

🔍 Analysis Process

	1.	Data Import & Cleaning:
	•	Loaded the dataset from GitHub and handled potential encoding issues.
	•	Checked for missing values and performed necessary preprocessing.
	2.	Exploratory Data Analysis (EDA):
	•	Visualized the distribution of key financial ratios.
	•	Identified correlations between features and the target variable.
	3.	Model Building:
	•	Implemented various models: Logistic Regression, Decision Tree, Random Forest, and XGBoost.
	•	Evaluated model performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
	4.	Results & Recommendations:
	•	Compared the performance of different models.
	•	Provided insights on the most significant predictors of bankruptcy.

📝 Results Summary

	•	Best Model: XGBoost achieved the highest accuracy and F1-score.
	•	Key Predictors: Liquidity ratios and profitability indicators were found to be the most significant features in predicting bankruptcy.

📊 Visualizations

Several visualizations were created to support the analysis:
	•	Correlation heatmap to identify feature relationships.
	•	ROC curve to compare model performance.
	•	Feature importance plot from the XGBoost model.

 🚀 How to Reproduce

To reproduce the analysis:
	1.	Clone the repository:
 git clone https://github.com/jtmtran/Corporate-Bankruptcy-Prediction.git

 2.	Install the required packages:
  pip install -r requirements.txt

  3.	Run the Google Colab notebook:
GoogleColab notebook Corporate_Bankruptcy_Prediction_final.ipynb

🔍 Example Code & Outputs

1. Data Loading
   
3. Correlation Heatmap
   
4. Model Evaluation
   
5. Confusion Matrix

📬 Contact

For any questions, please reach out to Jennie Tran via LinkedIn.
