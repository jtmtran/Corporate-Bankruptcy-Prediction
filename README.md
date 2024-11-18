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
```
# Load the dataset and handle potential encoding issues
# Define the URL for the dataset (hosted on GitHub)
url = 'https://raw.githubusercontent.com/jtmtran/Corporate-Bankruptcy-Prediction/refs/heads/main/Company%20Bankruptcy.csv'

# Fetch the dataset from the URL
response = requests.get(url)

# Check if the request was successful before loading the dataset
if response.status_code == 200:
    # Load the dataset into a DataFrame using the correct encoding
    df = pd.read_csv(BytesIO(response.content), encoding='ISO-8859-1')
    print("Dataset successfully loaded.")
else:
    print(f"Failed to retrieve the file. Status code: {response.status_code}")
```
   
2. Correlation Heatmap
```
# Plot the correlation heatmap to visualize relationships between features
# Plot the correlation heatmap to visualize feature relationships
import matplotlib.pyplot as plt
import seaborn as sns

#Define X_rfe using the selected features from RFE
X_rfe = df[selected_features]

#Correlation Analysis for selected Features
corr_matrix_rfe = X_rfe.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix_rfe, cmap='coolwarm', annot=True, fmt='.2f')
plt.title("Correlation Matrix (After RFE)")
plt.show()

#Identify and Drop Highly Correlated Features (|r| > 0.85)
high_corr_pairs = [(col1, col2) for col1 in corr_matrix_rfe.columns for col2 in corr_matrix_rfe.columns
                   if col1 != col2 and abs(corr_matrix_rfe.loc[col1, col2]) > 0.85]
print("Highly correlated feature pairs:", high_corr_pairs)

#Drop one feature from each highly correlated pair
features_to_drop = {pair[1] for pair in high_corr_pairs}  # Keep the first feature, drop the second
print("Features to drop due to high correlation:", features_to_drop)

# Update the DataFrame by dropping highly correlated features
X_final = X_rfe.drop(columns=list(features_to_drop))
print("Final set of features after removing highly correlated ones:", X_final.columns.tolist())
```
   
3. Model Evaluation
```
# Initialize and fit the XGBoost classifier
# Initialize and train the model
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# List of models to compare
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(random_state=42, probability=True)
}

# Dictionary to store the results
results = []

# Loop through each model, fit it, and evaluate
for model_name, model in models.items():
    # Fit the model on the SMOTE-resampled training data
    model.fit(X_train_smote, y_train_smote)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Store the results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    })

# Convert the results to a DataFrame and display
import pandas as pd
results_df = pd.DataFrame(results)
print(results_df.sort_values(by="F1 Score", ascending=False))
```
   
4. Confusion Matrix
```
# Plot the correlation heatmap to visualize relationships between features
# Plot the correlation heatmap to visualize feature relationships
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Generate the confusion matrix
#cm = confusion_matrix(y_test, y_pred_final)
cm = confusion_matrix(y_test, y_pred_xgb)


# Plot the confusion matrix using Seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix for XGBoost Classifier")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0.5, 1.5], ['Non-Bankrupt', 'Bankrupt'], rotation=0)
plt.yticks([0.5, 1.5], ['Non-Bankrupt', 'Bankrupt'], rotation=0)
plt.show()
```

📬 Contact

For any questions, please reach out to Jennie Tran via [LinkedIn](www.linkedin.com/in/jennietmtran).
