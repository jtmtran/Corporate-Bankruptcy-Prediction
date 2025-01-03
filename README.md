# Corporate Bankruptcy Prediction

## Project Overview

This project predicts corporate bankruptcy using machine learning models based on financial indicators. The dataset includes various metrics related to company performance, with the goal of identifying factors contributing to bankruptcy. This analysis supports financial risk management and decision-making.

[Notebook](https://github.com/jtmtran/Corporate-Bankruptcy-Prediction/blob/02e556e58a67700372e6dc475a5c1dd6839fb4fd/Corporate_Bankruptcy_Prediction_finalll.ipynb)

## Dataset

Dataset
- Source: Taiwan Economic Journal, covering 1999–2009.
- Features: Financial ratios representing liquidity, profitability, leverage, and other metrics.
- Target Variable: Bankrupt?, where 1 indicates bankruptcy and 0 indicates non-bankruptcy.
- Data Preprocessing:
  - Column names were cleaned by stripping unnecessary spaces for consistency.
  - Missing values were not observed in the dataset.

**Data Source**: [Corporate Bankruptcy Prediction](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction)

## Tools & Libraries
- Python: Programming Environment
- Pandas & NumPy: Data manipulation
- Matplotlib & Seaborn: Data visualization
- Scikit-learn: Model building and evaluation.
- XGBoost: Advanced boosting techniques

## Analysis Process
### Data Import & Cleaning
- Dataset loaded directly from an external URL.
- Handled potential encoding issues during data loading.
- Cleaned column names to remove whitespace and ensure compatibility.

### Exploratory Data Analysis (EDA)
- Visualized the distribution of the target variable (Bankrupt?).
- Examined relationships between selected financial ratios using scatterplots and pairplots.
- Identified skewness and outliers but preserved original feature values to maintain financial interpretability.

### Feature Engineering
- Created new features, including:
- Gross Profit Ratio: Operating Gross Margin / (Operating Profit Rate + 0.01).
- Adjusted ROA: ROA(C) before interest and depreciation before interest / (1 + Debt ratio %).
- Leverage Ratio: Total debt / Total net worth.
- Liquidity Index: Quick Ratio / Current Ratio.

### Model Building

- Implemented multiple machine learning models:
  - Extra Trees
  - Decision Tree
  - Random Forest
  - XGBoost
  - Gradient Boosting
  - Easy Ensemble
  - CatBoost

- Balanced the dataset using SMOTETomek to address class imbalance in the target variable.
- Evaluated model performance using metrics such as:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC

## Results Summary

- Best Model: XGBoost achieved the best performance, delivering the highest F1-score, which balances precision and recall effectively.

- Metrics for XGBoost:
  - F1-score: Focused on balancing the accuracy of bankruptcy (precision) and non-bankruptcy predictions (recall).
  - Accuracy: Provided a general measure of how well the model performed across all cases.
  - Precision: Ensured the model minimized false positives, reducing unnecessary costs or interventions.
  - Recall: Avoided missing true bankruptcy cases, albeit not the primary focus.

- Key Insights


  1.	Feature Importance:
         - Liquidity Indicators: Ratios such as Quick Ratio and Current Ratio played a significant role in predicting financial health.
         - Profitability Metrics: Features like Operating Profit Rate and ROA (Return on Assets) were critical in distinguishing bankrupt from non-bankrupt companies.
         - Leverage: Excessive debt, as measured by Total debt / Total net worth, was a strong predictor of bankruptcy risk.


  2. Model Selection:
     - XGBoost excelled due to its ability to handle feature interactions and provide robust predictions on imbalanced datasets.
     - Random Forest showed competitive performance but did not achieve the same F1-score as XGBoost.


  3. Balancing Precision and Recall:
      - While recall ensures catching bankruptcies, focusing on F1-score aligns better with scenarios where both false positives and false negatives carry significant costs.
      - The model effectively reduced false positives while maintaining strong recall, achieving a balanced F1-score.


  4. Class Imbalance:
     - Bankruptcy cases were underrepresented in the dataset, necessitating techniques like SMOTETomek for balancing the training data and improving the F1-score.


  5. Visualizations:
      - Feature Importance: Liquidity and profitability ratios emerged as the strongest predictors.
      - Confusion Matrix: Highlighted how the model managed false positives and false negatives, with an emphasis on achieving balance.


## Visualizations

Several visualizations were created to support the analysis:
- Correlation heatmap to identify feature relationships.
- ROC curve to compare model performance.
- Feature importance plot from the XGBoost model.

## Key Observations
- Focus on F1: The project prioritized recall, as correctly predicting bankrupt companies is critical to mitigating financial risks.
- Outliers: Outliers were visually inspected but not removed or capped, ensuring the original data’s interpretability.
- Class Imbalance: Successfully addressed using SMOTETomek to balance training data.

## Code & Outputs
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
![Unknown-7](https://github.com/user-attachments/assets/8cd8967b-4126-4dff-911b-ad4faf85b0de)

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
<img width="486" alt="Screenshot 2024-12-15 at 12 27 27 PM" src="https://github.com/user-attachments/assets/96eafd23-b6cb-4ace-81dc-51ea33d42230" />

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
![Unknown-8](https://github.com/user-attachments/assets/c44b5c1b-9d34-4438-8ac8-674fb86f70dd)

## Contact
- **Name**: Jennie Tran
- **Email**: jennie.tmtran@gmail.com
- **LinkedIn**: [jennietmtran](www.linkedin.com/in/jennietmtran)
- **GitHub**: [jtmtran](https://github.com/jtmtran)
