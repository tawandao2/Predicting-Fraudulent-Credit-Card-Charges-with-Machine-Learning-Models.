# Predicting Fraudulent Credit-Card Charges with Machine Learning-Models.
Detect fraud in a 1.8M record dataset (0.52% fraud rate). This ML project uses SMOTE and feature engineering (temporal/geographic) to tackle class imbalance. Evaluated Logistic Regression, Random Forest, and XGBoost, with XGBoost achieving 98.89% recall. Designed to prioritize catching fraud over accuracy to minimize financial risk.

# EXECUTIVE SUMMARY
This analysis addresses the critical challenge of credit card fraud detection by leveraging machine learning to identify fraudulent transactions within a simulated dataset containing over 1.8 million records. A primary obstacle was a severe class imbalance, with fraudulent cases comprising only 0.52% of the data. To overcome this, the methodology focused on meticulous data cleaning, the extraction of temporal and geographic features, and the application of SMOTE oversampling exclusively to the training set to prevent data leakage while enhancing the model's ability to learn fraud patterns. Three supervised models like Logistic Regression, Random Forest, and XGBoost, were evaluated, with a strategic emphasis on recall rather than simple accuracy to ensure the capture of as many true fraud cases as possible. The final results identified XGBoost as the superior solution, achieving a high recall rate of 98.89%, outperforming the other models in sensitivity and operational speed. While the model effectively prioritizes financial security, limitations such as oversensitivity to false alarms and the need for supplementary behavioral or device-level verification remain key areas for future operational improvement.


# METHODOLOGY
The project implemented a robust machine learning pipeline designed to address the challenges of extreme class imbalance and high-dimensionality data. The systematic approach involved:


## Data Cleaning & Integrity
1. Verified over 1.8 million records for completeness, resulting in zero missing values.

2. Deliberately preserved high-value outliers in the transaction amount (amt) field, as these are primary indicators of fraudulent behavior.

3. Identified and removed duplicate records and redundant index columns to ensure data integrity.

## Feature Engineering


1. Temporal Features: Extracted hour, day, and month from timestamps to identify patterns related to high-risk time windows.


2. Geographic Features: Calculated the physical distance between customers and merchants to flag transactions occurring at "unreasonable" distances.


3. Categorical Transformations: Applied frequency encoding to high-cardinality variables (merchant, job, street, city) to maintain predictive power while reducing complexity.


4. Risk-based Features: Created dummy variables for categories like gender and state to allow models to capture patterns associated with higher-risk demographic tendencies.

## Imbalance Management (SMOTE)

1. To prevent the model from becoming biased toward the 99.5% majority class, SMOTE (Synthetic Minority Over-sampling Technique) was applied.

2. This process was performed exclusively on the training set after a stratified split and scaling numeric features to avoid data leakage.

3. SMOTE generated synthetic fraud instances by creating new points along the lines between real fraud samples, reducing bias toward predicting everything as genuine.

## Model Selection & Training

1. Evaluated three distinct supervised learning architectures: Logistic Regression (baseline), Random Forest, and XGBoost.

2. The strategy prioritized Recall over standard accuracy, ensuring the system is optimized to capture the maximum number of true fraudulent events even at the cost of higher false positives.

## DATASET OVERVIEW
The project utilizes a comprehensive, high-volume dataset designed to simulate real-world financial transaction patterns and fraudulent behaviors.

#### Source & Scope
The data was generated using the Sparkov Data Generation tool, covering a two-year window from January 1, 2019, to December 31, 2020.

It simulates the activities of 1,000 customers across 800 merchants, providing a diverse set of demographic and transactional profiles.

### Volume & Imbalance
The raw dataset consists of 1,852,394 total records.

Class Imbalance: Only 9,651 records (0.52%) are labeled as fraudulent (is_fraud = 1), while 1,842,743 records (99.48%) are legitimate.

This extreme disparity necessitated moving beyond simple accuracy to more granular evaluation metrics.

### Key Features

Transactional: Amount (amt), merchant category, and precise timestamps.

Demographic: Customer age, gender, job title, and residence location.

Geospatial: Latitude and longitude for both the customer and the merchant, enabling the calculation of travel distances.


## RESULTS
The models were evaluated primarily on their ability to capture fraudulent activity (Recall), with the following performance outcomes:

### Model Performance Comparison:

1. XGBoost: The top-performing model, achieving an exceptional Recall of 0.9889.
 <img width="520" height="416" alt="image" src="https://github.com/user-attachments/assets/3ac44eaa-25d0-456c-a2b2-19d2d7045600" />

XGBoost model results. The model achieves extremely high recall (0.992), correctly identifying nearly all fraudulent transactions, with a significant reduction in false negatives compared to previous models.




2. Random Forest: Achieved a Recall of 0.8610.

<img width="504" height="391" alt="image" src="https://github.com/user-attachments/assets/ade227cd-dbcb-4896-ae7c-86bca92bc5e9" />

Random Forest model using a 30% subsample of the SMOTE‑balanced data. The model improves both balanced accuracy and precision compared to Logistic Regression, while maintaining strong recall.”



3. Logistic Regression: Served as the baseline with a Recall of 0.7958.
 <img width="486" height="394" alt="image" src="https://github.com/user-attachments/assets/891a2396-4504-432d-b5c2-7a558e528df2" />

Logistic Regression baseline model. The model achieves strong recall but struggles with precision, correctly identifying most fraud cases while generating many false positives. This establishes a linear benchmark for comparison with more complex models.



### Feature Importance (RFE Ranking):

Using Recursive Feature Elimination (RFE) with the XGBoost estimator, the analysis identified specific variables as Rank 1 (most significant)

SHAP values confirmed that transaction amount (amt) and these merchant categories were the dominant indicators of fraud.
<img width="1273" height="748" alt="image" src="https://github.com/user-attachments/assets/a061df00-57e8-4937-a81e-cb414e928753" />

#### The following features were ranked Rank 1, indicating they are the most significant variables for the model's decision-making:

1. category_grocery_pos: High-frequency, point-of-sale grocery transactions often serve as testing grounds for stolen cards.

2. category_shopping_net: Online shopping patterns showed a high correlation with fraudulent activity due to the lack of physical card presence.

3. category_shopping_pos: Physical retail shopping was identified as a major theater for fraudulent spending.

These results were further cross-referenced with SHAP values, which highlighted that transaction amount (amt) and merchant category were the dominant indicators of fraudulent behavior.

### The Accuracy Paradox:

While all models maintained high balanced accuracy, the F1-Score for XGBoost (0.2032) reflects a high number of false positives. This was a deliberate strategic choice: in fraud detection, the financial and security cost of a "False Negative" (missing actual fraud) is far greater than the operational cost of a "False Positive" (requesting additional verification for a legitimate user).

## Recommendations for Improving Fraud Detection
1. Optimize the Classification Threshold
The XGBoost model achieves extremely high recall but relatively low precision. Adjusting the probability threshold (instead of using the default 0.5) can help balance false positives and false negatives based on business needs. This is especially useful in fraud detection, where the cost of missing fraud is much higher than the cost of flagging a genuine transaction.

2. Introduce a Fraud Risk Scoring System
Instead of producing a simple fraud/not‑fraud label, the model can output a continuous fraud‑risk score. This allows banks to prioritize investigations, apply tiered alerts, and reduce unnecessary manual reviews.

3. Explore Hybrid Resampling Techniques
While SMOTE improved recall significantly, hybrid methods such as SMOTE + Tomek Links or SMOTEENN can help remove noisy synthetic samples and improve precision. These techniques clean the decision boundary and often lead to more stable predictions.

4. Engineer Additional Behavioral Features
Fraud is often behavioral rather than purely transactional. Adding features such as transaction velocity, spending deviation from personal norms, merchant frequency, or distance from previous transactions can further strengthen model performance.

5. Apply Cost‑Sensitive Learning
Fraud detection is inherently cost‑imbalanced. Incorporating cost‑sensitive learning—either through adjusted class weights or custom loss functions—can help the model prioritize fraud cases more effectively without relying solely on resampling.

6. Consider an Ensemble Approach
Each model captures different fraud patterns. Combining Logistic Regression, Random Forest, and XGBoost into a stacked ensemble could improve robustness and reduce model variance, especially in edge‑case transactions.

7. Monitor Model Drift Over Time
Fraud patterns evolve quickly. Implementing regular model retraining, drift detection, and performance monitoring ensures the system remains accurate as new fraud strategies emerge.

8. Integrate Explainability into Deployment
SHAP values provide clear insight into why a transaction was flagged. Including these explanations in dashboards or analyst tools increases trust, transparency, and the usefulness of the model in real‑world fraud investigations.

9. Use Precision–Recall AUC for Evaluation
Given the extreme class imbalance, PR‑AUC is a more informative metric than ROC‑AUC. Incorporating PR‑AUC into model evaluation will give a clearer picture of performance on rare fraud cases.

10. Prepare for Real‑Time Scoring
If deployed in production, the model must operate under strict latency requirements. Ensuring that feature engineering steps (such as distance and time‑based features) can be computed in real time is essential for preventing fraudulent transactions before approval.



