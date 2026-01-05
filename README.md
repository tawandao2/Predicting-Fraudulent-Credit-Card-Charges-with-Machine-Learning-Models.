# Predicting-Fraudulent-Credit-Card-Charges-with-Machine-Learning-Models.
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

 This gradient-boosted decision tree algorithm was the top performer, achieving an exceptional Recall of 0.9889. It proved to be the most effective at identifying complex,   non-linear fraud patterns while maintaining high operational speed.




2. Random Forest: Achieved a Recall of 0.8610.

<img width="504" height="391" alt="image" src="https://github.com/user-attachments/assets/ade227cd-dbcb-4896-ae7c-86bca92bc5e9" />


This ensemble method, which builds multiple decision trees to improve accuracy, achieved a Recall of 0.8610. While robust, it was less sensitive than XGBoost in         detecting the minority fraud class within the synthetic training environment.




3. Logistic Regression: Served as the baseline with a Recall of 0.7958.
 <img width="486" height="394" alt="image" src="https://github.com/user-attachments/assets/891a2396-4504-432d-b5c2-7a558e528df2" />

 Used as a linear baseline, this model predicts probabilities based on a logistic function and achieved a Recall of 0.7958. The results confirmed that a simple linear approach is insufficient for capturing the nuances of extreme class imbalance.



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

## RECOMMENDATIONS
Based on the high performance of the machine learning models and the specific insights gained from the feature importance analysis, the following actions are recommended for deployment and long-term maintenance:

#### Model Deployment 
Deploy the XGBoost model as the primary detection engine. Its 98.89% recall rate ensures that the vast majority of fraudulent transactions are flagged before they can cause significant financial loss.

Use a real-time scoring API to integrate the model directly into the transaction processing pipeline, allowing for sub-second decision-making.

### Threshold Optimization 
Management should perform a cost-benefit analysis to determine the optimal classification threshold.

Since the current F1-Score (0.2032) indicates high false positives, fine-tuning the probability cutoff can help reduce customer friction (e.g., declined cards for legitimate purchases) while maintaining a high safety net for actual fraud.

### Layered Defense Strategy
Algorithmic detection should be the first line of defense. For transactions flagged as "high risk" by the model, implement automated secondary verification steps: * Multi-Factor Authentication (MFA): Trigger a mobile app push or SMS code for suspicious high-value shopping. * Biometric Prompts: Require a fingerprint or face ID for transactions flagged due to "unreasonable" merchant distance.

### Continuous Model Evolution
Feedback Loops: Establish a pipeline to feed confirmed fraud cases back into the training data. This ensures the model learns from new, emerging fraud tactics—such as shifts in merchant category trends or evolving "card-not-present" techniques.

### Drift Monitoring
Regularly monitor features like "Geographic Distance" and "Transaction Amount." Significant shifts in consumer behavior (e.g., a sudden increase in remote work or international travel) can cause model performance to degrade over time.

### Focus on High-Risk Categories
Given that grocery_pos, shopping_net, and shopping_pos were identified as top-tier predictors (Rank 1 in RFE), the fraud investigation team should prioritize manual audits or stricter verification rules for these specific merchant categories


