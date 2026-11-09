# Which Fraud-Detection Strategy Works Best Under Severe Class Imbalance?

## Group
Sumit Pradeep Kolpe, Urav Dalal, Mahalakshmi Phaldesai

## Topic
21 — A rare positive class.

## Question
When fraudulent credit-card transactions are extremely rare, undersampling, oversampling, class weighting or classification-threshold adjustment offers the best trade-off between detecting fraud and avoiding false alarms?

## Why it matters
Banks must detect fraud without unnecessarily declining legitimate purchases. Missed fraud can cause customer losses, while false alarms inconvenience customers and create investigation costs. The findings can inform fraud-screening decisions.

## Data
Source: OpenML-CreditCardFraudDetection (dataset ID 42175): https://www.openml.org/d/42175. It contains 284,807 transactions, including 492 fraud cases (about 0.172%, i.e. <1%), 30 numerical inputs (Time, Amount, and V1–V28) and one binary target Class (1 = fraud; 0 = legitimate) so 31 columns in total. No missing values present.

## Baseline
Use Logistic Regression without resampling as the baseline. Tune regularization with stratified five-fold cross-validation on training data. Compare the default 0.5 threshold with alternatives. Section 3D gives the cost-sensitive rule, τ* = c_FP / (c_FP + c_FN), assuming calibrated probabilities and specified error costs.

## Comparison
Compare Gradient Boosting with Logistic Regression. Test no correction, undersampling, oversampling, class weighting, and threshold adjustment. Resample only within training folds to prevent data leakage. Use predeclared false-positive/false-negative cost ratios as hypothetical scenarios, not actual bank cost estimates.

## Evaluation
Use a stratified 80/20 train/test split. Tune models and select thresholds with five-fold cross-validation on training data; use the test set once for final evaluation. Report Average Precision (AP), precision, recall, and cost (c_FP × FP + c_FN × FN). Accuracy alone is not a good measure because fraud is rare. Report 95% bootstrap confidence intervals.

## Key figure
A grouped bar chart of Average Precision by imbalance strategy with separate bars for Logistic Regression and Gradient Boosting and 95% confidence-interval error bars. A table will report precision, recall, cost, and selected threshold.

## Risks and fallback
Few fraud cases may make estimates variable. The cost-sensitive threshold assumes calibrated probabilities. We will state this assumption and compare it with validation-selected thresholds. All resampling and preprocessing will occur within training folds. If Gradient Boosting is impractical we will focus on Logistic Regression and compare the five imbalance strategies.

## Who does what
Sumit — Load and inspect the data, perform descriptive analysis and preprocessing, and implement Logistic Regression.
Urav — Implement Gradient Boosting, undersampling, oversampling, class weighting, and model tuning.
Mahalakshmi — Analyze thresholds and costs, evaluate metrics, prepare figures and tables.
All members will verify results and contribute to the paper and presentation.

## AI plan
Use Claude to refine the question, explain methods, draft and debug code and review the analysis. The team will run and verify experiments, check references, and reproduce results. AI will not be treated as a source of experimental results.
