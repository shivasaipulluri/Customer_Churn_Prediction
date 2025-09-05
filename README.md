Customer Churn Prediction — End-to-End Capstone

A complete, notebook-driven churn prediction project that takes you from synthetic data generation to modeling, segmentation, and model packaging. The repo is designed to be portfolio-ready: it shows how to frame the business problem, engineer signal, compare models fairly, and translate results into actions.

TL;DR (results on held-out set of 20,000 customers)

Best global model (by F1): Logistic Regression + engineered interactions → F1 0.3501, Recall 0.4930, Precision 0.2714, Accuracy 0.5027.

Best XGBoost (with class-imbalance fix): F1 0.3205, Recall 0.3931, Precision 0.2705, Accuracy 0.5471.

Segment-wise evaluation shows much stronger pockets: several customer personas hit F1 0.55–0.61, with a small niche segment peaking at F1 0.7692 (see segment tables below).

Note: ROC-AUC values hover around ~0.50 in these runs — a sign that thresholded performance is coming from localized segment signal rather than strong global ranking. This is a great case for segmentation-aware targeting and threshold tuning per segment.

📦 Data & Features

Scale: 100,000 customers (data/features.csv), with a labeled target in data/hidden_churn_labels.csv (positives: 27.17%).

Train/validation sample used in notebooks: 20,000 rows (stratified split inside notebooks).

Base feature set (26 columns) across demographics, account tenure, balances, activity, loans, complaints, credit score, overdraft usage, and loyalty.

Engineered feature sets (all prebuilt as CSVs under /data/):

features_with_behavioral.csv (47 cols) — frequency & intensity style features.

features_with_proxies.csv (92 cols) — “churn-proxy” risk flags and cross-signals.

features_with_composites.csv (112 cols) — composite indices built from behavior & risk.

features_with_composites_normalized.csv (112 cols) — normalized composites for stability.

features_with_segments.csv (67 cols) — adds customer personas for segment-wise modeling.

features_with_interactions.csv — interaction features used in the baseline bump.

Data quality callout: Issue_Resolved is missing for 70,171 rows and is explicitly imputed/fixed in modeling notebooks.

🗂️ Repository Structure
<pre> ```bash /data/ # all ready-to-use feature tables & labels /models/ # saved preprocessors & trained models (.pkl) /notebooks/ # step-by-step project notebooks ``` </pre>


Model artifacts:

models/preprocessor_composites_normalized.pkl

models/xgb_model_with_proxies.pkl

models/xgb_model_with_composites.pkl

models/xgb_model_with_composites_normalized.pkl

models/xgb_model_with_composites_normalized_weighted.pkl

📒 Notebook-by-Notebook Walkthrough (with key results)
01_generate_dataset.ipynb

Build synthetic, business-shaped dataset and persist: features.csv, hidden_churn_labels.csv.

Add behavioral signals and proxy churn flags; export enriched tables.

Outcome: ready-to-model tables with multiple engineered variants and segment tags.

02_baseline_model.ipynb — Logistic Regression

Pipeline: ColumnTransformer(OneHotEncoder + StandardScaler) → LogisticRegression (with class_weight to address imbalance).

Three passes:

Baseline

After fixing Issue Reporting & Resolution (imputes/fixes Issue_Resolved)

After adding explicit feature interactions

Metrics (best pass — with interactions): Accuracy 0.5027, Precision 0.2714, Recall 0.4930, F1 0.3501, ROC-AUC 0.4997.
Earlier passes for reference:
• Baseline — Acc 0.4976, Prec 0.2673, Recall 0.4875, F1 0.3452, AUC 0.4944.
• After Issue fix — Acc 0.4980, Prec 0.2667, Recall 0.4845, F1 0.3440, AUC 0.4937.

03_xgboost_with_behavioral_features.ipynb — XGBoost

Pipeline: same preprocessing → XGBClassifier (with scale_pos_weight variants).

Variants tried:

Behavioral only → (degenerate at threshold 0.5).

Balanced class weighting → Acc 0.5481, Prec 0.2707, Recall 0.3916, F1 0.3201, AUC 0.4990.

Personas → Acc 0.5467, Prec 0.2670, Recall 0.3828, F1 0.3146, AUC 0.4953.

+ Personas (alt run) → Acc 0.5349, Prec 0.2684, Recall 0.4126, F1 0.3253, AUC 0.4966.

+ Behavioral + Personas + Proxy Flags → Acc 0.5434, Prec 0.2665, Recall 0.3885, F1 0.3162, AUC 0.4948.

03_xgboost_with_composites.ipynb — XGBoost + Composite Indices

Adds composite risk & behavior indices.

Metrics: Acc 0.6206, Prec 0.2672, Recall 0.2275, F1 0.2457, AUC 0.4974.

03_xgboost_with_composites_normalized.ipynb — XGBoost + Normalized Composites

Uses normalized composites and class_weight in preprocessing.

Metrics: Acc 0.5625, Prec 0.2697, Recall 0.3574, F1 0.3074, AUC 0.4982.

04_random_forest.ipynb — Random Forest

Pipeline: OHE + scaling → RandomForestClassifier (with class_weight).

Metrics: Acc 0.6222, Prec 0.2770, Recall 0.2425, F1 0.2586, AUC 0.5032.

05_mlp_neural_network.ipynb — Sklearn MLP

Fully connected MLP via sklearn.neural_network.MLPClassifier.

Metrics: Acc 0.7283, Prec 0.0000, Recall 0.0000, F1 0.0000, AUC 0.5000 (collapsed to majority prediction at 0.5 threshold).

3_xgboost_model.ipynb — XGBoost (class-imbalance focus)

Starts from a strong accuracy baseline that fails to recall positives; adds class-imbalance treatment.

Best metrics (with imbalance fix): Acc 0.5471, Prec 0.2705, Recall 0.3931, F1 0.3205, AUC 0.4988.

🏁 Model Leaderboard (held-out 20k rows)
| Notebook                                  | Variant                                                                              | Accuracy | Precision | Recall | F1 Score | ROC-AUC Score |
| ----------------------------------------- | ------------------------------------------------------------------------------------ | -------- | --------- | ------ | -------- | ------------- |
| 02\_baseline\_model                       | Model Performance After Adding Feature Interactions                                  | 0.5027   | 0.2714    | 0.4930 | 0.3501   | 0.4997        |
| 3\_xgboost\_model                         | Model Performance (XGBoost + Class Imbalance Fix)                                    | 0.5471   | 0.2705    | 0.3931 | 0.3205   | 0.4988        |
| 03\_xgboost\_with\_behavioral\_features   | Model Performance (XGBoost + Behavioral + Personas + Proxy Flags)                    | 0.5434   | 0.2665    | 0.3885 | 0.3162   | 0.4948        |
| 03\_xgboost\_with\_composites\_normalized |                                                                                      | 0.5625   | 0.2697    | 0.3574 | 0.3074   | 0.4982        |
| 04\_random\_forest                        | Model Performance (Random Forest)                                                    | 0.6222   | 0.2770    | 0.2425 | 0.2586   | 0.5032        |
| 03\_xgboost\_with\_composites             | Model Performance (XGBoost + Behavioral + Personas + Proxies + Composite Indicators) | 0.6206   | 0.2672    | 0.2275 | 0.2457   | 0.4974        |
| 05\_mlp\_neural\_network                  | Model Performance (MLP)                                                              | 0.7283   | 0.0000    | 0.0000 | 0.0000   | 0.5000        |


👥 Segment-Wise Findings (personas)

Segmentation actually amplifies signal versus a single global threshold. Two complementary tables:

A. Composite persona performance (risk/behavior composites)

| Segment                       | Samples | Precision | Recall | F1 Score |
| ----------------------------- | ------- | --------- | ------ | -------- |
| BalanceFluctuation Risk       | 7307    | 0.5481    | 0.6722 | 0.6038   |
| YoungLowBalance               | 3047    | 0.6005    | 0.5706 | 0.5852   |
| HighOverdraft HighRisk        | 14415   | 0.5861    | 0.5655 | 0.5756   |
| MultipleProducts LowBalance   | 7582    | 0.5322    | 0.5614 | 0.5464   |
| FrequentComplaints Unresolved | 14927   | 0.5610    | 0.5389 | 0.5498   |
| StickyButLowDeposit           | 5970    | 0.5176    | 0.5275 | 0.5225   |
| LowLoyalty HighActivity       | 23787   | 0.5535    | 0.4900 | 0.5198   |
| YoungLowIncome LowBalance     | 2098    | 0.7410    | 0.4786 | 0.5816   |
| DepositDependent IncomeDrop   | 12679   | 0.5517    | 0.4739 | 0.5098   |
| YoungAggressiveSaver          | 2826    | 0.6402    | 0.4621 | 0.5367   |
| HighBalance NoIncomeIncrease  | 10015   | 0.5523    | 0.4586 | 0.5011   |
| OlderLowRiskLowEngagement     | 76      | 1.0000    | 0.4000 | 0.5714   |

B. Global churn model evaluated inside each persona

| Segment                       | Samples | Precision | Recall | F1 Score |
| ----------------------------- | ------- | --------- | ------ | -------- |
| BalanceFluctuation Risk       | 7307    | 0.4758    | 0.8226 | 0.6029   |
| YoungLowBalance               | 3047    | 0.5060    | 0.7153 | 0.5927   |
| HighOverdraft HighRisk        | 14415   | 0.5031    | 0.7149 | 0.5906   |
| MultipleProducts LowBalance   | 7582    | 0.4556    | 0.7094 | 0.5548   |
| StickyButLowDeposit           | 5970    | 0.4675    | 0.6892 | 0.5571   |
| FrequentComplaints Unresolved | 14927   | 0.4944    | 0.6707 | 0.5692   |
| OlderLowRiskLowEngagement     | 76      | 0.9091    | 0.6667 | 0.7692   |
| LowLoyalty HighActivity       | 23787   | 0.4752    | 0.6606 | 0.5528   |
| DepositDependent IncomeDrop   | 12679   | 0.4449    | 0.6564 | 0.5303   |
| HighBalance NoIncomeIncrease  | 10015   | 0.4768    | 0.6376 | 0.5456   |
| YoungAggressiveSaver          | 2826    | 0.5460    | 0.6085 | 0.5756   |
| YoungLowIncome LowBalance     | 2098    | 0.6133    | 0.6068 | 0.6100   |

Highlights

Top performing persona: OlderLowRiskLowEngagement — F1 0.7692 (small niche, 76 samples).

Consistently strong, high-volume personas:
• BalanceFluctuation Risk — F1 0.6038–0.6029 across tables (7,307 samples).
• YoungLowBalance — F1 0.5852–0.5927 (3,047 samples).
• HighOverdraft HighRisk — F1 0.5756–0.5906 (14,415 samples).

Takeaway: A segment-aware playbook (different thresholds/offers per persona) is likely to outperform a single global model in this dataset.

🧰 Tech Stack

Python (pandas, numpy)

Modeling: scikit-learn (LogisticRegression, RandomForestClassifier, MLPClassifier), XGBoost

Preprocessing: ColumnTransformer, OneHotEncoder, StandardScaler

Imbalance: class_weight, scale_pos_weight (XGBoost)

Explainability/EDA: seaborn, matplotlib, SHAP (imports present)

Persistence: joblib (.pkl artifacts)

▶️ Reproducibility — How to Run

Clone the repo and open /notebooks/ in Jupyter or VS Code.

Ensure dependencies (Python 3.10+ recommended):
pip install -r requirements.txt  # or install: pandas numpy scikit-learn xgboost seaborn matplotlib shap joblib faker

(Optional) Start from data generation: run 01_generate_dataset.ipynb to rebuild /data/ artifacts.

Run modeling notebooks in order (02 → 05) to reproduce metrics and save models into /models/.

For persona insights, run segment_composite_analysis.ipynb and segment_wise_analysis.ipynb.

Tip: Thresholds are currently evaluated at 0.5. If you care about recall (typical for churn), tune thresholds per persona on validation data for large gains.

📈 Business Interpretation & Next Steps

Global model reliably recovers ~49% of churners at ~27% precision — enough to drive targeted retention offers with acceptable cost per save if offer cost is modest.

Persona targeting outperforms global in both F1 and recall across several sizable groups — prioritize these for campaigns.

What to do next

Threshold tuning per persona to hit business-specific precision/recall trade-offs.

Calibrate probabilities (Platt/Isotonic) and re-evaluate AUC/PR-AUC.

Cost-sensitive learning: incorporate explicit save-cost vs. CLV uplift into training/selection.

Temporal validation: simulate month-over-month churn and evaluate model drift.

Feature audit: promote proxy/composite features that meaningfully move recall without hurting precision.

Last updated: 2025-09-04
Author: Shiva Sai Pulluri — open to opportunities in Data Science / ML Engineering.
