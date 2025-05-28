# SCT_DS_3
# 📊 Bank Marketing Analysis and Decision Tree Classification

This project analyzes the **Bank Marketing dataset** from a Portuguese banking institution and builds a **Decision Tree Classifier** to predict whether a client will subscribe to a term deposit.

---

## 📁 Dataset

- Source: [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- File Used: `bank-additional-full.csv`
- Separator: `;` (semicolon)
- Target Variable: `y` (binary: yes/no)

---

## ✅ Project Workflow

### 1. Data Loading
- Dataset is loaded using pandas with error handling.
- Initial structure and sample rows are displayed.

### 2. Data Visualization
We generated multiple plots to understand the data better:
- 📌 Distribution of Target Variable (`y`)
- 🧑‍💼 Job Type Distribution & Subscription Rate
- 🎓 Education Level Distribution & Subscription Rate
- 💍 Marital Status Distribution

### 3. Preprocessing
- Dropped the `duration` column (as recommended).
- Applied **one-hot encoding** to categorical variables.
- Mapped target variable `y` → {`no`: 0, `yes`: 1}.

### 4. Model Training
- Used **DecisionTreeClassifier** from scikit-learn.
- Set `max_depth=5` for better generalization.
- Trained using `train_test_split` with `stratify=y`.

### 5. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report

### 6. Visualization
- Decision Tree exported as PNG.
- Report compiled into a downloadable **PDF** with all plots and metrics.

### 7. New Customer Prediction
A mock customer data row was created and passed to the trained model to demonstrate inference and prediction probability.

---

## 📂 Output Files

- `bank_marketing_report_complete.pdf`: Full report with metrics and plots.
- `output_log.txt`: Contains model evaluation summary and classification report.

---

## 🛠️ Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn fpdf
