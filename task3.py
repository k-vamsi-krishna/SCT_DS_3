import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Import libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load the Dataset ---
# MODIFIED: Using the absolute path provided by the user.
file_path = "C:\\Users\\bharg\\Downloads\\bank+marketing (1)\\bank-additional\\bank-additional\\bank-additional-full.csv"
print(f"--- Loading dataset from {file_path} ---")
try:
    df = pd.read_csv(file_path, delimiter=';')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file {file_path} was not found. Please ensure it's in the correct directory,")
    print("or provide the full, correct path to the file on your system.")
    exit() # Exit if the file isn't found
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()

print("\nSample Data Head:")
print(df.head())
print("\nData Info:")
df.info()
print("\nTarget Variable Distribution (before mapping):")
print(df['y'].value_counts())

# --- 2. Data Visualization (Added Section) ---
print("\n--- Generating Visualizations ---")

# Set aesthetic style of the plots
sns.set_style("whitegrid")

# Plot 1: Distribution of the Target Variable 'y'
plt.figure(figsize=(6, 5))
sns.countplot(x='y', data=df, palette='viridis')
plt.title('Distribution of Term Deposit Subscription (y)')
plt.xlabel('Subscribed to Term Deposit')
plt.ylabel('Count')
plt.show()

# Plot 2: Distribution of 'Job' categories
plt.figure(figsize=(12, 6))
sns.countplot(x='job', data=df, palette='cubehelix', order=df['job'].value_counts().index)
plt.title('Distribution of Job Categories')
plt.xlabel('Job Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
plt.tight_layout()
plt.show()

# Plot 3: Subscription rate by 'Job'
# Temporarily map 'y' to numeric for plotting averages
temp_df_for_plot = df.copy()
temp_df_for_plot['y_numeric'] = temp_df_for_plot['y'].map({'no': 0, 'yes': 1})
plt.figure(figsize=(12, 6))
sns.barplot(x='job', y='y_numeric', data=temp_df_for_plot, palette='mako', order=df['job'].value_counts().index)
plt.title('Subscription Rate by Job Type')
plt.xlabel('Job Type')
plt.ylabel('Subscription Rate (1 = Yes, 0 = No)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot 4: Distribution of 'Education' categories
plt.figure(figsize=(10, 6))
sns.countplot(x='education', data=df, palette='rocket', order=df['education'].value_counts().index)
plt.title('Distribution of Education Levels')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot 5: Subscription rate by 'Education'
plt.figure(figsize=(10, 6))
sns.barplot(x='education', y='y_numeric', data=temp_df_for_plot, palette='crest', order=df['education'].value_counts().index)
plt.title('Subscription Rate by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Subscription Rate (1 = Yes, 0 = No)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot 6: Distribution of 'Marital' status
plt.figure(figsize=(8, 5))
sns.countplot(x='marital', data=df, palette='flare', order=df['marital'].value_counts().index)
plt.title('Distribution of Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

print("Visualizations generated. Please close the plot windows to continue script execution.")


# --- 3. Preprocess the Data (Continuing from previous steps) ---

# Drop the 'duration' column
if 'duration' in df.columns:
    df = df.drop('duration', axis=1)
    print("\nDropped 'duration' column as per dataset notes.")


# Identify categorical and numerical columns for one-hot encoding
categorical_cols = df.select_dtypes(include='object').columns.tolist()
if 'y' in categorical_cols: # Ensure 'y' is not treated as a feature for encoding
    categorical_cols.remove('y')

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns identified: {categorical_cols}")
print(f"Numerical columns identified: {numerical_cols}")

# Convert categorical features into numerical using one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# Convert the target variable 'y' to numerical (0 for 'no', 1 for 'yes')
df_encoded['y'] = df_encoded['y'].map({'no': 0, 'yes': 1})

print("\n--- Data After One-Hot Encoding and Target Mapping (Head): ---")
print(df_encoded.head())
print("\nTarget Variable Distribution (after mapping):")
print(df_encoded['y'].value_counts())

# Separate features (X) and target (y)
X = df_encoded.drop('y', axis=1)
y = df_encoded['y']

print(f"\nShape of features (X): {X.shape}")
print(f"Shape of target (y): {y.shape}")

# --- 4. Split the Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Proportion of 'yes' in training target: {y_train.value_counts(normalize=True)[1]:.4f}")
print(f"Proportion of 'yes' in testing target: {y_test.value_counts(normalize=True)[1]:.4f}")

# --- 5. Initialize and Train the Decision Tree Classifier ---
print("\n--- Training Decision Tree Classifier ---")
dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_classifier.fit(X_train, y_train)
print("Decision Tree Classifier trained successfully.")

# --- 6. Make Predictions ---
y_pred = dt_classifier.predict(X_test)

# --- 7. Evaluate the Model ---
print("\n--- Model Evaluation ---")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- 8. Visualize the Decision Tree (Optional) ---
# To run this part, you need to install graphviz and pydotplus:
# pip install graphviz pydotplus
# You also need to install Graphviz system-wide (e.g., from graphviz.org) and add it to your PATH.

try:
    import graphviz
    import pydotplus

    dot_data = export_graphviz(dt_classifier,
                                feature_names=X.columns,
                                class_names=['No Subscribe', 'Subscribe'], # Class names for 'y'
                                filled=True, rounded=True,
                                special_characters=True,
                                out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png("bank_marketing_decision_tree.png")
    print("\nDecision tree saved as 'bank_marketing_decision_tree.png'.")
    print("You can view this file to see the structure of the decision tree.")
except ImportError:
    print("\nSkipping decision tree visualization.")
    print("To visualize the tree, install 'graphviz' and 'pydotplus' (pip install graphviz pydotplus)")
    print("Also, ensure Graphviz is installed on your system and added to your PATH.")
except Exception as e:
    print(f"\nAn error occurred during decision tree visualization: {e}")
    print("Please ensure Graphviz is correctly installed and accessible in your system's PATH.")

# --- 9. How to use the trained model for new predictions ---
print("\n--- Making a Prediction for a New Customer ---")

# Create a blank DataFrame with all features from the training data
new_customer_data = pd.DataFrame(np.zeros((1, len(X.columns))), columns=X.columns)

# Example new customer data based on common sense from the dataset description
# Numeric features:
new_customer_data['age'] = 40
new_customer_data['campaign'] = 2
new_customer_data['pdays'] = 999
new_customer_data['previous'] = 0
new_customer_data['emp.var.rate'] = 1.1
new_customer_data['cons.price.idx'] = 93.994
new_customer_data['cons.conf.idx'] = -36.4
new_customer_data['euribor3m'] = 4.857
new_customer_data['nr.employed'] = 5191.0

# Categorical features example (set appropriate one-hot encoded columns to 1)
new_customer_data['job_admin.'] = 1
new_customer_data['marital_married'] = 1
new_customer_data['education_university.degree'] = 1
new_customer_data['default_no'] = 1
new_customer_data['housing_yes'] = 1
new_customer_data['loan_no'] = 1
new_customer_data['contact_cellular'] = 1
new_customer_data['month_may'] = 1
new_customer_data['day_of_week_mon'] = 1
new_customer_data['poutcome_nonexistent'] = 1

try:
    new_customer_prediction = dt_classifier.predict(new_customer_data)
    prediction_proba = dt_classifier.predict_proba(new_customer_data)

    if new_customer_prediction[0] == 1:
        print(f"The new customer is predicted to SUBSCRIBE to a term deposit.")
    else:
        print(f"The new customer is predicted NOT to subscribe to a term deposit.")

    print(f"Probability of Not Subscribing: {prediction_proba[0][0]:.4f}")
    print(f"Probability of Subscribing: {prediction_proba[0][1]:.4f}")
except Exception as e:
    print(f"\nError making prediction for new customer: {e}")
    print("Please ensure the 'new_customer_data' DataFrame is correctly structured with all expected features.")