import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

### Data Preparation ###

# Load the combined dataset
combined_df = pd.read_csv("./combined_dataset.csv")

features = combined_df[["SaO2", "EMG", "NEW AIR", "ABDO RES"]]
labels = combined_df["Apnea/Hypopnea"]

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

#Due to class imbalance, we will balance the dataset using bootstrapping.
# We will also see the effect of balancing on the model performance.

# The original unbalanced dataset
X_train_unbalanced, y_train_unbalanced = X_train, y_train

# Creating a balanced dataset using bootstrapping
# Separate majority and minority classes
train_data = pd.concat([X_train, y_train], axis=1)
majority_class = train_data[train_data["Apnea/Hypopnea"] == 0]
minority_class = train_data[train_data["Apnea/Hypopnea"] == 1]

# Oversample minority class
minority_class_bootstrap = resample(
    minority_class,
    replace=True,
    n_samples=len(majority_class),  # Match majority class size
    random_state=42
)

# Create balanced dataset
balanced_train_data = pd.concat([majority_class, minority_class_bootstrap])
X_train_balanced = balanced_train_data[["SaO2", "EMG", "NEW AIR", "ABDO RES"]]
y_train_balanced = balanced_train_data["Apnea/Hypopnea"]


### Random Forest Classifier ###
# Similar to 10.4. Random forests -> just use the RandomForestClassifier from sklearn
# Play around with number of trees? n_estimators=100
# Random Forest on unbalanced dataset
rf_unbalanced = RandomForestClassifier(random_state=42, n_estimators=100)
rf_unbalanced.fit(X_train_unbalanced, y_train_unbalanced)
rf_preds_unbalanced = rf_unbalanced.predict(X_test)
rf_probs_unbalanced = rf_unbalanced.predict_proba(X_test)[:, 1]

# Random Forest on balanced dataset
rf_balanced = RandomForestClassifier(random_state=42, n_estimators=100)
rf_balanced.fit(X_train_balanced, y_train_balanced)
rf_preds_balanced = rf_balanced.predict(X_test)
rf_probs_balanced = rf_balanced.predict_proba(X_test)[:, 1]

## Evaluation ##
#As accuracy is not a good metric for imbalanced datasets., we will use AUC score and F1-score for comparison.
# Classification reports
print("Unbalanced Dataset:")
print(classification_report(y_test, rf_preds_unbalanced, target_names=["Normal", "Apnea/Hypopnea"]))

print("Balanced Dataset:")
print(classification_report(y_test, rf_preds_balanced, target_names=["Normal", "Apnea/Hypopnea"]))

# AUC Scores
auc_unbalanced = roc_auc_score(y_test, rf_probs_unbalanced)
auc_balanced = roc_auc_score(y_test, rf_probs_balanced)
print(f"AUC for Unbalanced Dataset: {auc_unbalanced:.3f}")
print(f"AUC for Balanced Dataset: {auc_balanced:.3f}")


# F1-scores for comparison
f1_unbalanced = classification_report(y_test, rf_preds_unbalanced, output_dict=True)["weighted avg"]["f1-score"]
f1_balanced = classification_report(y_test, rf_preds_balanced, output_dict=True)["weighted avg"]["f1-score"]
