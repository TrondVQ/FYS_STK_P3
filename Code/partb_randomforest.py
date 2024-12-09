import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  classification_report, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.utils import resample
import matplotlib.pyplot as plt

### Data Preparation ### -> maybe a function? however needs to be different for CNN

# Load the combined dataset
combined_df = pd.read_csv("./combined_dataset.csv")

features = combined_df[["SaO2", "EMG", "NEW AIR", "ABDO RES"]]
labels = combined_df["Apnea/Hypopnea"]

seed = 2024

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state = seed)

#Standardization -> combineddf is a 2d df
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

#Due to class imbalance, we will balance the dataset using oversample the minority class.

# The original unbalanced dataset
X_train_unbalanced, y_train_unbalanced = X_train, y_train

#Oversampling of the minority class:
train_data = pd.concat([X_train, y_train], axis=1)
majority_class = train_data[train_data["Apnea/Hypopnea"] == 0]
minority_class = train_data[train_data["Apnea/Hypopnea"] == 1]

# Oversample minority class
minority_class_oversampled = resample(
    minority_class,
    replace=True,
    n_samples=len(majority_class),  # Match majority class size
    random_state = seed
)

# Create balanced dataset
balanced_train_data = pd.concat([majority_class, minority_class_oversampled])
X_train_balanced = balanced_train_data[["SaO2", "EMG", "NEW AIR", "ABDO RES"]]
y_train_balanced = balanced_train_data["Apnea/Hypopnea"]

from sklearn.model_selection import GridSearchCV

# Grid search
#Based on this article: https://pmc.ncbi.nlm.nih.gov/articles/PMC11567982/
param_grid = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['auto', 'sqrt', 'log2', 0.2, 0.5],
}
""" 
unbalanced_param = {
    'n_estimators': [50, 100, 150, 200, 300, 500, 1000],
    'max_depth': [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    #'class_weight': [None, 'balanced', "balanced_subsample"]
}
"""


# Create the grid search object
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state = seed),
    param_distributions=param_grid,
    n_iter=100,  # Increased iterations
    scoring='roc_auc',  # AUC for imbalanced data
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state = seed),
    verbose=2,
    random_state = seed,
    n_jobs=-1
)
"""
random_search_unbalanced = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state = seed),
    param_distributions=unbalanced_param,
    n_iter=50,
    scoring='f1',
    cv=3,  # 3-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit the grid search to the balanced training data
random_search.fit(X_train_balanced, y_train_balanced)

# Best parameters and model
print("Best Hyperparameters:", random_search.best_params_)
best_rf_model = random_search.best_estimator_

# Fit the grid search to the balanced training data
random_search_unbalanced.fit(X_train_unbalanced, y_train_unbalanced)

# Best parameters and model
print("Best Hyperparameters:", random_search_unbalanced.best_params_)
best_rf_model_unbalanced = random_search.best_estimator_


"""

### Random Forest Classifier ###
# Similar to 10.4. Random forests -> just use the RandomForestClassifier from sklearn
# Play around with number of trees? n_estimators=100
# Random Forest on unbalanced dataset
rf_unbalanced = RandomForestClassifier(random_state = seed, n_estimators=100)
rf_unbalanced.fit(X_train_unbalanced, y_train_unbalanced)
rf_preds_unbalanced = rf_unbalanced.predict(X_test)
rf_probs_unbalanced = rf_unbalanced.predict_proba(X_test)[:, 1]

# Random Forest on balanced dataset
rf_balanced = RandomForestClassifier(random_state = seed, n_estimators=100)
rf_balanced.fit(X_train_balanced, y_train_balanced)
rf_preds_balanced = rf_balanced.predict(X_test)
rf_probs_balanced = rf_balanced.predict_proba(X_test)[:, 1]

## Evaluation ##

def evaluate_model(model,  X_test, y_test, model_type = "NN", dataset_balance = "balanced" ):
    print(f"Evaluation for {dataset_balance} dataset and model {model_type}:")

    #if random_forest
    if(model_type == "rf"):
        y_pred_probs = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_probs = model.predict(X_test)

    y_pred = (y_pred_probs > 0.5).astype(int)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Apnea/Hypopnea"]))

    r_auc_score = roc_auc_score(y_test, y_pred_probs)
    print(f"ROC AUC Score: {r_auc_score:.4f}")

# Evaluate the RNN trained on unbalanced data
evaluate_model(rf_unbalanced, X_test, y_test, "Unbalanced")

feature_importances = rf_unbalanced.feature_importances_
feature_names = ["SaO2", "EMG", "NEW AIR", "ABDO RES"]

# Plot feature importance
plt.barh(feature_names, feature_importances, color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance for Random Forest')
plt.show()

# Print feature importances
for feature, importance in zip(feature_names, feature_importances):
    print(f"{feature}: {importance:.4f}")


# Evaluate the RNN trained on balanced data
evaluate_model(rf_balanced, X_test, y_test, "Balanced")

feature_importances = rf_balanced.feature_importances_
feature_names = ["SaO2", "EMG", "NEW AIR", "ABDO RES"]

# Plot feature importance
plt.barh(feature_names, feature_importances, color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance for Random Forest')
plt.show()

# Print feature importances
for feature, importance in zip(feature_names, feature_importances):
    print(f"{feature}: {importance:.4f}")
