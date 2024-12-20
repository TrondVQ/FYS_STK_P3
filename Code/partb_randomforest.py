import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from evaluate_model import *

### Data Preparation ###
seed = 2024
# Load the combined dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'combined_dataset.csv')
combined_df = pd.read_csv(file_path)

features = combined_df[["SaO2", "EMG", "NEW AIR", "ABDO RES"]]
labels = combined_df["Apnea/Hypopnea"]
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state = seed)

#Standardization
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


# The original unbalanced dataset
X_train_unbalanced, y_train_unbalanced = X_train, y_train


#  SMOTE
smote = SMOTE(random_state=seed)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


# Grid search
param_grid = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.2, 0.5],
}


# Create the grid search object
random_search_smote = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state = seed),
    param_distributions=param_grid,
    n_iter=100, 
    scoring='roc_auc',  # AUC for imbalanced data
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state = seed),
    verbose=2,
    random_state = seed,
    n_jobs=-1
)

random_search_unbalanced = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state = seed),
    param_distributions=param_grid,
    n_iter=100, 
    scoring='roc_auc',  # AUC for imbalanced data
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state = seed),
    verbose=2,
    random_state = seed,
    n_jobs=-1
)


#start the search
random_search_smote.fit(X_train_smote, y_train_smote)
print("Best Hyperparameters for balanced dataset using SMOTE:", random_search_smote.best_params_)
best_rf_model_smote = random_search_smote.best_estimator_

random_search_unbalanced.fit(X_train_unbalanced, y_train_unbalanced)
print("Best Hyperparameters for unbalanced dataset:", random_search_unbalanced.best_params_)
best_rf_model_unbalanced = random_search_unbalanced.best_estimator_


evaluate_model(best_rf_model_smote, X_test, y_test, "rf", "Hypertuned_smote")
evaluate_model(best_rf_model_unbalanced, X_test, y_test, "rf", "Hypertuned_Unbalanced")

"""
#BASELINE PERFORMANCE CODE:
#Oversampling of the minority class (Random Oversampling ):
train_data = pd.concat([X_train, y_train], axis=1)
majority_class = train_data[train_data["Apnea/Hypopnea"] == 0]
minority_class = train_data[train_data["Apnea/Hypopnea"] == 1]

minority_class_oversampled = resample(
    minority_class,
    replace=True,
    n_samples=len(majority_class),  # Match majority class size
    random_state = seed
)

balanced_train_data = pd.concat([majority_class, minority_class_oversampled])
X_train_balanced = balanced_train_data[["SaO2", "EMG", "NEW AIR", "ABDO RES"]]
y_train_balanced = balanced_train_data["Apnea/Hypopnea"]

##Random undersampling
majority_class_undersampled = resample(
    majority_class,
    replace=False,  # No duplicates
    n_samples=len(minority_class),  # Match minority class size
    random_state=seed
)

undersampled_data = pd.concat([majority_class_undersampled, minority_class])

# Split into features and labels
X_train_undersampled = undersampled_data[["SaO2", "EMG", "NEW AIR", "ABDO RES"]]
y_train_undersampled = undersampled_data["Apnea/Hypopnea"]


### Random Forest Classifier ###
# Similar to 10.4. Random forests -> just use the RandomForestClassifier from sklearn
# Play around with number of trees? n_estimators=100
# Random Forest on unbalanced dataset
rf_unbalanced = RandomForestClassifier(random_state = seed, n_estimators=100)
rf_unbalanced.fit(X_train_unbalanced, y_train_unbalanced)

# Random Forest on balanced dataset (oversampling)
rf_balanced = RandomForestClassifier(random_state = seed, n_estimators=100)
rf_balanced.fit(X_train_balanced, y_train_balanced)
#undersampling

rf_under= RandomForestClassifier(random_state = seed, n_estimators=100)
rf_under.fit(X_train_undersampled, y_train_undersampled)

# smote
rf_smote = RandomForestClassifier(random_state = seed, n_estimators=100)
rf_smote.fit(X_train_smote, y_train_smote)


## Evaluation ##


evaluate_model(rf_unbalanced, X_test, y_test, "rf", "Unbalanced")

evaluate_model(rf_balanced, X_test, y_test,"rf", "Balanced_oversampling")


evaluate_model(rf_under, X_test, y_test, "rf", "Balanced_Undersampling")

evaluate_model(rf_smote, X_test, y_test, "rf", "Balanced_smote")

"""