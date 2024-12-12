import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from project2_classes_and_functions import *
from project3_functions import *
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from scipy.stats import uniform, randint
from sklearn.neural_network import MLPClassifier
import numpy as np
from imblearn.over_sampling import SMOTE

### Data Preparation ### About the same as in partb_randomforest.py, but with numpy arrays

#HMM er jo 1d, behandler jeg denne som 2d? dobbelsjekk

seed = 2024
# Load the combined dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'combined_dataset.csv')
combined_df = pd.read_csv(file_path)

features = combined_df[["SaO2", "EMG", "NEW AIR", "ABDO RES"]]
labels = combined_df["Apnea/Hypopnea"]
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state = seed)

y_test = y_test.values

#Standardization
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

#Due to class imbalance, we will balance the dataset using .
# We will also see the effect of balancing on the model performance.

# The original unbalanced dataset -> These are converted to numpy. also y train now is also 2d
X_train_unbalanced, y_train_unbalanced = X_train.values, y_train.values

# Creating a balanced dataset using random oversampling
# Separate majority and minority classes
train_data = pd.concat([X_train, y_train], axis=1)
majority_class = train_data[train_data["Apnea/Hypopnea"] == 0]
minority_class = train_data[train_data["Apnea/Hypopnea"] == 1]

# Oversample minority class
minority_class_OS = resample(
    minority_class,
    replace=True,
    n_samples=len(majority_class),  # Match majority class size
    random_state=seed
)

# Create balanced dataset
balanced_train_data = pd.concat([majority_class, minority_class_OS])
#These are converted to numpy
X_train_balanced = balanced_train_data[["SaO2", "EMG", "NEW AIR", "ABDO RES"]].values
y_train_balanced = balanced_train_data["Apnea/Hypopnea"].values


##Random undersampling
majority_class_undersampled = resample(
    majority_class,
    replace=False,  # No duplicates
    n_samples=len(minority_class),  # Match minority class size
    random_state=seed
)

# Combine undersampled majority with minority class
undersampled_data = pd.concat([majority_class_undersampled, minority_class])

# Split into features and labels
X_train_undersampled = undersampled_data[["SaO2", "EMG", "NEW AIR", "ABDO RES"]].values
y_train_undersampled = undersampled_data["Apnea/Hypopnea"].values




# Initialize SMOTE
smote = SMOTE(random_state=seed)

# Apply SMOTE to training data. smote returns numpy array
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
y_train_smote = y_train_smote.values





# Grid search
#Based on this article: https://pmc.ncbi.nlm.nih.gov/articles/PMC11567982/
#From the previous project: . A hyperparameter grid search showed that higher learning rates (η) and lower regularization values (λ) generally led to better results. + leaky relu (tho cant test this here)
#using the same rang as from project 2
#Source: https://sklearner.com/scikit-learn-random-search-mlpclassifier/

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,), (200,), (50, 50), (100, 100)],
    'activation': ['relu', 'logistic', 'tanh'],
    'learning_rate_init': uniform(0.0001, 0.01),
    'alpha': uniform(0.0001, 0.01),
    'max_iter': randint(300, 1000), # we know from the baseline that it needs more than 200 to converge
    'batch_size': [16, 32, 64, 128]
}

# Create the grid search object
random_search_smote = RandomizedSearchCV(
    estimator= MLPClassifier(random_state=seed, max_iter = 500), #setting max iter here incase
    param_distributions=param_grid,
    n_iter=100, 
    scoring='roc_auc',  # AUC for imbalanced data
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state = seed),
    verbose=2,
    random_state = seed,
    n_jobs=-1
)

random_search_unbalanced = RandomizedSearchCV(
    estimator= MLPClassifier(random_state=seed, max_iter = 500),
    param_distributions=param_grid,
    n_iter=100, 
    scoring='roc_auc',  # AUC for imbalanced data
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state = seed),
    verbose=2,
    random_state = seed,
    n_jobs=-1
)

# Fit the grid search to the balanced training data
##random_search_smote.fit(X_train_smote, y_train_smote)

# Best parameters and model
#print("Best Hyperparameters for balanced dataset using SMOTE:", random_search_smote.best_params_)
#best_FFNN_model_smote = random_search_smote.best_estimator_

# Fit the grid search to the balanced training data
random_search_unbalanced.fit(X_train_unbalanced, y_train_unbalanced)

# Best parameters and model
print("Best Hyperparameters for unbalanced dataset:", random_search_unbalanced.best_params_)
best_FFNN_model_unbalanced = random_search_unbalanced.best_estimator_

### Evaluation ###

#evaluate_model(best_FFNN_model_smote, X_test, y_test, "ffnn", "Balanced - SMOTE")

# Evaluate the RNN trained on unbalanced data
evaluate_model(best_FFNN_model_unbalanced, X_test, y_test, "ffnn", "Unbalanced")


""" 
#Baseline

balanced_nn_SMOTE.fit(
    X_train_balanced,
    y_train_balanced
)


unbalanced_nn.fit(
    X_train_unbalanced,
    y_train_unbalanced
)



balanced_nn_OS.fit(
    X_train_balanced,
    y_train_balanced
)

balanced_nn_US.fit(
    X_train_balanced,
    y_train_balanced
)

evaluate_model(balanced_nn_OS, X_test, y_test, "ffnn", "Balanced - OS")

evaluate_model(balanced_nn_US, X_test, y_test, "ffnn", "Balanced - US")
unbalanced_nn = MLPClassifier(random_state=seed)
balanced_nn_OS =MLPClassifier(random_state=seed)

"""
