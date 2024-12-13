import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from project2_classes_and_functions import *
from project3_functions import *
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

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

y_test = y_test.values.reshape(-1,1)

#Standardization
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

#Due to class imbalance, we will balance the dataset using .
# We will also see the effect of balancing on the model performance.

# The original unbalanced dataset -> These are converted to numpy. also y train now is also 2d
X_train_unbalanced, y_train_unbalanced = X_train.values, y_train.values.reshape(-1,1)

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
y_train_balanced = balanced_train_data["Apnea/Hypopnea"].values.reshape(-1,1)


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
y_train_undersampled = undersampled_data["Apnea/Hypopnea"].values.reshape(-1,1)


from imblearn.over_sampling import SMOTE

# Initialize SMOTE
smote = SMOTE(random_state=seed)

# Apply SMOTE to training data. smote returns numpy array
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
y_train_smote = y_train_smote.values.reshape(-1,1)

# Project 2 FFNN implementation -> use the best architechture there even though it is a different dataset. classification


from sklearn.model_selection import GridSearchCV

# Grid search
#Based on this article: https://pmc.ncbi.nlm.nih.gov/articles/PMC11567982/
param_grid = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.2, 0.5],
}

balanced_nn_SMOTE = NetworkClass()

unbalanced_nn = NetworkClass(
    cost_fun=CostCrossEntropy,
    cost_der=CostCrossEntropyDer,
    network_input_size=X_train_balanced.shape[1],
    layer_output_sizes=[50, 1],
    activation_funcs=[ReLU, sigmoid],
    activation_ders=[ReLU_der, sigmoid_derivative]
)


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

### Evaluation ###

evaluate_model(balanced_nn_SMOTE, X_test, y_test, "ffnn", "Balanced - SMOTE")

# Evaluate the RNN trained on unbalanced data
evaluate_model(unbalanced_nn, X_test, y_test, "ffnn", "Unbalanced")


""" 
#Baseline
balanced_nn_SMOTE = NetworkClass(
    cost_fun=CostCrossEntropy,
    cost_der=CostCrossEntropyDer,
    network_input_size=X_train_balanced.shape[1],
    layer_output_sizes=[50, 1],
    activation_funcs=[ReLU, sigmoid],
    activation_ders=[ReLU_der, sigmoid_derivative]
)

unbalanced_nn = NetworkClass(
    cost_fun=CostCrossEntropy,
    cost_der=CostCrossEntropyDer,
    network_input_size=X_train_balanced.shape[1],
    layer_output_sizes=[50, 1],
    activation_funcs=[ReLU, sigmoid],
    activation_ders=[ReLU_der, sigmoid_derivative]
)

# Train the networks. NEED TO TUNE THE PARAMETERS


balanced_nn_SMOTE.train(
    X_train_balanced,
    y_train_balanced,
    epochs=100,
    batch_size=32,
    learning_rate=0.01,
    lmbd=0.001
)


unbalanced_nn.train(
    X_train_unbalanced,
    y_train_unbalanced,
    epochs=100,
    batch_size=32,
    learning_rate=0.01,
    lmbd=0.001
)
balanced_nn_OS = NetworkClass(
    cost_fun=CostCrossEntropy,
    cost_der=CostCrossEntropyDer,
    network_input_size=X_train_balanced.shape[1],
    layer_output_sizes=[50, 1],
    activation_funcs=[ReLU, sigmoid],
    activation_ders=[ReLU_der, sigmoid_derivative]
)

balanced_nn_US = NetworkClass(
    cost_fun=CostCrossEntropy,
    cost_der=CostCrossEntropyDer,
    network_input_size=X_train_balanced.shape[1],
    layer_output_sizes=[50, 1],
    activation_funcs=[ReLU, sigmoid],
    activation_ders=[ReLU_der, sigmoid_derivative]
)

balanced_nn_OS.train(
    X_train_balanced,
    y_train_balanced,
    epochs=100,
    batch_size=32,
    learning_rate=0.01,
    lmbd=0.001
)

balanced_nn_US.train(
    X_train_balanced,
    y_train_balanced,
    epochs=100,
    batch_size=32,
    learning_rate=0.01,
    lmbd=0.001
)

# Evaluate the RNN trained on balanced data
evaluate_model(balanced_nn_OS, X_test, y_test, "ffnn", "Balanced - OS")

evaluate_model(balanced_nn_US, X_test, y_test, "ffnn", "Balanced - US")


"""