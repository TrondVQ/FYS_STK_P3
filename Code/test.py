import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from scipy.stats import randint, uniform
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score
import os
import tensorflow as tf
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

seed = 2024

# Load the combined dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'combined_dataset.csv')
combined_df = pd.read_csv(file_path)

# Feature and label selection
features = combined_df[["SaO2", "EMG", "NEW AIR", "ABDO RES"]]
labels = combined_df["Apnea/Hypopnea"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=seed)

# Standardize the features
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Fixed parameters
timesteps = 30
num_features = 4

# Function to create sliding windows
def create_sliding_windows(data, labels, window_size):
    X, y = [], []
    for i in range(len(data) - window_size + 1):
        X.append(data[i:i + window_size])
        y.append(labels[i + window_size - 1])
    return np.array(X), np.array(y)

# Prepare training data for oversampling and undersampling
train_data = pd.concat([X_train, y_train], axis=1)
majority_class = train_data[train_data["Apnea/Hypopnea"] == 0]
minority_class = train_data[train_data["Apnea/Hypopnea"] == 1]

# Oversample minority class
minority_class_OS = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=seed)
balanced_train_data = pd.concat([majority_class, minority_class_OS])
X_train_balanced_OS = balanced_train_data[["SaO2", "EMG", "NEW AIR", "ABDO RES"]].values
y_train_balanced_OS = balanced_train_data["Apnea/Hypopnea"].values.reshape(-1, 1)
X_train_balanced_OS, y_train_balanced_OS = create_sliding_windows(X_train_balanced_OS, y_train_balanced_OS, timesteps)

# Undersample majority class
majority_class_undersampled = resample(majority_class, replace=False, n_samples=len(minority_class), random_state=seed)
undersampled_data = pd.concat([majority_class_undersampled, minority_class])
X_train_undersampled = undersampled_data[["SaO2", "EMG", "NEW AIR", "ABDO RES"]].values
y_train_undersampled = undersampled_data["Apnea/Hypopnea"].values.reshape(-1, 1)
X_train_undersampled, y_train_undersampled = create_sliding_windows(X_train_undersampled, y_train_undersampled, timesteps)

# Apply SMOTE
smote = SMOTE(random_state=seed)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
X_train_smote, y_train_smote = create_sliding_windows(X_train_smote.values, y_train_smote.values, timesteps)

# Create sliding windows for unbalanced data
X_train_unbalanced, y_train_unbalanced = create_sliding_windows(X_train.values, y_train.values, timesteps)
X_test, y_test = create_sliding_windows(X_test.values, y_test.values, timesteps)

# CNN model definition
def create_cnn(time_steps, num_features, filters1=32, filters2=64, kernel_size=3, pool_size=2, alpha=0.0001):
    model = Sequential()
    model.add(Input(shape=(time_steps, num_features)))  
    model.add(Conv1D(filters=filters1, kernel_size=kernel_size, activation='relu', kernel_regularizer=l2(alpha)))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(filters=filters2, kernel_size=kernel_size, activation='relu', kernel_regularizer=l2(alpha)))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

# Wrap model for RandomizedSearchCV
model_unbalanced = KerasClassifier(
    model=create_cnn,
    time_steps=timesteps,
    num_features=num_features,
    filters1=None, 
    filters2=None,
    kernel_size=None,
    pool_size=None,
    alpha=None,     
    verbose=0
)

model_smote = KerasClassifier(
    model=create_cnn,
    time_steps=timesteps,
    num_features=num_features,
    filters1=None, 
    filters2=None,
    kernel_size=None,
    pool_size=None,
    alpha=None,    
    verbose=0
)

print(model_smote.get_params().keys())

# Parameter grid for RandomizedSearchCV
param_grid = {
    'filters1': [32, 64, 128],
    'filters2': [64, 128],
    'kernel_size': [3, 5, 7],
    'pool_size': [2],
    'epochs': randint(30, 50), #time constrains
    'alpha': uniform(0.0001, 0.01),
    'batch_size': [128, 256, 512] #batch size 16 took too long about 20 min per. Also based on FFNN, larger batch sized performed better
}

# Randomized search for SMOTE - this took 3h
random_search_smote = RandomizedSearchCV(
    estimator=model_smote,
    param_distributions=param_grid,
    n_iter=20, # time constrains
    scoring='roc_auc',
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
    verbose=2,
    random_state=seed,
    n_jobs=-1
)

# Randomized search for unbalanced
random_search_unbalanced = RandomizedSearchCV(
    estimator=model_unbalanced,
    param_distributions=param_grid,
    n_iter=20,
    scoring='roc_auc',
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
    verbose=2,
    random_state=seed,
    n_jobs=-1
)

print("Starting search!")

# Train and find best models
random_search_smote.fit(X_train_smote, y_train_smote)
print("Best Hyperparameters for balanced dataset using SMOTE:", random_search_smote.best_params_)
best_CNN_model_smote = random_search_smote.best_estimator_

random_search_unbalanced.fit(X_train_unbalanced, y_train_unbalanced)
print("Best Hyperparameters for unbalanced dataset:", random_search_unbalanced.best_params_)
best_CNN_model_unbalanced = random_search_unbalanced.best_estimator_

# Evaluation function
def evaluate_model(model, X_test, y_test, model_type="NN", dataset_balance="balanced"):
    print(f"Evaluation for {dataset_balance} dataset and model {model_type}:")
    y_pred_probs = model.predict(X_test)[:, 0]
    y_pred = (y_pred_probs > 0.5).astype(int)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Apnea/Hypopnea"]))
    r_auc_score = roc_auc_score(y_test, y_pred_probs)
    print(f"ROC AUC Score: {r_auc_score:.4f}")

# Evaluate models
evaluate_model(best_CNN_model_smote, X_test, y_test, "cnn", "Balanced - SMOTE")
evaluate_model(best_CNN_model_unbalanced, X_test, y_test, "cnn", "Unbalanced")
