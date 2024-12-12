import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
import os
from project3_functions import *
from scipy.stats import randint, uniform
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import uniform, randint
from imblearn.over_sampling import SMOTE

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

timesteps = 30  # Assuming each 30-second window corresponds to 30 time steps
num_features = 4  # SaO2, EMG, NEW AIR, ABDO RES


def create_sliding_windows(data, labels, window_size):
    X, y = [], []
    for i in range(len(data) - window_size + 1):
        X.append(data[i:i + window_size])
        y.append(labels[i + window_size - 1])
    return np.array(X), np.array(y)

#CNN ->asked chatgpt to help with this:



train_data = pd.concat([X_train, y_train], axis=1)
#train_data = pd.DataFrame(X_train.reshape(X_train.shape[0], -1))  # Flatten each window (if needed)
#train_data["Apnea/Hypopnea"] = y_train

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
#X_train_balanced_OS = balanced_train_data.drop(columns=["Apnea/Hypopnea"]).values
#y_train_balanced_OS = balanced_train_data["Apnea/Hypopnea"].values
X_train_balanced_OS = balanced_train_data[["SaO2", "EMG", "NEW AIR", "ABDO RES"]].values
y_train_balanced_OS = balanced_train_data["Apnea/Hypopnea"].values.reshape(-1, 1)
X_train_balanced_OS, y_train_balanced_OS = create_sliding_windows(X_train_balanced_OS, y_train_balanced_OS, timesteps)


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
y_train_undersampled = undersampled_data["Apnea/Hypopnea"].values.reshape(-1, 1)
X_train_undersampled, y_train_undersampled = create_sliding_windows(X_train_undersampled, y_train_undersampled, timesteps)


from imblearn.over_sampling import SMOTE

# Initialize SMOTE
smote = SMOTE(random_state=seed)

# Apply SMOTE to training data. smote returns numpy array
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
X_train_smote, y_train_smote = create_sliding_windows(X_train_smote.values, y_train_smote.values, timesteps)
#y_train_smote = y_train_smote.values.reshape(-1, 1)



X_train, y_train = create_sliding_windows(X_train.values, y_train.values, timesteps)
X_test, y_test = create_sliding_windows(X_test.values, y_test.values, timesteps)

print("X_train shape:", X_train.shape)  # (num_windows, window_size, num_features)
print("y_train shape:", y_train.shape)  # (num_windows,)

X_train_unbalanced = X_train
y_train_unbalanced = y_train

""" 
#CNN baseline adjusted for random search. Archiecture is the same
def create_cnn(time_steps, num_features, filters1=32, filters2=64, kernel_size=3, pool_size=2, alpha=0.0001):
    model = Sequential()
    model.add(Conv1D(filters=filters1, kernel_size=kernel_size, activation='relu', input_shape=(time_steps, num_features), kernel_regularizer=l2(alpha)))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(filters=filters2, kernel_size=kernel_size, activation='relu', kernel_regularizer=l2(alpha)))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(filters=filters2, kernel_size=kernel_size, activation='relu', kernel_regularizer=l2(alpha)))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

#for it to be able to be used for ranfom search in scikit learn
model_unbalanced = KerasClassifier(build_fn=create_cnn, time_steps=30, num_features=4, verbose=0)
model_smote = KerasClassifier(build_fn=create_cnn, time_steps=30, num_features=4, verbose=0)


param_grid = {
    'filters1': [32, 64, 128],
    'filters2': [64, 128],
    'kernel_size': [3, 5, 7],
    'pool_size': [2, 3],
    'epochs ': randint(10, 50),
    'alpha': uniform(0.0001, 0.01), # L2 regularization same as FFNN
    'batch_size': [16, 32, 64, 128]
}


random_search_smote = RandomizedSearchCV(
    estimator= model_unbalanced, 
    param_distributions=param_grid,
    n_iter=30, #due to CNN being more complex
    scoring='roc_auc',  # AUC for imbalanced data
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state = seed),
    verbose=2,
    random_state = seed,
    n_jobs=-1
)

random_search_unbalanced = RandomizedSearchCV(
    estimator= model_smote,
    param_distributions=param_grid,
    n_iter=30,  #due to CNN being more complex
    scoring='roc_auc',  # AUC for imbalanced data
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state = seed),
    verbose=2,
    random_state = seed,
    n_jobs=-1
)



# Fit the grid search to the balanced training data
random_search_smote.fit(X_train_smote, y_train_smote)

# Best parameters and model
print("Best Hyperparameters for balanced dataset using SMOTE:", random_search_smote.best_params_)
best_CNN_model_smote = random_search_smote.best_estimator_

# Fit the grid search to the balanced training data
random_search_unbalanced.fit(X_train_unbalanced, y_train_unbalanced)

# Best parameters and model
print("Best Hyperparameters for unbalanced dataset:", random_search_unbalanced.best_params_)
best_CNN_model_unbalanced = random_search_unbalanced.best_estimator_

### Evaluation ###

evaluate_model(best_CNN_model_smote, X_test, y_test, "cnn", "Balanced - SMOTE")

evaluate_model(best_CNN_model_unbalanced, X_test, y_test, "cnn", "Unbalanced")
"""


# CNN implementation similar to 16.8. The CIFAR01 data set, but 1d -> using the same architecutre as basline
# with L2 as with FFNN using their default parameters
def create_cnn(time_steps, num_features):
    model = Sequential()
    #The Input layer -> 1d convolutional layer
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu',  input_shape=(time_steps, num_features), kernel_regularizer=l2(0.0001)))

    # Pooling Layer -> 1d
    model.add(MaxPooling1D(pool_size=2))


    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'), kernel_regularizer=l2(0.0001))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'),  kernel_regularizer=l2(0.0001))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    #  Last fully connected layer
  #  model.add(Dropout(0.2))  # Dropout for regularization
    # Binary classification  so one output and sigmoid
    model.add(Dense(units=1, activation='sigmoid'))  

    # Compile Model -> use AUC due to unbalanced dataset. Binary classification, therefore binary_cross entropht
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

#Baseline test
balancedOS_cnn = create_cnn(timesteps, num_features)
balancedOS_cnn.summary()
balancedOS_cnn.fit(X_train_balanced_OS, y_train_balanced_OS, epochs=20, batch_size=32,
                                         validation_data=(X_test, y_test))

balancedUS_cnn = create_cnn(timesteps, num_features)
balancedUS_cnn.summary()
balancedUS_cnn.fit(X_train_undersampled, y_train_undersampled, epochs=20, batch_size=32,
                                         validation_data=(X_test, y_test))

balancedSMOTE_cnn= create_cnn(timesteps, num_features)
balancedSMOTE_cnn.summary()
balancedSMOTE_cnn.fit(X_train_smote, y_train_smote, epochs=20, batch_size=32,
                                         validation_data=(X_test, y_test))

# Unbalanced model
unbalanced_cnn = create_cnn(timesteps, num_features)
unbalanced_cnn.summary()
unbalanced_cnn.fit(X_train_unbalanced, y_train_unbalanced, epochs=20, batch_size=32,
                                         validation_data=(X_test, y_test))


### Evaluation ###

# Evaluate the CNN trained on unbalanced data
evaluate_model(unbalanced_cnn, X_test, y_test, "CNN", "Unbalanced")

evaluate_model(balancedOS_cnn, X_test, y_test, "CNN", "Balanced - OS")

evaluate_model(balancedUS_cnn, X_test, y_test, "CNN", "Balanced - US")

evaluate_model(balancedSMOTE_cnn, X_test, y_test, "CNN", "Balanced - SMOTE")



