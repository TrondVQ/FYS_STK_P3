import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from project2_classes_and_functions import *
from sklearn.preprocessing import StandardScaler

### Data Preparation ### About the same as in partb_randomforest.py

# Load the combined dataset
combined_df = pd.read_csv("./combined_dataset.csv")

features = combined_df[["SaO2", "EMG", "NEW AIR", "ABDO RES"]]
labels = combined_df["Apnea/Hypopnea"]

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

#Due to class imbalance, we will balance the dataset using bootstrapping.
# We will also see the effect of balancing on the model performance.

# The original unbalanced dataset -> These are converted to numpy
X_train_unbalanced, y_train_unbalanced = X_train.values, y_train.values.reshape(-1, 1)

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
#These are converted to numpy
X_train_balanced = balanced_train_data[["SaO2", "EMG", "NEW AIR", "ABDO RES"]].values
y_train_balanced = balanced_train_data["Apnea/Hypopnea"].values.reshape(-1, 1)

# Project 2 FFNN implementation

#Normalize the datasets
scaler = StandardScaler()
X_train_balanced = scaler.fit_transform(X_train_balanced)
X_test_balanced = scaler.transform(X_test.values)

X_train_unbalanced = scaler.fit_transform(X_train_unbalanced)
X_test_unbalanced = scaler.transform(X_test.values)

# Architechture: NEED TO TUNE THESE
balanced_nn = NetworkClass(
    cost_fun=CostCrossEntropy,
    cost_der=CostCrossEntropyDer,
    network_input_size=X_train_balanced.shape[1],
    layer_output_sizes=[50, 1],
    activation_funcs=[sigmoid, sigmoid],
    activation_ders=[sigmoid_derivative, sigmoid_derivative]
)

unbalanced_nn = NetworkClass(
    cost_fun=CostCrossEntropy,
    cost_der=CostCrossEntropyDer,
    network_input_size=X_train_balanced.shape[1],
    layer_output_sizes=[50, 1],
    activation_funcs=[sigmoid, sigmoid],
    activation_ders=[sigmoid_derivative, sigmoid_derivative]
)

# Train the networks. NEED TO TUNE THE PARAMETERS
balanced_nn.train(
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

#Evaluation for balanced data
balanced_train_predictions = balanced_nn.predict(X_train_balanced)
balanced_test_predictions = balanced_nn.predict(X_test_balanced)

balanced_train_predictions_binary = (balanced_train_predictions >= 0.5).astype(int)
balanced_test_predictions_binary = (balanced_test_predictions >= 0.5).astype(int)

print("Balanced Data Evaluation:")
print("Train Classification Report:")
print(classification_report(y_train_balanced, balanced_train_predictions_binary))
print("Test Classification Report:")
print(classification_report(y_test.values, balanced_test_predictions_binary))

balanced_train_auc = roc_auc_score(y_train_balanced, balanced_train_predictions)
balanced_test_auc = roc_auc_score(y_test.values, balanced_test_predictions)
print(f"Balanced Train AUC-ROC: {balanced_train_auc:.4f}")
print(f"Balanced Test AUC-ROC: {balanced_test_auc:.4f}")

#Evaluation for unbalanced data
unbalanced_train_predictions = unbalanced_nn.predict(X_train_unbalanced)
unbalanced_test_predictions = unbalanced_nn.predict(X_test_unbalanced)

unbalanced_train_predictions_binary = (unbalanced_train_predictions >= 0.5).astype(int)
unbalanced_test_predictions_binary = (unbalanced_test_predictions >= 0.5).astype(int)

print("Unbalanced Data Evaluation:")
print("Train Classification Report:")
print(classification_report(y_train_unbalanced, unbalanced_train_predictions_binary))
print("Test Classification Report:")
print(classification_report(y_test.values, unbalanced_test_predictions_binary))

unbalanced_train_auc = roc_auc_score(y_train_unbalanced, unbalanced_train_predictions)
unbalanced_test_auc = roc_auc_score(y_test.values, unbalanced_test_predictions)
print(f"Unbalanced Train AUC-ROC: {unbalanced_train_auc:.4f}")
print(f"Unbalanced Test AUC-ROC: {unbalanced_test_auc:.4f}")