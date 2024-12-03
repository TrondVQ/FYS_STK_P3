import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout

### Data Preparation ### Similar to FFNN

# Load the combined dataset
combined_df = pd.read_csv("./combined_dataset.csv")

features = combined_df[["SaO2", "EMG", "NEW AIR", "ABDO RES"]]
labels = combined_df["Apnea/Hypopnea"]

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Orignial unbalanced Dataset
X_train_unbalanced, y_train_unbalanced = X_train.values, y_train.values

# Creating a balanced dataset using bootstrapping
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

balanced_train_data = pd.concat([majority_class, minority_class_bootstrap])

X_train_balanced = balanced_train_data[["SaO2", "EMG", "NEW AIR", "ABDO RES"]].values
y_train_balanced = balanced_train_data["Apnea/Hypopnea"].values #Different fron FFNN

#Normalize the datasets
scaler = StandardScaler()

X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test.values)
X_train_unbalanced_scaled = scaler.fit_transform(X_train_unbalanced)

# RNN -> need to reshape data for 3d input
X_train_rnn_balanced = np.expand_dims(X_train_balanced_scaled, axis=1)
X_test_rnn = np.expand_dims(X_test_scaled, axis=1)
X_train_rnn_unbalanced = np.expand_dims(X_train_unbalanced_scaled, axis=1)

### RNN model
def create_rnn_model(input_shape):
    model = Sequential([
        SimpleRNN(32, activation='tanh', input_shape=input_shape),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

#Create and train models NEED TUNING
balanced_rnn = create_rnn_model((X_train_rnn_balanced.shape[1], X_train_rnn_balanced.shape[2]))
balanced_rnn.fit(
    X_train_rnn_balanced,
    y_train_balanced,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)

# Unbalanced dataset
unbalanced_rnn = create_rnn_model((X_train_rnn_unbalanced.shape[1], X_train_rnn_unbalanced.shape[2]))
unbalanced_rnn.fit(
    X_train_rnn_unbalanced,
    y_train_unbalanced,
     epochs=100,
     batch_size=32,
     validation_split=0.2
)

### Evaluation ###
def evaluate_model(model, X_test, y_test, dataset_name):
    print(f"Evaluation for {dataset_name} Dataset:")
    #test_loss, test_accuracy = model.evaluate(X_test, y_test)
    # print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Apnea/Hypopnea"]))

    auc_score = roc_auc_score(y_test, y_pred_probs)
    print(f"AUC Score: {auc_score:.4f}")

# Evaluate the RNN trained on unbalanced data
evaluate_model(unbalanced_rnn, X_test_rnn, y_test.values, "Unbalanced")

# Evaluate the RNN trained on balanced data
evaluate_model(balanced_rnn, X_test_rnn, y_test.values, "Balanced")
