import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

### Data Preparation ### About the same as FFNN, but tweaked as CNN expects 3D input

# Load the combined dataset
combined_df = pd.read_csv("./combined_dataset.csv")

features = combined_df[["SaO2", "EMG", "NEW AIR", "ABDO RES"]]
labels = combined_df["Apnea/Hypopnea"]

sequence_length = 30  # 30s window size -> combined.df dataset
num_channels = 4 # [[SaO2, EMG, NEW AIR, ABDO RES]]

# Reshape features for the CNN format -> 3D input
# Different than from FFNN: Assume each sample corresponds to sequence_length time steps (30s)
num_samples = len(features) // sequence_length
features_cnn = features.values[:num_samples * sequence_length].reshape(num_samples, sequence_length, num_channels)
labels_cnn = labels.values[:num_samples * sequence_length:sequence_length]  # One label per sequence

X_train, X_test, y_train, y_test = train_test_split(features_cnn, labels_cnn, test_size=0.2, random_state=42)

#Due to class imbalance, we will balance the dataset using bootstrapping.
# We will also see the effect of balancing on the model performance.
# The original unbalanced dataset
X_train_unbalanced, y_train_unbalanced = X_train, y_train

"""
#Standardscaler only works with 2D data, so we do it manually
#An issue here is that, if we standardize/normalize the data, then the unbalanced dataset does not produce precision
#Explore more? or write about? maybe tuning the parameters will fix the issue?
mean = X_train.mean(axis=(0, 1), keepdims=True)
std = X_train.std(axis=(0, 1), keepdims=True)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
 """

# Balance the training data
# Expand y_train to match X_train's sequence length: Denne trenger jeg å se mer på
y_train_expanded = np.repeat(y_train[:, np.newaxis], repeats=sequence_length, axis=1)
y_train_expanded = np.expand_dims(y_train_expanded, axis=-1)
train_data = np.concatenate([X_train, y_train_expanded], axis=-1)

# Separate majority and minority classes
majority_class = train_data[train_data[:, 0, -1] == 0]  # Normal class
minority_class = train_data[train_data[:, 0, -1] == 1]  # Apnea/Hypopnea class

# Oversample minority class
minority_class_bootstrap = resample(
    minority_class,
    replace=True,
    n_samples=len(majority_class),  # Match majority class size
    random_state=42
)

# Combine majority and oversampled minority class
balanced_train_data = np.concatenate([majority_class, minority_class_bootstrap], axis=0)

# Separate features and labels -> 3D
X_train_balanced = balanced_train_data[:, :, :-1]  # Features (exclude the label dimension)
y_train_balanced = balanced_train_data[:, 0, -1]  # Extract one label per sequence

# CNN implementation similar to 16.8. The CIFAR01 data set, but 1d
#NEED TO TUNE
def create_cnn(sequence_length, num_channels):
    model = Sequential()
    #The Input layer -> 1d convolutional layer
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(sequence_length, num_channels)))

    # Pooling Layer -> 1d
    model.add(MaxPooling1D(pool_size=2))

    # Can add more convolutional and pooling layers -> need to experiment with
    """ 
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    """

    model.add(Flatten())

    #  Last fully connected layer
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout for regularization
    model.add(Dense(units=1, activation='sigmoid'))  # Binary classification (Apnea vs. Normal)

    # Compile Model ->
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

sequence_length = 30
num_channels = 4

# Balanced model
balanced_cnn = create_cnn(sequence_length, num_channels)
balanced_cnn.summary()
balanced_history = balanced_cnn.fit(
    X_train_balanced, y_train_balanced,
    validation_data=(X_test, y_test),
    epochs=250,
    batch_size=32
)

#Need to tune these model parameters
# Unbalanced model
unbalanced_cnn = create_cnn(sequence_length, num_channels)
unbalanced_cnn.summary()
unbalanced_history = unbalanced_cnn.fit(
    X_train_unbalanced, y_train_unbalanced,
    validation_data=(X_test, y_test),
    epochs=250, # Need 250 epochs to converge
    batch_size=32
)

### Evaluation ###
def evaluate_model(model, X_test, y_test, dataset_name): #Same function for FFNN, CNN, RNN
    print(f"Evaluation for {dataset_name} Dataset:")
    #test_loss, test_accuracy = model.evaluate(X_test, y_test)
   # print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Apnea/Hypopnea"]))

    auc_score = roc_auc_score(y_test, y_pred_probs)
    print(f"AUC Score: {auc_score:.4f}")

# Evaluate the CNN trained on unbalanced data
evaluate_model(unbalanced_cnn, X_test, y_test, "Unbalanced")

# Evaluate the CNN trained on balanced data
evaluate_model(balanced_cnn, X_test, y_test, "Balanced")
