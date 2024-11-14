import pyedflib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# https://pyedflib.readthedocs.io/en/latest/ref/edfreader.html

#LOGGER = getConfigLogger(__name__)

# General EDF signal functions from Marta (one of my supervisors)

def load_edf_signals(nsdrid, channels):
    # Get edf file path
    edf_file_path = edf_path #get_edf_path(f'shhs1-{nsdrid}')
    # Load edf data
    signals, sampling_rates = get_edf_channels(edf_file_path, channels)

    return signals, sampling_rates


def get_edf_channels(file_path, channels, shhs='1'):
    # Read edf file
    f = pyedflib.EdfReader(file_path)

    # Gather relevant info about
    signal_channels = {chn: i for i, chn in enumerate(f.getSignalLabels())}
    sampling_rates = {chn: f.getSampleFrequency(i) for chn, i in signal_channels.items()}
    #print(sampling_rates)

    signals = {}

    for channel in channels:
        channel_id = signal_channels.get(channel)
        if channel_id is not None:
            signals[channel] = f.readSignal(channel_id)
        else:
            print(f"Channel {channel} not found in signal!")

    f.close()

    return signals, sampling_rates


def get_channel_signal(f, channel, channel_id):
    # Check channel exists in signal
    if channel_id is not None:
        # Read channel_id
        signal = f.readSignal(channel_id)
        # freq_calculated = len(signal) / signal_duration
        # freq_recorded = f.getSampleFrequency(channel_id)
        # header = f.getSignalHeader(channel_id)
    else:
        print(f'{channel} not found in signal!')
        signal = None

    return signal

#### own code:


#Resampling code:
from scipy.signal import resample
def resample_signals(signals, sampling_rates, target_rate=1):
    resampled_signals = {}
    for channel, signal in signals.items():
        original_rate = sampling_rates[channel]
        target_length = int(len(signal) * (target_rate / original_rate))
        resampled_signals[channel] = resample(signal, target_length)
    return resampled_signals


# XML annotation parsing functions
import xml.etree.ElementTree as ET

import pandas as pd

#XML parser -> den går fint egt
def parse_xml_annotations(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    events = []

    #Get data from XML
    for scored_event in root.findall("ScoredEvents/ScoredEvent"):
        event_type = scored_event.find("EventType").text if scored_event.find("EventType") is not None else None
        #Skip "Stages|Stages" event type as we don't need it for sleep apnea classification
        if event_type == "Stages|Stages":
            break
        event_concept = scored_event.find("EventConcept").text if scored_event.find("EventConcept") is not None else None
        start = float(scored_event.find("Start").text) if scored_event.find("Start") is not None else None
        duration = float(scored_event.find("Duration").text) if scored_event.find("Duration") is not None else None
        signal_location = scored_event.find("SignalLocation").text if scored_event.find("SignalLocation") is not None else None

        # Store parsed event details in dictionary
        event = {
            "event_type": event_type,
            "event_concept": event_concept,
            "start": start,
            "duration": duration,
            "signal_location": signal_location
        }
        events.append(event)

    return events

#Combine the two
#Need to use sliding window: Segments EDF data into windows of a fixed size and labels each window based on overlap with XML annotations.


def segment_and_label_edf_data(edf_df, xml_annontations_df, window_size=30):
    """
    Segments EDF data into windows of a fixed size and labels each window based on overlap with relevant XML annotations.

    Parameters:
        edf_df (DataFrame): Resampled EDF data where each column is a channel.
        xml_annontations_df (DataFrame): XML annotations which is used for labeling
        window_size (int): Size of each window in seconds.

    Returns:
        DataFrame: Combined DataFrame with each window and its corresponding label.
    """
    # Set parameters and initialize list
    sampling_rate = 1  # this could be a parameter
    samples_per_window = window_size * sampling_rate
    segments = []

    # Set apnea-related events in XML which the label puts as 1, this could also be a parameter.
    apnea_related_events = ["Obstructive apnea|Obstructive Apnea", "Hypopnea|Hypopnea", "SpO2 desaturation|SpO2 desaturation"]

    num_windows = len(edf_df) // samples_per_window

    # Go through all the windows
    for i in range(num_windows):
        segment_start = i * window_size
        segment_end = segment_start + window_size

        # Extract the segment from EDF data
        # This line extracts a window of data from edf_df starting at i * samples_per_window and ending at (i + 1)
        # * samples_per_window. The window size here is defined by samples_per_window
        # (e.g., 30 samples for a 30-second window if resampled to 1 Hz).
        segment = edf_df.iloc[i * samples_per_window: (i + 1) * samples_per_window]

        # Initialize the label as 0 (non-apnea)
        label = 0

        # Check if any relevant annotation overlaps with the segment
        # "_," is the index of the row, "event" is the row.
        for _, event in xml_annontations_df.iterrows():
            # if event_conept is in "apnea_related_events" list
            if event["event_concept"] in apnea_related_events:
                event_start = event["start"]
                event_end = event["start"] + event["duration"]

                # If event_start and event_end is within the segment_start and segment_end
                if (event_start < segment_end) and (event_end > segment_start):
                    label = 1
                    break

        # Flatten the segment data into a single row for each channel
        segment_flat = segment.values.flatten()
        segment_dict = {f"{col}_t{t}": segment_flat[t] for t, col in enumerate(segment.columns)}

        # Append segment data and label
        segment_dict['Start Time'] = segment_start
        segment_dict['End Time'] = segment_end
        segment_dict['Label'] = label

        segments.append(segment_dict)

    combined_df = pd.DataFrame(segments)

    return combined_df



#Example usage:

#EDF

#Currently a direct file, otherwise will have a id and for loop
edf_path = "../SHHS_dataset/shhs1-200001.edf"

#Load the relevant channels
sleep_ap_channels = ["SaO2", "EMG", "NEW AIR", "ABDO RES"]
edf_signals, sampling_rates = load_edf_signals(0, channels=sleep_ap_channels)
""" 
#Original -> cannot become a pandas data frame. array length is not the same
for ch, sig in edf_signals.items():
    print(f"{ch}: {sig[:10]}")
"""

#Resample to get the same sampling rate
resampled_edf_signals = resample_signals(edf_signals, sampling_rates, target_rate=1)


pd_resampled_edf_signals = pd.DataFrame(resampled_edf_signals)
#print(pd_resampled_edf_signals.head())
# Print loaded EDF signals
#pd_resampled_edf_signals.to_csv("edf_df", encoding='utf-8', index=False)

#XML
xml_path = "../SHHS_dataset/shhs1-200001-nsrr.xml"
xml_annotations = parse_xml_annotations(xml_path)
df_xml_annotations = pd.DataFrame(xml_annotations)
#print(df_xml_annotations.head())
#print(df_xml_annotations.to_string())
#df_xml_annotations.to_csv("xml_df", encoding='utf-8', index=False)



#Combine - er ikke helt ferdig

window_size = 30  # Window size in seconds
combined_df = segment_and_label_edf_data(pd_resampled_edf_signals, df_xml_annotations, window_size)
#combined_df.to_csv("combined_df", encoding='utf-8', index=False)

""" 
# View the combined DataFrame with segments and labels
#print(combined_df.to_string())
# Import necessary metrics for imbalanced classification
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score

from sklearn.model_selection import train_test_split

# Separate features (X) and labels (y)
X = combined_df.drop(columns=["Start Time", "End Time", "Label"])
y = combined_df["Label"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Pipeline with Standard Scaling
logistic_pipeline = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))

# Train the Logistic Regression model
logistic_pipeline.fit(X_train, y_train)
y_pred_logistic = logistic_pipeline.predict(X_test)
y_pred_proba_logistic = logistic_pipeline.predict_proba(X_test)[:, 1]

# Evaluate Logistic Regression performance
print("Logistic Regression Results")
print(classification_report(y_test, y_pred_logistic))
print("Accuracy:", accuracy_score(y_test, y_pred_logistic))
print("Precision (class 1):", precision_score(y_test, y_pred_logistic, pos_label=1))
print("Recall (class 1):", recall_score(y_test, y_pred_logistic, pos_label=1))
print("F1-Score (class 1):", f1_score(y_test, y_pred_logistic, pos_label=1))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba_logistic))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logistic))

# Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Evaluate Random Forest performance
print("\nRandom Forest Results")
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision (class 1):", precision_score(y_test, y_pred_rf, pos_label=1))
print("Recall (class 1):", recall_score(y_test, y_pred_rf, pos_label=1))
print("F1-Score (class 1):", f1_score(y_test, y_pred_rf, pos_label=1))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
"""