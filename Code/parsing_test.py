import pyedflib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# https://pyedflib.readthedocs.io/en/latest/ref/edfreader.html

#LOGGER = getConfigLogger(__name__)

# General EDF signal functions from Marta (one of my supervisors)
#Husk recording start og slutt -> kanskje kutt de ut
#Husk example files : istedenfor ekte data.

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
#The basic thought is to use the XML data to find the proper labels for Apnea and hypopnea events
# So we use the XML data to label the EDF data using their event types and durations.
    # Ex if the XML data has an event type "Obstructive apnea|Obstructive Apnea" with a duration of 10 seconds,
    # find the corresponding EDF data points and label them as apnea events.(1)
    # Do the same for hypopnea events.
def label_edf_from_xml(edf_df, xml_annotations_df):
    """
    Labels EDF data points with separate columns for apnea and hypopnea events based on XML annotations.

    Parameters:
        edf_df (DataFrame): Resampled EDF data with time indices.
        xml_annotations_df (DataFrame): XML annotations with start, duration, and event types.

    Returns:
        DataFrame: EDF data with added 'Apnea Label' and 'Hypopnea Label' columns.
    """

    edf_df["Apnea Label"] = 0
    edf_df["Hypopnea Label"] = 0

    #Find the relevant events from the XML data. Also make sure that their duration is at least 10 seconds(Criteria)
    apnea_events = xml_annotations_df[
        (xml_annotations_df["event_concept"] == "Obstructive apnea|Obstructive Apnea") &
        (xml_annotations_df["duration"] >= 10)
        ]
    hypopnea_events = xml_annotations_df[
        (xml_annotations_df["event_concept"] == "Hypopnea|Hypopnea") &
        (xml_annotations_df["duration"] >= 10)
        ]

    # Label EDF data points for apnea events, the first item is a index, so skip that.
    for _, event in apnea_events.iterrows():
        event_start = event["start"]
        event_end = event["start"] + event["duration"]
        edf_df.loc[(edf_df.index >= event_start) & (edf_df.index < event_end), "Apnea Label"] = 1

    # Label EDF data points for hypopnea events, the first item is a index, so skip that.
    for _, event in hypopnea_events.iterrows():
        event_start = event["start"]
        event_end = event["start"] + event["duration"]
        edf_df.loc[(edf_df.index >= event_start) & (edf_df.index < event_end), "Hypopnea Label"] = 1

    return edf_df

#Example usage:
#EDF
#Currently a direct file, otherwise will have a id and for loop
edf_path = "../SHHS_dataset/shhs1-200001.edf"

#Load the relevant channels (the ones noted in the XML file)
sleep_ap_channels = ["SaO2", "EMG", "NEW AIR", "ABDO RES"]
edf_signals, sampling_rates = load_edf_signals(0, channels=sleep_ap_channels)
""" 
#Original without resampling -> cannot become a pandas data frame. array length is not the same
for ch, sig in edf_signals.items():
    print(f"{ch}: {sig[:10]}")
"""

#Resample to get the same sampling rate. Choose 1 as target rate as SpO2 is 1 Hz
resampled_edf_signals = resample_signals(edf_signals, sampling_rates, target_rate=1)
pd_resampled_edf_signals = pd.DataFrame(resampled_edf_signals)
#print(pd_resampled_edf_signals.head())
#pd_resampled_edf_signals.to_csv("edf_df", encoding='utf-8', index=False)

#XML
xml_path = "../SHHS_dataset/shhs1-200001-nsrr.xml"
xml_annotations = parse_xml_annotations(xml_path)
df_xml_annotations = pd.DataFrame(xml_annotations)
#print(df_xml_annotations.head())
#print(df_xml_annotations.to_string())
#df_xml_annotations.to_csv("xml_df", encoding='utf-8', index=False)



#Combine - må snakke med veileder
combined_df = label_edf_from_xml(pd_resampled_edf_signals, df_xml_annotations)
combined_df.to_csv("combined_df", encoding='utf-8', index=False)
print(combined_df.to_string())
#print(combined_df.head())