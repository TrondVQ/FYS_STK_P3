import os
import pandas as pd
from pyedflib import EdfReader
from scipy.signal import resample
import xml.etree.ElementTree as ET

""" 
No need to run this file as the combined dataset(combined_dataset.csv) is already provided in the repository. 
Though the one in the github only contains "fake data" as the original dataset needs permission to access.

This is the code that will be used to preprocess the SHHS1 dataset where it combines the EDF and XML files, 
 resamples the signals, and segments the data into windows of a fixed size.
 The code will then label each window based on the overlap with relevant XML annotations.
 The processed data will be saved as a combined dataset CSV file.

"""
#Path to folders
EDF_FOLDER = "../SHHS_dataset/edf_files"
XML_FOLDER = "../SHHS_dataset/annontation_files"
OUTPUT_FOLDER = "./processed_data/"
COMBINED_FILE = "./combined_dataset.csv" #  To process your own files-> can add the SHHS1 files to the SHHS_dataset folder

#Get an initial version of this edfReader from Marta Quemada Lopez, but have tuned it for our use-case
# gets the EDF channels and their sampling rates from the EDF file.
def get_edf_channels(file_path, channels):
    with EdfReader(file_path) as f:
        signal_channels = {chn: i for i, chn in enumerate(f.getSignalLabels())}
        sampling_rates = {chn: f.getSampleFrequency(i) for chn, i in signal_channels.items()}

        signals = {}
        for channel in channels:
            channel_id = signal_channels.get(channel)
            if channel_id is not None:
                signals[channel] = f.readSignal(channel_id)
            else:
                if channel == "NEW AIR": #New Air will have alternative names. https://sleepdata.org/datasets/shhs/pages/08-equipment-shhs1.md.
                    print(f"NEW AIR not found. Trying Alternative names in EDF file {file_path}")
                    alt_names= ["AUX", "NEWAIR", "new A/F", "airflow", "AIRFLOW"] #The files with Airflow are double-checked to be correct as channel 12 sometimes has this name as well. Files: 51, 54, 43, 37, 05, 61, 74
                    for alt_name in alt_names:
                        if alt_name in signal_channels:
                            print(f"Found {alt_name} in channel list.")
                            channel_id = signal_channels[alt_name]
                            signals["NEW AIR"] = f.readSignal(channel_id)
                            sampling_rates["NEW AIR"] = sampling_rates.pop(alt_name)  # Update the name in the dictionary
                            break
                else:
                    print(f"Channel {channel} not found in {file_path}")

    if "NEW AIR" not in signals:
        print(f"Warning: 'NEW AIR/AIRFLOW' channel is missing in {file_path}.")

    return signals, sampling_rates

#  Resamples all signals to a target sampling rate.
# Asked ChatGPT for an initial version of this. Has since tuned it.
def resample_signals(signals, sampling_rates, target_rate=1):
    resampled_signals = {}
    for channel, signal in signals.items():
        original_rate = sampling_rates[channel]
        target_length = int(len(signal) * (target_rate / original_rate))
        resampled_signals[channel] = resample(signal, target_length)
    return resampled_signals

#    Parses XML annotations and extracts event details.
# Asked ChatGPT for an initial version of this. Has since tuned it.
def parse_xml_annotations(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    events = []

    for scored_event in root.findall("ScoredEvents/ScoredEvent"):
        event_type = scored_event.find("EventType").text if scored_event.find("EventType") is not None else None
        if event_type == "Stages|Stages":
            continue

        event_concept = scored_event.find("EventConcept").text if scored_event.find("EventConcept") is not None else None
        start = float(scored_event.find("Start").text) if scored_event.find("Start") is not None else None
        duration = float(scored_event.find("Duration").text) if scored_event.find("Duration") is not None else None

        event = {
            "event_type": event_type,
            "event_concept": event_concept,
            "start": start,
            "duration": duration
        }
        events.append(event)

    return events

# Asked ChatGPT for an initial version of this. Has since tuned it based on domain knowledge and information about the dataset.
#  Segments EDF data into windows of a fixed size with overlap, and labels each window based on overlap with relevant XML annotations.
def segment_and_label_edf_data(edf_df, xml_annotations_df, window_size=30, overlap_size=15):
    segments = []

    apnea_related_events = ["Obstructive apnea|Obstructive Apnea", "Hypopnea|Hypopnea"]

    window_step_size = window_size - overlap_size
    num_windows = len(edf_df) // window_step_size

    for i in range(num_windows):
        segment_start = i * window_step_size
        segment_end = segment_start + window_size

        segment = edf_df.iloc[segment_start: segment_end]

        label = 0  #0 = no apnea/hypopnea
        for _, event in xml_annotations_df.iterrows():
            #if annontation is annotated as hypopnea/apnea
            if event["event_concept"] in apnea_related_events:
                event_start = event["start"]
                event_end = event["start"] + event["duration"]

                # If there is 10 seconds overlap between the annontation and the window, label the segment as apnea/hypopnea
                overlap_start = max(segment_start, event_start)
                overlap_end = min(segment_end, event_end)
                overlap_duration = overlap_end - overlap_start

                if overlap_duration >= 10:
                    label = 1
                    break

        segment_dict = {col: segment[col].mean() for col in segment.columns}
        segment_dict['Start Time'] = segment_start
        segment_dict['End Time'] = segment_end
        segment_dict['Apnea/Hypopnea'] = label

        segments.append(segment_dict)

    return pd.DataFrame(segments)

# Processes a single EDF and XML file pair, resampling and segmenting the data.
#ChatGPT helped produce both process_single_file and process_all_files
def process_single_file(edf_file_path, xml_file_path, target_channels, target_rate=1):

    edf_signals, sampling_rates = get_edf_channels(edf_file_path, target_channels)

    resampled_signals = resample_signals(edf_signals, sampling_rates, target_rate=target_rate)
    edf_df = pd.DataFrame(resampled_signals)

    xml_annotations = parse_xml_annotations(xml_file_path)
    xml_df = pd.DataFrame(xml_annotations)

    combined_df = segment_and_label_edf_data(edf_df, xml_df)
    return combined_df

#Processes all EDF and XML file pairs in the dataset folder and combines them into one DataFrame.
def process_all_files(edf_folder, xml_folder, target_channels, target_rate=1, output_folder="./processed_data/"):
    combined_data = []

    for edf_file in os.listdir(edf_folder):
        if edf_file.endswith(".edf"):
            nsrr_id = edf_file.split("-")[1].split(".")[0]
            xml_file = f"shhs1-{nsrr_id}-nsrr.xml"
            edf_file_path = os.path.join(edf_folder, edf_file)
            xml_file_path = os.path.join(xml_folder, xml_file)

            if os.path.exists(xml_file_path):
                print(f"Processing: {edf_file_path} and {xml_file_path}")

                try:
                    combined_df = process_single_file(edf_file_path, xml_file_path, target_channels, target_rate)
                    output_file = os.path.join(output_folder, f"combined_{nsrr_id}.csv")
                    combined_df.to_csv(output_file, index=False)
                    combined_data.append(combined_df)
                except Exception as e:
                    print(f"Error processing {edf_file_path} and {xml_file_path}: {e}")
            else:
                print(f"XML file for {edf_file} not found. Skipping...")

    # Combine all processed DataFrames into one
    if combined_data:
        final_combined_df = pd.concat(combined_data, ignore_index=True)
        final_combined_df.to_csv(COMBINED_FILE, index=False)
        print(f"Saved combined dataset to {COMBINED_FILE}")


sleep_apnea_channels = ["SaO2", "EMG", "NEW AIR", "ABDO RES"]
#process_single_file("/Users/tvq/Documents/FYS_STK_P3/SHHS_dataset/edf_files/shhs1-200022.edf", "/Users/tvq/Documents/FYS_STK_P3/SHHS_dataset/annontation_files/shhs1-200022-nsrr.xml", sleep_apnea_channels, 1)
process_all_files(EDF_FOLDER, XML_FOLDER, target_channels=sleep_apnea_channels, target_rate=1)
