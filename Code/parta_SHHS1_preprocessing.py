import os
import pandas as pd
from pyedflib import EdfReader
from scipy.signal import resample
import xml.etree.ElementTree as ET

#This is the code that will be used to preprocess the SHHS1 dataset where it combines the EDF and XML files,
# resamples the signals, and segments the data into windows of a fixed size.
# The code will then label each window based on the overlap with relevant XML annotations.
# The processed data will be saved as a combined dataset CSV file.
# No need to run this file as the combined dataset is already provided in the repository.

#Path to folders
EDF_FOLDER = "../SHHS_dataset/edf_files"
XML_FOLDER = "../SHHS_dataset/annontation_files"
OUTPUT_FOLDER = "./processed_data/"
COMBINED_FILE = "./combined_dataset.csv" # 1-25 -> can add it on you own. just use combined/processed data directly


def get_edf_channels(file_path, channels):
    """
    Reads specified channels from an EDF file and returns their signals and sampling rates.

    Parameters:
        file_path (str): Path to the EDF file.
        channels (list): List of channel names to extract.

    Returns:
        signals (dict): Dictionary of channel signals.
        sampling_rates (dict): Dictionary of sampling rates for each channel.
    """
    with EdfReader(file_path) as f:
        signal_channels = {chn: i for i, chn in enumerate(f.getSignalLabels())}
        sampling_rates = {chn: f.getSampleFrequency(i) for chn, i in signal_channels.items()}

        signals = {}
        for channel in channels:
            channel_id = signal_channels.get(channel)
            if channel_id is not None:
                signals[channel] = f.readSignal(channel_id)
            else:
                print(f"Channel {channel} not found in {file_path}")

    return signals, sampling_rates


def resample_signals(signals, sampling_rates, target_rate=1):
    """
    Resamples all signals to a target sampling rate.

    Parameters:
        signals (dict): Dictionary of signals.
        sampling_rates (dict): Dictionary of sampling rates for each signal.
        target_rate (int): Target sampling rate in Hz.

    Returns:
        dict: Dictionary of resampled signals.
    """
    resampled_signals = {}
    for channel, signal in signals.items():
        original_rate = sampling_rates[channel]
        target_length = int(len(signal) * (target_rate / original_rate))
        resampled_signals[channel] = resample(signal, target_length)
    return resampled_signals


def parse_xml_annotations(xml_file_path):
    """
    Parses XML annotations and extracts event details.

    Parameters:
        xml_file_path (str): Path to the XML annotation file.

    Returns:
        list: List of event dictionaries containing event details.
    """
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


def segment_and_label_edf_data(edf_df, xml_annotations_df, window_size=30):
    """
    Segments EDF data into windows of a fixed size and labels each window based on overlap with relevant XML annotations.

    Parameters:
        edf_df (DataFrame): Resampled EDF data where each column is a channel.
        xml_annotations_df (DataFrame): XML annotations used for labeling.
        window_size (int): Size of each window in seconds.

    Returns:
        DataFrame: Combined DataFrame with each window and its corresponding label.
    """
    sampling_rate = 1  # This could be a parameter
    samples_per_window = window_size * sampling_rate
    segments = []

    apnea_related_events = ["Obstructive apnea|Obstructive Apnea", "Hypopnea|Hypopnea"]

    num_windows = len(edf_df) // samples_per_window

    for i in range(num_windows):
        segment_start = i * window_size
        segment_end = segment_start + window_size

        segment = edf_df.iloc[i * samples_per_window: (i + 1) * samples_per_window]

        label = 0
        for _, event in xml_annotations_df.iterrows():
            if event["event_concept"] in apnea_related_events:
                event_start = event["start"]
                event_end = event["start"] + event["duration"]

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


def process_single_file(edf_file_path, xml_file_path, target_channels, target_rate=1):
    """
    Processes a single EDF and XML file pair, resampling and segmenting the data.

    Parameters:
        edf_file_path (str): Path to the EDF file.
        xml_file_path (str): Path to the XML file.
        target_channels (list): Channels to extract from the EDF file.
        target_rate (int): Target sampling rate for resampling.

    Returns:
        DataFrame: Combined DataFrame of resampled EDF data and labeled annotations.
    """
    edf_signals, sampling_rates = get_edf_channels(edf_file_path, target_channels)

    resampled_signals = resample_signals(edf_signals, sampling_rates, target_rate=target_rate)
    edf_df = pd.DataFrame(resampled_signals)

    xml_annotations = parse_xml_annotations(xml_file_path)
    xml_df = pd.DataFrame(xml_annotations)

    combined_df = segment_and_label_edf_data(edf_df, xml_df)
    return combined_df


def process_all_files(edf_folder, xml_folder, target_channels, target_rate=1, output_folder="./processed_data/"):
    """
    Processes all EDF and XML file pairs in the dataset folder and combines them into one DataFrame.

    Parameters:
        edf_folder (str): Path to the folder containing EDF files.
        xml_folder (str): Path to the folder containing XML files.
        target_channels (list): Channels to extract from the EDF files.
        target_rate (int): Target sampling rate for resampling.
        output_folder (str): Folder to save the processed combined DataFrames.
    """
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


# Example usage
sleep_apnea_channels = ["SaO2", "EMG", "NEW AIR", "ABDO RES"]
process_all_files(EDF_FOLDER, XML_FOLDER, target_channels=sleep_apnea_channels, target_rate=1)
