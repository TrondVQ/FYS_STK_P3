# General information about the datasets:
Source: https://sleepdata.org/datasets/shhs/files/polysomnography

In summary:
EDF Data: This contains the actual physiological signals (e.g., respiratory, oxygen saturation) and serves as the input data. You’ll extract relevant signal segments from this data to train your model.

XML Annotations: This contains the labels or annotations, specifying events like respiratory disturbances (e.g., apnea, hypopnea), which will act as your target labels. These annotations will tell your model which segments in the EDF data correspond to apnea events.

The data is a time-series data, and can be seen as a 1 dimentional picture.
* How the classification works, will be a sliding window, where we classifiy each.
* Have more data

# Info about the Dataset - SHHS 1
This one datafile captures physiological and environmental signals recorded during a sleep study over approximately 9 hours, with each channel providing a unique measurement of the subject's status or environment.
### Dataset Overview
- **Total Duration**: ~9 hours (0 to 32,520 seconds)
- **Total Time Points**: 4,065,000
- **Number of Channels**: 14
- **File Size**: 465.2 MB
- **Type**: EDF (European Data Format) -> Signal data with different sampling rates(Hz) for each channel
- **Data Type**: All values are in `float64` format.
---

### Channels and Measurements
Source: https://sleepdata.org/datasets/shhs/pages/10-montage-and-sampling-rate-information-shhs1.md
| Channel           | Description                              | Sampling rate | Notes                                           |
|-------------------|------------------------------------------|---------------|-------------------------------------------------|
| **SaO2** (Useful) | Oxygen saturation levels                 | 1 Hz          | Potential artifacts with negative values        |
| **H.R.**          | Heart rate (beats per minute)            | 1 Hz          | Negative values may indicate noise              |
| **EEG(sec)**      | EEG signal                               | 125 hz        | Captures neural activity                        |
| **ECG**           | Electrocardiogram signal                 | 125 hz        | Monitors heart rhythm                           |
| **EMG**(Useful)   | Electromyography signal                  | 125 hz        | Measures muscle activity                        |
| **EOG(L)**        | Left eye electrooculogram                | 50 hz         | Tracks eye movement                             |
| **EOG(R)**        | Right eye electrooculogram               | 50 hz         | Tracks eye movement                             |
| **EEG**           | EEG signal (additional placement)        | 125 hz        | Secondary EEG channel                           |
| **THOR RES**      | Thoracic respiratory effort              | 10 hz         | Captures breathing patterns                     |
| **ABDO RES**(Useful)| Abdominal respiratory effort             | 10 hz         | Measures abdominal breathing                    |
| **POSITION**      | Body position indicator                  | 1 hz          | Indicates subject’s posture                     |
| **LIGHT**         | Environmental light level                | 1 hz          | Constant value; may indicate light presence     |
| **NEW AIR**(Useful)| Airflow measurement                      | 10 hz         | Related to respiratory function                 |
| **OX stat**       | Oxygen status indicator                  | 1 hz          | Likely linked to oxygen metrics                |

SaO2 may be the most important for sleep apnea detection. Use annonated data with sleep score to use for supervised data.

In order to use this file for supervised learning, we need to parse the XML file to get the labels.
  - The channels in annontation: SaO2, EMG, NEW AIR, ABDO RES, and are therefore the ones we need to use for supervised learning.
  - Basiccly -> if apnea = 1, then we have a label. if not or the time is unannontated = 0.

Need to get common sampling rate -> resampling (1hz -> as it is the lowest)

# Info about the Dataset - SHHS 1 XML

An xml file that contains the labels for the data in the SHHS 1 dataset where certian parts of the data is annotated with labels.

### Event types

- EventType "Arousals" refer to  brief periods where the sleeper’s brain activity shifts toward wakefulness, 
often detected by spikes in EEG (electroencephalography) activity.
These events are typically short and may not lead to full awakening, but they can disrupt the continuity of sleep.

- EventType "Respiratory" signifies events related to breathing disturbances. 
  - Common types include apnea (a complete pause in breathing) and hypopnea (a partial reduction in airflow).

In this dataset the signal channels are as follows:

Event type:
Respiratory and Arousals (This we need to use for supervised learning):
- **SaO2**: Oxygen saturation levels -> Respiratory -> <EventConcept>SpO2 artifact|SpO2 artifact</EventConcept> or SpO2 desaturation|SpO2 desaturation
- **EMG**: Electromyography signal -> Arousals -> <EventConcept>Arousal|Arousal ()</EventConcept>
- **NEW AIR**: Airflow measurement -> Respiratory (assoiated with event concept: Hypopnea)
- **ABDO RES**: Abdominal respiratory effort -> Respiratory (assoiated with event concept: Obstructive Apnea)

Så basiccly -> parse the XML get the labels and then use the data from the other dataset to train the model.

This file also contains information about the
Sleep Stages (Don't need to use this for supervised learning):
<EventType>Stages|Stages</EventType>
<EventConcept>Wake|0</EventConcept>
and:
<EventConcept>Stage 3 sleep|3</EventConcept> -> where stage is from 1-3, with wake as 0. REM sleep|5 and 4 -> ? not in this data set

### Combining the dataset
The respiratory abnormalities which are the focus of the SHHS are apneas and hypopneas. An apnea is a complete or 
almost complete cessation of airflow, lasting at least 10 seconds,
and usually associated with desaturation or an arousal. 
A hypopnea is a reduction in airflow (<70% of a “baseline” level), associated with desaturation or arousal."