# FYSTK4155 Autumn 2024 - Project 3  Data Analysis with Machine Learning


### Installation
To set up the project locally, follow these steps:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/TrondVQ/FYS_STK_P3
    cd FYS_STK_P3
    ```

2. **Install the required dependencies:**
    Note that tensorflow only supports Python 3.9–3.12
    ```sh
    pip install numpy matplotlib scikit-learn pandas tensorflow==2.12.0  pyedflib imblearn
    if python3 run:
    pip3 install numpy matplotlib scikit-learn pandas tensorflow==2.12.0 pyedflib imblearn
    ```
3. **Dataset files:**
    The real SHHS1 dataset files are not included in the repository due to privacy reasons. The SHHS1 dataset files can be downloaded from the following link (you will need to apply for access):
    https://sleepdata.org/datasets/shhs
    The combined_dataset.csv included in this repository has therefore been randomly generated and not as long as the original(around 150000 lines). You can download the SHHS1 datafiles, put them in the corresponding folders and run "parta_SHHS1_preprosessing" to generate the SHHS1 combined_dataset.csv

### Project Structure
- **Code/**: Contains the final Python code for exercises a(preprosessing of dataset) and b(Random forest, FFNN and CNN algorithms).
- **Supplemental material/**: Contains leftover code.

