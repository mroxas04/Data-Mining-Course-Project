# Basketball Activity Classification Project

This project implements a supervised learning pipeline to recognize basketball activities from wearable sensor data.

## Overview

The dataset consists of accelerometer and gyroscope readings collected from a sensor worn on the right arm of four participants. Each participant performed five basketball activities — **hold**, **pass**, **dribble**, **shoot**, and **pickup** — several times. Each trial is saved in a comma‑separated `.txt` file whose name encodes the participant and activity (e.g. `D_pass3.txt`).

This project parses the raw time‑series files, extracts statistical features, normalizes them, and trains multiple classifiers (k‑Nearest Neighbors, Random Forest, and Support Vector Machine) to recognize the activity. Cross‑validation is used to evaluate the models, and performance metrics are printed to the console.

## Usage

1. Ensure the dataset is extracted to a directory (by default `../basketball_dataset_unzipped/proyecto`).
2. Install the required Python packages:

```sh
pip install -r requirements.txt
```

3. Run the main script:

```sh
python main.py
```

The script will load the data, compute features, train and evaluate the models, and print the results.

## Files

- `main.py` – Main script that loads data, extracts features, trains classifiers, and reports results.
- `requirements.txt` – Python dependencies for the project.

## License

This code is provided for educational purposes as part of a course project. Feel free to modify and reuse it for research or learning.