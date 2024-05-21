"""Main module to run technical test for Volta Medical."""

import sys
from pathlib import Path

from test_volta import config, features_processing, load_data, model, preprocess


def main():
    """Main function for technical test."""
    ### Load user input
    folder_path = Path(sys.argv[1])
    ### Data Loading stage
    signals = load_data.load_all_signals(folder_path)
    annotations = load_data.load_all_annotations(folder_path)

    ### Pre-processing stage
    signal_heartbeat = preprocess.preprocess(annotations=annotations, signals=signals)

    ### Feature calculation stage
    features, target, list_patient_id = features_processing.build_features(
        signal_heartbeat=signal_heartbeat, annotations=annotations
    )
    ### Create output result folder
    Path.mkdir(config.RESULT_FOLDER_PATH, exist_ok=True)
    ### Training and inference stage
    model.train_and_compute(
        features=features, target=target, list_patient_id=list_patient_id
    )


main()
