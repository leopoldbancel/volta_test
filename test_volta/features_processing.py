"""Module computing features to input the model with."""

import numpy as np
import pandas as pd
from tqdm import tqdm

from test_volta import config


def compute_features(sliding_window: np.ndarray):
    """Compute the features from a fixes-sized time window."""
    moving_average = np.mean(sliding_window[:, 1:13], axis=0)

    moving_std = np.std(sliding_window[:, 1:13], axis=0)
    moving_max = np.max(sliding_window[:, 1:13], axis=0)
    moving_amplitude = np.max(sliding_window[:, 1:13], axis=0) - np.min(
        sliding_window[:, 1:13], axis=0
    )
    moment_in_heartbeat = np.mean(sliding_window[:, -1])
    curve_length = np.sum(np.abs(np.diff(sliding_window[:, 1:13], axis=0)), axis=0)
    return np.hstack(
        [
            moving_average,
            moving_std,
            moving_max,
            moving_amplitude,
            moment_in_heartbeat,
            curve_length,
        ]
    )


def find_target_vector(
    annotations: pd.DataFrame, patient_id: int, time_start: float, time_end: float
):
    """Compute target vector (boolean array of shape (2,))."""
    annotation_patient = annotations[
        (annotations["pat_id"] == patient_id)
        & (annotations["signal_type"] == config.SIGNAL_USED_FOR_ANNOTATION)
        & (annotations["time"] >= time_start)
        & (annotations["time"] < time_end)
    ]
    target_vector = np.zeros(3)
    if annotation_patient["symbol"].str.contains("p").any():
        target_vector[0] = 1
    if annotation_patient["symbol"].str.contains("N").any():
        target_vector[1] = 1
    if annotation_patient["symbol"].str.contains("t").any():
        target_vector[2] = 1
    return target_vector


def build_features(signal_heartbeat: pd.DataFrame, annotations: pd.DataFrame):
    """Build the feature arrays that will later input the model."""
    X = []
    y = []
    list_patient_id = []
    for _, beat_df in tqdm(signal_heartbeat.groupby("beat_nb")):
        # Create sliding windows with overlaps of 1/2 window size
        for sliding_window in np.lib.stride_tricks.sliding_window_view(
            x=beat_df,
            window_shape=int(config.SLIDING_WINDOW_SIZE * config.SAMPLING_FREQUENCY),
            axis=0,
        )[:: int(config.SLIDING_WINDOW_SIZE * config.SAMPLING_FREQUENCY / 2)]:
            X += [compute_features(sliding_window.T)]
            patient_id = int(beat_df["pat_id"].iloc[0])
            list_patient_id += [patient_id]
            y += [
                find_target_vector(
                    annotations=annotations,
                    patient_id=patient_id,
                    time_start=sliding_window[0, 0],
                    time_end=sliding_window[0, -1],
                )
            ]
    return np.asarray(X), np.asarray(y), np.asarray(list_patient_id)
