"""Module to pre-process data before inputting to model."""

import numpy as np
import pandas as pd

from test_volta import config


def process_annotations_beat_per_beat(annotations: pd.DataFrame):
    """Process annotations to cut from p signal onset to the next."""
    all_annot_only_p = annotations[
        (
            annotations["symbol"].str.contains("onset_p")
            & (annotations["signal_type"] == config.SIGNAL_USED_FOR_ANNOTATION)
        )
    ]
    p_onsets = []
    for _, patient_df in all_annot_only_p.groupby("pat_id"):
        patient_df_copy = patient_df.reset_index(drop=True)
        patient_df_copy = patient_df_copy.rename(columns={"time": "time_start"})
        patient_df_copy["time_end"] = np.nan
        patient_df_copy.loc[:, "time_end"] = patient_df_copy.loc[:, "time_start"].shift(
            -1
        )
        p_onsets += [patient_df_copy.dropna()]
    return pd.concat(p_onsets)


def gather_signal_beat_per_beat(p_onsets: pd.DataFrame, signals: pd.DataFrame):
    """Build DataFrame containing 12-lead ECG signal for each heart beat."""
    # Normalize signals patient per patient
    all_heartbeats = []
    beat_nb = 0
    for patient_id, patient_df in p_onsets.groupby("pat_id"):
        for _, beat in patient_df.iterrows():
            tmp_df = signals.loc[
                (signals["pat_id"] == patient_id)
                & (signals["time"] >= beat["time_start"])
                & (signals["time"] < beat["time_end"])
            ].copy()
            tmp_df["beat_nb"] = beat_nb
            tmp_df["time_from_p"] = (tmp_df["time"] - beat["time_start"]) / (
                beat["time_end"] - beat["time_start"]
            )
            all_heartbeats += [tmp_df]
            beat_nb += 1
    return pd.concat(all_heartbeats)


def preprocess(signals: pd.DataFrame, annotations: pd.DataFrame):
    """Run the preprocessing chain."""
    # Divide annotation to segment heartbeats
    p_onsets = process_annotations_beat_per_beat(annotations=annotations)
    # Gather ECG signal heartbeat to heart beart
    return gather_signal_beat_per_beat(p_onsets=p_onsets, signals=signals)
