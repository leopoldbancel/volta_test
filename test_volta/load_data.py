"""Module to load data from dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from test_volta import config


def load_all_signals(folder_path: Path):
    """Load all signals from LUDB dataset."""
    signal_list = []
    for patient_id in np.arange(1, 201, 1):
        tmp_df = (
            wfdb.io.rdrecord(folder_path / str(patient_id)).to_dataframe().reset_index()
        )
        tmp_df["index"] = tmp_df["index"].dt.total_seconds()
        tmp_df = tmp_df.rename(columns={"index": "time"})
        tmp_df["pat_id"] = patient_id
        signal_list.append(tmp_df)
    return pd.concat(signal_list)


def readable_symbols(symbols: list):
    """Transform annotation symbol list to readable format."""
    number_in_list = []
    for i, symbol in enumerate(symbols):
        number_in_list += [int(i / 3)]
        if symbol == "(":
            symbols[i] = "onset_" + symbols[i + 1]
        if symbol == ")":
            symbols[i] = "offset_" + symbols[i - 1]
    return symbols, number_in_list


def load_all_annotations(folder_path: Path):
    """Load all annotations and return DataFrame."""
    signal_list = []
    for patient_id in np.arange(1, 201, 1):
        for signal_type in config.SIGNAL_TYPES:
            tmp_dict = wfdb.io.rdann(
                (folder_path / str(patient_id)).as_posix(), extension=signal_type
            )
            symbols, number_in_list = readable_symbols(tmp_dict.symbol)
            tmp_df = pd.DataFrame(
                data=np.array([tmp_dict.sample, symbols, number_in_list]).T,
                columns=["sample", "symbol", "number_in_list"],
            )
            tmp_df["pat_id"] = patient_id
            tmp_df["signal_type"] = signal_type
            tmp_df["time"] = tmp_df["sample"].astype(int) / config.SAMPLING_FREQUENCY
            signal_list += [tmp_df]
    return pd.concat(signal_list)
