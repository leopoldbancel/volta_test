"""Module containing training and inference code."""

import numpy as np
import xgboost
from sklearn import ensemble, model_selection

from test_volta import config, plots

MODELS = {
    "random_forest": ensemble.RandomForestClassifier,
    "xgboost": xgboost.XGBClassifier,
}


def to_readable_label(array):
    """Return label from target 1D vector."""
    if array.astype(int)[0]:
        return "p_signal"
    if array.astype(int)[1]:
        return "qrs_complex"
    if array.astype(int)[2]:
        return "t_signal"
    return "none"


def transform_fonction(array):
    """Return label array from complete target vector."""
    return np.asarray([to_readable_label(arr) for arr in array])


def train_and_compute(
    features: np.ndarray,
    target: np.ndarray,
    list_patient_id: np.ndarray,
    segmentation_type: str = config.SEGMENTATION_TYPE,
):
    """Train model and compute results."""
    # Save dataset repartition plot
    plots.plot_dataset_distribution(dataset=target)
    # Get and split training and testing data
    if segmentation_type == "beat-based":
        # Use beat-based segmentation for train and test data
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            features, target, train_size=config.TRAIN_SIZE
        )
    if segmentation_type == "patient-based":
        # Use patient-based segmentation for train and test data
        max_patient_id_for_training = int(np.max(list_patient_id) * config.TRAIN_SIZE)
        index_max_patient_id = np.where(list_patient_id == max_patient_id_for_training)[
            0
        ][0]
        # Use patient 1 to max_patient_id_for_training for training, the rest of patients for testing
        X_train, X_test = (
            features[:index_max_patient_id],
            features[index_max_patient_id:],
        )
        y_train, y_test = (
            target[:index_max_patient_id],
            target[index_max_patient_id:],
        )
    # Get model from chosen one
    model = MODELS[config.MODEL_TYPE](random_state=42)
    # Fit the model to the data
    model.fit(X_train, y_train)
    # Get the results on test set
    y_pred = model.predict(X_test)
    # Plot the confusion matrix
    plots.plot_confusion_matrix(y_pred=y_pred, y_true=y_test)
