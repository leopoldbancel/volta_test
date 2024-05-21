"""Module to compute and save plots."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics

from test_volta import config, model


def plot_dataset_distribution(dataset: np.ndarray):
    """Plot dataset distribution in histogram format."""
    plt.figure()
    ax = sns.histplot(
        model.transform_fonction(np.asarray(dataset)),
        discrete=True,
        kde=False,
        stat="count",
    )
    ax.bar_label(ax.containers[0])
    plt.savefig(config.RESULT_FOLDER_PATH / "dataset_repartition.png")


def plot_confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray):
    """Plot and return confusion matrix."""
    for normalization in [None, "true", "pred"]:
        plt.figure()
        labels = ["none", "p_signal", "qrs_complex", "t_signal"]
        conf_mat = metrics.confusion_matrix(
            model.transform_fonction(y_true),
            model.transform_fonction(y_pred),
            labels=labels,
            normalize=normalization,
        )
        cmd = metrics.ConfusionMatrixDisplay(conf_mat).plot()
        cmd.ax_.xaxis.set_ticklabels(labels)
        cmd.ax_.yaxis.set_ticklabels(labels)
        plt.savefig(config.RESULT_FOLDER_PATH / f"result_{str(normalization)}.png")
