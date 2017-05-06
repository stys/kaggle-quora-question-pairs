# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression


def platt(labels, predictions):
    """
    Не помогает
    """

    def platt(labels, predictions):
        model = LogisticRegression(fit_intercept=True, solver='lbfgs', C=1.e6)
        model.fit(predictions.reshape(-1, 1), labels)

        def transform(p):
            return model.predict_proba(p.reshape(-1, 1))[:, 1]

        return transform


def reliability_curve(labels, predictions, nbins):
    """
    Диагональные графики
    """

    labels = np.array(labels)
    predictions = np.array(predictions)

    assert len(labels) == len(predictions)
    assert len(labels) >= nbins

    ns = int(len(labels) / nbins)
    rem = len(labels) - ns

    sort_idx = np.argsort(predictions)
    count = np.zeros(nbins)
    avg_pred = np.zeros(nbins)
    avg_label = np.zeros(nbins)

    jbin = 0
    for j, idx in enumerate(sort_idx):
        avg_pred[jbin] += predictions[idx]
        avg_label[jbin] += labels[idx]
        count[jbin] += 1
        if rem > 0 and count[jbin] == ns + 1:
            jbin += 1
            rem -= 1
        elif rem == 0 and count[jbin] == ns:
            jbin += 1

    return avg_label / count, avg_pred / count
