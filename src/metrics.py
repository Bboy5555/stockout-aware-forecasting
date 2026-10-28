"""Shared metrics so every stream scores results the same way.

All functions take arrays (or pandas columns) of actual and predicted values.
"""
import numpy as np
import pandas as pd


def wape(actual, pred):
    """Weighted absolute percentage error: total absolute error / total actual. Lower is better."""
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return np.abs(actual - pred).sum() / actual.sum()


def mae(actual, pred):
    """Mean absolute error, in sales units. Lower is better."""
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return np.abs(actual - pred).mean()


def bias(actual, pred):
    """Total over- or under-forecast as a share of actual. Negative = under-forecasting."""
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return (pred - actual).sum() / actual.sum()


def pinball(actual, pred_q, q):
    """Quantile (pinball) loss for a forecast of quantile q (e.g. q=0.3). Lower is better."""
    actual, pred_q = np.asarray(actual, float), np.asarray(pred_q, float)
    diff = actual - pred_q
    return np.mean(np.maximum(q * diff, (q - 1) * diff))


def score(df, actual_col="actual", pred_col="pred", by=None):
    """Table of WAPE, MAE and bias, overall or per group (e.g. by='first_category_id')."""
    def _row(g):
        return pd.Series({
            "wape": wape(g[actual_col], g[pred_col]),
            "mae": mae(g[actual_col], g[pred_col]),
            "bias": bias(g[actual_col], g[pred_col]),
            "n": len(g),
        })
    if by is None:
        return _row(df).to_frame("overall").T
    return df.groupby(by).apply(_row)


def recovery_error(true_hidden, recovered):
    """For Stream 2's masking test: WAPE and bias on the hours that were hidden."""
    return {"wape": wape(true_hidden, recovered), "bias": bias(true_hidden, recovered)}
