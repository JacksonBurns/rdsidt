import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def parity_plot_with_intervals(
    y_true,
    y_pred,
    y_err,
    outfile,
    ax=None,
    title="Parity Plot",
    point_kwargs=None,
    errorbar_kwargs=None,
):
    """
    Make a parity plot with prediction intervals and summary statistics.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        True values.
    y_pred : array-like, shape (n,)
        Predicted values.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    title : str
        Plot title.
    point_kwargs : dict, optional
        Keyword arguments passed to plt.scatter.
    errorbar_kwargs : dict, optional
        Keyword arguments passed to plt.errorbar.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    point_kwargs = point_kwargs or dict(s=30, alpha=0.8)
    errorbar_kwargs = dict(
        fmt="none", ecolor="gray", alpha=0.6, capsize=2
    )  |errorbar_kwargs

    # Error bars (prediction intervals)
    ax.errorbar(
        y_true,
        y_pred,
        yerr=y_err,
        **errorbar_kwargs,
    )

    # Scatter points
    ax.scatter(y_true, y_pred, **point_kwargs)

    # Parity line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1)

    # Statistics
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    coverage = np.mean((y_true >= (y_pred - y_err)) & (y_true <= (y_pred + y_err)))

    stats_text = (
        f"RMSE = {rmse:.3g}\n"
        f"MAE  = {mae:.3g}\n"
        f"R²   = {r2:.3f}\n"
        f"Coverage = {coverage:.2%}"
    )

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("True Value")
    ax.set_ylabel("Predicted Value")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
