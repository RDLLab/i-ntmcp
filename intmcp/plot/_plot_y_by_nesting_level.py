"""Function for plotting y by nesting level """
import os
from typing import Optional
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import intmcp.plot._parts as parts


def plot_y_by_nesting_level(plot_df: pd.DataFrame,
                            y_key: str,
                            y_err_key: Optional[str],
                            row_key: str,
                            col_key: str,
                            results_dir: str,
                            extra_label: str = '',
                            logy: bool = False):
    """Plot Y by nesting level by policy classe """
    # Different rows of plots
    row_labels = plot_df[row_key].unique()
    row_labels.sort()

    # Different columns of plots
    col_labels = plot_df[col_key].unique()
    col_labels.sort()

    # X-Axis for each plot
    x_key = "nesting_level"
    x_labels = plot_df[x_key].unique()
    x_labels.sort()

    pi_classes = plot_df[parts.POLICY_KEY].unique()

    nrows = len(row_labels)
    ncols = len(col_labels)
    figsize = (9*nrows, 9*ncols)
    fig, axs = plt.subplots(
        nrows=ncols,
        ncols=nrows,
        sharex=True,
        sharey=True,
        figsize=figsize,
        squeeze=False
    )

    for i, (row_id, pi) in enumerate(product(row_labels, pi_classes)):
        row_df = plot_df[plot_df[row_key] == row_id]
        pi_df = row_df[row_df[parts.POLICY_KEY] == pi]

        for j, col_id in enumerate(col_labels):
            ax = axs[j][i]

            pi_nesting_levels = pi_df[parts.NESTING_LVL_KEY].unique()
            pi_nesting_levels.sort()
            pi_label = parts.POLICY_NAME_MAP.get(pi, pi)

            pi_df = pi_df.sort_values(by=[x_key])

            x = pi_df[x_key]
            y = pi_df[y_key]

            if y_err_key:
                err = pi_df[y_err_key]

            if len(x) < len(x_labels) and len(x) == 1:
                x = x_labels
                y = np.full(len(x_labels), y)
                if y_err_key:
                    err = np.full(len(x_labels), err)

            ax.plot(x, y, label=pi_label)
            if y_err_key:
                ax.fill_between(x, y-err, y+err, alpha=0.1)

            ax.set_title(f"{row_key}={row_id} {col_key}={col_id}")
            ax.set_xlabel(x_key)
            ax.set_ylabel(y_key)
            if logy:
                ax.set_yscale('symlog')

        ax.legend()

    fig.tight_layout()
    fig_title = f"{extra_label}y={y_key}_x={x_key}"
    if logy:
        fig_title += "_logy"
    fig.savefig(os.path.join(results_dir, f"{fig_title}.png"))
