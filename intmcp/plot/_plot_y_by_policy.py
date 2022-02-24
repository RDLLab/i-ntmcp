"""Function for plotting y by Policy """
import os
from typing import Optional
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import intmcp.plot._parts as parts


def plot_y_by_policy(plot_df: pd.DataFrame,
                     y_key: str,
                     y_err_key: Optional[str],
                     x_key: str,
                     row_key: str,
                     col_key: str,
                     results_dir: str,
                     extra_label: str = '',
                     **kwargs):
    """Plot Y by policy classes """
    # Different rows of plots
    row_labels = plot_df[row_key].unique()
    row_labels.sort()

    # Different columns of plots
    col_labels = plot_df[col_key].unique()
    col_labels.sort()

    # X-Axis for each plot
    x_labels = plot_df[x_key].unique()
    x_labels.sort()

    policy_classes = plot_df[parts.POLICY_KEY].unique()

    nrows = len(row_labels)
    ncols = len(col_labels)
    figsize = (9*ncols, 9*nrows)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=kwargs.get("sharex", True),
        sharey=kwargs.get("sharey", True),
        figsize=figsize,
        squeeze=False
    )

    for i, (row_id, pi) in enumerate(
            product(row_labels, policy_classes)
    ):
        row_df = plot_df[plot_df[row_key] == row_id]
        pi_df = row_df[row_df[parts.POLICY_KEY] == pi]
        pi_col_values = pi_df[col_key].unique()

        if len(pi_col_values) < len(col_labels):
            col_dfs = [
                pi_df[pi_df[col_key] == pi_col_values[0]]
                for _ in col_labels
            ]
        else:
            col_dfs = [
                pi_df[pi_df[col_key] == col_id]
                for col_id in col_labels
            ]

        for j, (col_df, col_id) in enumerate(zip(col_dfs, col_labels)):
            ax = axs[i][j]

            pi_nesting_levels = col_df[parts.NESTING_LVL_KEY].unique()
            pi_nesting_levels.sort()

            for level in pi_nesting_levels:
                nesting_df = col_df[col_df[parts.NESTING_LVL_KEY] == level]
                nesting_df = nesting_df.sort_values(by=[x_key])
                x = nesting_df[x_key]
                y = nesting_df[y_key]

                if y_err_key:
                    err = nesting_df[y_err_key]

                if len(x) < len(x_labels) and len(x) == 1:
                    x = x_labels
                    y = np.full(len(x_labels), y)
                    if y_err_key:
                        err = np.full(len(x_labels), err)

                label = f"{parts.POLICY_NAME_MAP.get(pi, pi)}"
                if len(pi_nesting_levels) > 1:
                    label += f", l={level}"

                ax.plot(x, y, label=label)

                if y_err_key:
                    ax.fill_between(x, y-err, y+err, alpha=0.1)

            if "ax_titles" in kwargs:
                ax.set_title(kwargs["ax_titles"][row_id][col_id])
            else:
                ax.set_title(f"{row_key}={row_id} {col_key}={col_id}")

            ax.set_xlabel(kwargs.get("xlabel", x_key))
            ax.set_ylabel(kwargs.get("ylabel", y_key))
            if kwargs.get("logx", True):
                ax.semilogx()

        ax.legend()

    fig.tight_layout()
    fig.savefig(
        os.path.join(results_dir, f"{extra_label}y={y_key}_x={x_key}.png")
    )
