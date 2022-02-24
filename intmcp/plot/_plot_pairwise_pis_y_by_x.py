"""Function for plotting pairwise comparison between policies by x variable """
import os
import math
from typing import Optional, Tuple, Sequence, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import intmcp.plot._parts as parts


def _get_fig_and_axs(num_j_pis: int,
                     **kwargs) -> Tuple[
                         Figure, Sequence[Sequence[Axes]], Sequence[Axes]
                     ]:
    ncols = kwargs.get("ncols", num_j_pis)
    num_plots = num_j_pis
    nrows = math.ceil(num_plots / ncols)

    figsize = kwargs.get("figsize", (9*ncols, 9*nrows))
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=kwargs.get("sharex", True),
        sharey=kwargs.get("sharey", True),
        figsize=figsize,
        squeeze=False
    )

    flattened_axs = []
    for row in range(nrows):
        flattened_axs.extend(axs[row])

    return fig, axs, flattened_axs


def plot_seperate_pairwise_pis_y_by_x(plot_df: pd.DataFrame,
                                      x_key: str,
                                      y_key: str,
                                      y_err_key: Optional[str],
                                      agent_i_id: int,
                                      agent_j_id: int,
                                      results_dir: str,
                                      extra_label: str = '',
                                      **kwargs):
    """Plot pairwise comparison of different agent pis against each other for
    a given x variable.

    This functions plots each pairwise comparison on a different figure
    """
    agent_j_df = plot_df[plot_df[parts.AGENT_ID_KEY] == agent_j_id]
    agent_j_pis, pi_j_labels = parts.get_agent_policies(agent_j_df)

    kwargs["figsize"] = kwargs.get("figsize", (9, 9))
    kwargs['fontsize'] = kwargs.get('fontsize', 'large')

    for i, (pi_j_label, (pi_j, pi_j_nl)) in enumerate(
            zip(pi_j_labels, agent_j_pis)
    ):
        filter_indexes = plot_df[
            (plot_df[parts.AGENT_ID_KEY] == agent_j_id)
            & (
                ~(plot_df[parts.POLICY_KEY] == pi_j)
                | ~(plot_df[parts.NESTING_LVL_KEY] == pi_j_nl)
            )
        ].index
        pi_j_df = plot_df.drop(index=filter_indexes)
        pi_j_label = pi_j_label.replace('\n', "_")

        if i == 0:
            kwargs["show_legend"] = kwargs.get("show_legend", True)
        else:
            kwargs["show_legend"] = False

        plot_pairwise_pis_y_by_x(
            pi_j_df,
            x_key=x_key,
            y_key=y_key,
            y_err_key=y_err_key,
            agent_i_id=agent_i_id,
            agent_j_id=agent_j_id,
            results_dir=results_dir,
            extra_label=f"pi_j={pi_j_label}_{extra_label}",
            **kwargs
        )


def _only_a_single_rpi(agent_df: pd.DataFrame,
                       agent_pis: List[Tuple[str, int]]) -> bool:
    for pi in agent_pis:
        pi_df = agent_df[
            (agent_df[parts.POLICY_KEY] == pi[0])
            & (agent_df[parts.NESTING_LVL_KEY] == pi[1])
        ]
        rollout_pis = pi_df[parts.ROLLOUT_PI_KEY].unique().tolist()
        if len(rollout_pis) > 1:
            return False
    return True


def plot_pairwise_pis_y_by_x(plot_df: pd.DataFrame,
                             x_key: str,
                             y_key: str,
                             y_err_key: Optional[str],
                             agent_i_id: int,
                             agent_j_id: int,
                             results_dir: str,
                             extra_label: str = '',
                             **kwargs):
    """Plot pairwise comparison of different agent pis against each other

    Assumes Agent j and agent i use only a single rollout policy

    """
    agent_i_df = plot_df[plot_df[parts.AGENT_ID_KEY] == agent_i_id]
    agent_i_pis, pi_i_labels = parts.get_agent_policies(agent_i_df)
    assert _only_a_single_rpi(agent_i_df, agent_i_pis)

    agent_j_df = plot_df[plot_df[parts.AGENT_ID_KEY] == agent_j_id]
    agent_j_pis, pi_j_labels = parts.get_agent_policies(agent_j_df)
    assert _only_a_single_rpi(agent_j_df, agent_j_pis)

    fig, _, axs = _get_fig_and_axs(len(agent_j_pis), **kwargs)

    x_labels = plot_df[x_key].unique()
    x_labels.sort()

    fontsize = kwargs.get("fontsize", "x-small")

    for pi_j_idx, (pi_j, pi_j_nl) in enumerate(agent_j_pis):
        j_df = agent_j_df[agent_j_df[parts.POLICY_KEY] == pi_j]
        j_df = j_df[j_df[parts.NESTING_LVL_KEY] == pi_j_nl]
        exp_ids = j_df[parts.EXP_ID_KEY]
        ax = axs[pi_j_idx]

        i_df = agent_i_df[agent_i_df.exp_id.isin(exp_ids)]

        for (pi_i, pi_i_nl), pi_i_label in zip(agent_i_pis, pi_i_labels):
            pi_i_df = i_df[
                (i_df[parts.POLICY_KEY] == pi_i)
                & (i_df[parts.NESTING_LVL_KEY] == pi_i_nl)
            ]
            pi_i_df.sort_values(by=[x_key])

            x = pi_i_df[x_key]
            y = pi_i_df[y_key]
            if y_err_key:
                y_err = pi_i_df[y_err_key]

            if len(x) < len(x_labels) and len(x) == 1:
                x = x_labels
                y = np.full(len(x_labels), y)
                if y_err_key:
                    y_err = np.full(len(x_labels), y_err)

            ax.plot(x, y, label=pi_i_label)
            if y_err_key:
                ax.fill_between(x, y-y_err, y+y_err, alpha=0.1)

            pi_j_label = pi_j_labels[pi_j_idx]
            pi_j_label = pi_j_label.replace("\n", " ")

        if kwargs.get("show_title", True):
            ax.set_title(
                f"Agent={agent_j_id} pi={pi_j_label}", fontsize=fontsize
            )
        if pi_j_idx == 0:
            ax.set_ylabel(kwargs.get('ylabel', y_key), fontsize=fontsize)
        ax.set_xlabel(kwargs.get("xlabel", x_key), fontsize=fontsize)
        if kwargs.get("logx", True):
            ax.semilogx()
        if kwargs.get("ylim", None):
            ax.set_ylim(*kwargs.get("ylim"))

    if kwargs.get("show_legend", True):
        axs[len(agent_j_pis)-1].legend(
            **kwargs.get("legend_kwargs", {})
        )

    fig.tight_layout()
    fig.savefig(
        os.path.join(
            results_dir,
            (
                f"{extra_label}y={y_key}_pairwise_"
                f"{agent_i_id}_vs_{agent_j_id}.png"
            )
        )
    )
