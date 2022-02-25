"""Function for plotting bar plot of pairwise comparison between policies """
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import intmcp.plot._parts as parts

# pylint: disable=[too-many-statements]

BAR_WIDTH = 0.8


def plot_categorical_line(plot_df: pd.DataFrame,
                          y_key: str,
                          y_err_key: Optional[str],
                          agent_i_id: int,
                          agent_j_id: int,
                          results_dir: str,
                          extra_label: str = '',
                          **kwargs):
    """Plot pairwise comparison of different agent pis against each other.

    Assumes Agent j and agent i use only a single rollout policy.
    """
    agent_i_df = plot_df[plot_df[parts.AGENT_ID_KEY] == agent_i_id]
    agent_i_pis, pi_i_labels = parts.get_agent_policies(agent_i_df)
    agent_i_rollout_pis = agent_i_df[parts.ROLLOUT_PI_KEY].unique().tolist()
    assert len(agent_i_rollout_pis) == 1

    agent_j_df = plot_df[plot_df[parts.AGENT_ID_KEY] == agent_j_id]
    agent_j_pis, pi_j_labels = parts.get_agent_policies(agent_j_df)
    agent_j_rollout_pis = agent_j_df[parts.ROLLOUT_PI_KEY].unique().tolist()
    assert len(agent_j_rollout_pis) == 1

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=kwargs.get("figsize", (18, 9))
    )

    x_pos = np.arange(len(agent_i_pis), dtype=np.float64)

    lines, labels = [], []

    for pi_j_idx, (pi_j, pi_j_nl) in enumerate(agent_j_pis):
        j_df = agent_j_df[agent_j_df[parts.POLICY_KEY] == pi_j]
        j_df = j_df[j_df[parts.NESTING_LVL_KEY] == pi_j_nl]
        exp_ids = j_df[parts.EXP_ID_KEY]

        i_ys = np.zeros(len(agent_i_pis))
        i_yerrs = np.zeros(len(agent_i_pis))

        pi_j_label = pi_j_labels[pi_j_idx]

        for exp_id in exp_ids:
            i_df = agent_i_df[agent_i_df[parts.EXP_ID_KEY] == exp_id]

            pi_i = i_df[parts.POLICY_KEY].values[0]
            pi_i_nl = i_df[parts.NESTING_LVL_KEY].values[0]
            pi_i_idx = agent_i_pis.index((pi_i, pi_i_nl))

            i_ys[pi_i_idx] = i_df[y_key]
            if y_err_key:
                i_yerrs[pi_i_idx] = i_df[y_err_key]

        line, = ax.plot(x_pos, i_ys, label=pi_j_label)
        if y_err_key:
            ax.fill_between(x_pos, i_ys-i_yerrs, i_ys+i_yerrs, alpha=0.1)

        lines.append(line)
        labels.append(pi_j_label)

    fontsize = kwargs.get("fontsize", "x-small")
    ax.set_ylabel(kwargs.get('ylabel', y_key), fontsize=fontsize)
    ax.set_xlabel(
        kwargs.get("xlabel", f"Agent {agent_i_id} Policy"), fontsize=fontsize
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(kwargs.get("x_tick_labels", pi_i_labels))
    if kwargs.get("ylim", None):
        ax.set_ylim(*kwargs.get("ylim"))

    if kwargs.get("show_legend", True):
        ax.legend(
            lines,
            labels,
            loc='lower left',
            bbox_to_anchor=(1.01, 0.0),
            ncol=1,
            borderaxespad=0,
            frameon=False,
            fontsize=fontsize
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
