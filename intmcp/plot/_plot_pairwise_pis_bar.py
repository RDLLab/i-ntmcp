"""Function for plotting bar plot of pairwise comparison between policies """
import os
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import intmcp.plot._parts as parts

# pylint: disable=[too-many-statements]


BAR_WIDTH = 0.8


def plot_seperate_pairwise_pis_bar(plot_df: pd.DataFrame,
                                   y_key: str,
                                   y_err_key: Optional[str],
                                   agent_i_id: int,
                                   agent_j_id: int,
                                   results_dir: str,
                                   extra_label: str = '',
                                   **kwargs):
    """Plot pairwise comparison of different agent pis against each other, with
    a different plot for each policy for agent j.
    """
    agent_j_df = plot_df[plot_df[parts.AGENT_ID_KEY] == agent_j_id]
    agent_j_pis, pi_j_labels = parts.get_agent_policies(agent_j_df)

    kwargs["bar_width"] = kwargs.get("bar_width", 0.4)
    kwargs["figsize"] = kwargs.get("figsize", (9, 9))
    kwargs["show_legend"] = kwargs.get("show_legend", False)
    kwargs['fontsize'] = kwargs.get('fontsize', 'large')

    for pi_j_label, (pi_j, pi_j_nl) in zip(pi_j_labels, agent_j_pis):
        filter_indexes = plot_df[
            (plot_df[parts.AGENT_ID_KEY] == agent_j_id)
            & (
                ~(plot_df[parts.POLICY_KEY] == pi_j)
                | ~(plot_df[parts.NESTING_LVL_KEY] == pi_j_nl)
            )
        ].index
        pi_j_df = plot_df.drop(index=filter_indexes)
        pi_j_label = pi_j_label.replace('\n', "_")
        plot_pairwise_pis_bar(
            pi_j_df,
            y_key=y_key,
            y_err_key=y_err_key,
            agent_i_id=agent_i_id,
            agent_j_id=agent_j_id,
            results_dir=results_dir,
            extra_label=f"pi_j={pi_j_label}_{extra_label}",
            **kwargs
        )


def plot_pairwise_pis_bar(plot_df: pd.DataFrame,
                          y_key: str,
                          y_err_key: Optional[str],
                          agent_i_id: int,
                          agent_j_id: int,
                          results_dir: str,
                          extra_label: str = '',
                          **kwargs):
    """Plot pairwise comparison of different agent pis against each other.

    Assumes Agent j and agent i use only a single rollout policy.

    Within a single plot, plots the y-value for each agent i policy (x-axis)
    against each agent j policy (z-axis/different color bars).
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

    bar_width = kwargs.get("bar_width", BAR_WIDTH)
    total_width = len(agent_j_pis) * bar_width
    x_pos = np.arange(len(agent_i_pis), dtype=np.float64)
    # + bar_width to add gap between sets of bars
    x_pos *= total_width + bar_width
    pi_j_bar_offset = [i*bar_width for i in range(len(agent_j_pis))]

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

        pi_j_x_pos = x_pos + pi_j_bar_offset[pi_j_idx]

        bar_kwargs: Dict[str, Any] = {'align': 'edge'}
        if y_err_key:
            bar_kwargs['yerr'] = i_yerrs

        ax.bar(
            pi_j_x_pos,
            i_ys,
            bar_width,
            label=pi_j_label,
            **bar_kwargs
        )

    fontsize = kwargs.get("fontsize", "x-small")
    ax.set_ylabel(kwargs.get('ylabel', y_key), fontsize=fontsize)
    ax.set_xlabel(f"Agent {agent_i_id} Policy", fontsize=fontsize)
    ax.set_xticks(x_pos + total_width/2)
    ax.set_xticklabels(pi_i_labels)
    if kwargs.get("ylim", None):
        ax.set_ylim(*kwargs.get("ylim"))

    if kwargs.get("show_legend", True):
        ax.legend(
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
