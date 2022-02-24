"""Function for plotting bar plot of pairwise comparison between policies """
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import intmcp.plot._parts as parts

# pylint: disable=[too-many-statements]


BAR_WIDTH = 0.4


def plot_pairwise_pis_by_rollout_pis_bar(plot_df: pd.DataFrame,
                                         y_key: str,
                                         y_err_key: Optional[str],
                                         agent_i_id: int,
                                         agent_j_id: int,
                                         results_dir: str,
                                         extra_label: str = '',
                                         **kwargs):
    """Plot pairwise comparison of different agent pis and rollout pis.

    Plots a different column for each agent j policy and
    a different row for each agent j rollout policy

    Each plot shows the y-value for each agent i policy, rollout policy pair
    against the given agent j policy (col), rollout policy (row) pair
    """
    agent_i_df = plot_df[plot_df[parts.AGENT_ID_KEY] == agent_i_id]
    agent_i_pis, pi_i_labels = parts.get_agent_policies(agent_i_df)
    agent_i_rollout_pis = agent_i_df[parts.ROLLOUT_PI_KEY].unique().tolist()
    agent_i_rollout_pis.sort()

    agent_j_df = plot_df[plot_df[parts.AGENT_ID_KEY] == agent_j_id]
    agent_j_pis, pi_j_labels = parts.get_agent_policies(agent_j_df)
    agent_j_rollout_pis = agent_j_df[parts.ROLLOUT_PI_KEY].unique().tolist()
    agent_j_rollout_pis.sort()

    ncols = len(agent_j_pis)
    nrows = len(agent_j_rollout_pis)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=kwargs.get("sharex", True),
        sharey=kwargs.get("sharey", True),
        figsize=(9*ncols, 9*nrows),
        squeeze=False
    )

    x_pos = np.arange(len(agent_i_pis))
    bar_width = kwargs.get("bar_width", BAR_WIDTH)
    rpi_i_bar_offset = [i*bar_width for i in range(len(agent_i_rollout_pis))]

    for pi_j_idx, (pi_j, pi_j_nl) in enumerate(agent_j_pis):
        j_df = agent_j_df[agent_j_df[parts.POLICY_KEY] == pi_j]
        j_df = j_df[j_df[parts.NESTING_LVL_KEY] == pi_j_nl]

        for rpi_j_idx, rpi_j in enumerate(agent_j_rollout_pis):
            rpi_j_df = j_df[j_df[parts.ROLLOUT_PI_KEY] == rpi_j]
            exp_ids = rpi_j_df[parts.EXP_ID_KEY]
            ax = axs[rpi_j_idx][pi_j_idx]

            i_ys = [
                np.zeros(len(agent_i_pis))
                for _ in range(len(agent_i_rollout_pis))
            ]
            i_yerrs = [
                np.zeros(len(agent_i_pis))
                for _ in range(len(agent_i_rollout_pis))
            ]

            for exp_id in exp_ids:
                i_df = agent_i_df[agent_i_df[parts.EXP_ID_KEY] == exp_id]

                pi_i = i_df[parts.POLICY_KEY].values[0]
                pi_i_nl = i_df[parts.NESTING_LVL_KEY].values[0]
                pi_i_idx = agent_i_pis.index((pi_i, pi_i_nl))

                rpi_i = i_df[parts.ROLLOUT_PI_KEY].values[0]
                rpi_i_idx = agent_i_rollout_pis.index(rpi_i)

                i_ys[rpi_i_idx][pi_i_idx] = i_df[y_key]
                if y_err_key:
                    i_yerrs[rpi_i_idx][pi_i_idx] = i_df[y_err_key]

            for rpi_i_idx, rpi_i in enumerate(agent_i_rollout_pis):
                rpi_i_x_pos = x_pos + rpi_i_bar_offset[rpi_i_idx]

                bar_kwargs = {'align': 'edge'}
                if y_err_key:
                    bar_kwargs['yerr'] = i_yerrs[rpi_i_idx]

                rpi_i_label = str(rpi_i)
                if 'greedy' in rpi_i_label.lower():
                    rpi_i_label = 'Greedy'

                ax.bar(
                    rpi_i_x_pos,
                    i_ys[rpi_i_idx],
                    bar_width,
                    label=rpi_i_label,
                    **bar_kwargs
                )

            rpi_j_label = str(rpi_j)
            if 'greedy' in rpi_j_label.lower():
                rpi_j_label = 'Greedy'

            ax.set_title(f"Agent={agent_j_id} pi={pi_j_labels[pi_j_idx]}")
            if pi_j_idx == 0:
                y_label = (
                    f"Rollout Pi={rpi_j_label}\n{kwargs.get('ylabel', y_key)}"
                )
                ax.set_ylabel(y_label)
            ax.set_xlabel(f"Agent={agent_i_id}")
            ax.set_xticks(x_pos + bar_width)
            ax.set_xticklabels(pi_i_labels)

    axs[-1][-1].legend()
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
