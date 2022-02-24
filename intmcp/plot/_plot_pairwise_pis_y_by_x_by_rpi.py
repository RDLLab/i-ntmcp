"""Function for plotting pairwise comparison between policies by x variable """
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import intmcp.plot._parts as parts


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
    agent_i_rollout_pis = agent_i_df[parts.ROLLOUT_PI_KEY].unique().tolist()
    agent_i_rollout_pis.sort()

    agent_j_df = plot_df[plot_df[parts.AGENT_ID_KEY] == agent_j_id]
    agent_j_pis, pi_j_labels = parts.get_agent_policies(agent_j_df)
    agent_j_rollout_pis = agent_j_df[parts.ROLLOUT_PI_KEY].unique().tolist()
    agent_j_rollout_pis.sort()

    nrows = len(agent_j_rollout_pis)
    ncols = len(agent_j_pis)
    figsize = (9*ncols, 9*nrows)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=kwargs.get("sharex", True),
        sharey=kwargs.get("sharey", True),
        figsize=figsize,
        squeeze=False
    )

    x_labels = plot_df[x_key].unique()
    x_labels.sort()

    for pi_j_idx, (pi_j, pi_j_nl) in enumerate(agent_j_pis):
        j_df = agent_j_df[agent_j_df[parts.POLICY_KEY] == pi_j]
        j_df = j_df[j_df[parts.NESTING_LVL_KEY] == pi_j_nl]

        for rpi_j_idx, rpi_j in enumerate(agent_j_rollout_pis):
            rpi_j_df = j_df[j_df[parts.ROLLOUT_PI_KEY] == rpi_j]

            exp_ids = rpi_j_df[parts.EXP_ID_KEY]
            ax = axs[rpi_j_idx][pi_j_idx]

            i_df = agent_i_df[agent_i_df.exp_id.isin(exp_ids)]

            for (pi_i, pi_i_nl), pi_i_label in zip(agent_i_pis, pi_i_labels):
                pi_i_df = i_df[
                    (i_df[parts.POLICY_KEY] == pi_i)
                    & (i_df[parts.NESTING_LVL_KEY] == pi_i_nl)
                ]
                pi_i_df.sort_values(by=[x_key])

                # TODO add rollout policy

                x = pi_i_df[x_key]
                y = pi_i_df[y_key]
                if y_err_key:
                    y_err = pi_i_df[y_err_key]

                if len(x) < len(x_labels) and len(x) == 1:
                    x = x_labels
                    y = np.full(len(x_labels), y)
                    if y_err_key:
                        y_err = np.full(len(x_labels), y_err)

                # rpi_i_label = str(rpi_i)
                # if 'greedy' in rpi_i_label.lower():
                #     rpi_i_label = 'Greedy'

                ax.plot(x, y, label=pi_i_label)
                if y_err_key:
                    ax.fill_between(x, y-y_err, y+y_err, alpha=0.1)

            rpi_j_label = str(rpi_j)
            if 'greedy' in rpi_j_label.lower():
                rpi_j_label = 'Greedy'

            ax.set_title(f"Agent={agent_j_id} pi={pi_j_labels[pi_j_idx]}")
            if pi_j_idx == 0:
                y_label = (
                    f"Rollout Pi={rpi_j_label}\n{kwargs.get('ylabel', y_key)}"
                )
                ax.set_ylabel(y_label)
            ax.set_xlabel(kwargs.get("xlabel", x_key))
            if kwargs.get("logx", True):
                ax.semilogx()

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
