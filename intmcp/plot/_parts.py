"""Plotting utility functions and definitions """
import os
from typing import Tuple, List

import numpy as np
import pandas as pd

from intmcp.config import BASE_RESULTS_DIR, IROS_DIR


AGENT_ID_KEY = "agent_id"
EXP_ID_KEY = "exp_id"
MODEL_KEY = "model"
POLICY_KEY = "policy_name"
NUM_SIMS_KEY = "num_sims"
STEP_LIMIT_KEY = "step_limit"
NESTING_LVL_KEY = "nesting_level"
ROLLOUT_PI_KEY = "rollout_policies"

IGNORE_KEYS = [
    "result_dir",
    'show_tree',
    'show_belief',
    'verbose',
    'parallel',
    'catch_errors',
    'verbosity',
]

NUM_EPISODES_COMPLETE_KEY = "num_episodes"
PROPORTION_KEYS = [
    "num_outcome_WIN",
    "num_outcome_LOSS",
    "num_outcome_DRAW",
    "num_outcome_NA",
    "num_errors"
]

POLICY_NAME_MAP = {
    "RandomPolicy": "Random",
    "NestedSearchTree": "I-NTMCP",
}

RANDOM_PI = "RandomPolicy"
NESTED_REASONING_PI = "nestedreasoning"
NESTED_REASONING_LVL_KEY = "reasoning_level"

COMPILED_RESULTS_FILENAME = "compiled_results.csv"


def get_results_file_and_dir(problem_name: str,
                             exp_result_dir: str,
                             iros: bool = False) -> Tuple[str, str]:
    """Get the results file and dir path

    This will first look for "compiled_results.csv" file in the exp_result_dir,
    if it can't find it in this directory it will look in sub directories
    of the exp_result_dir whose name startswith problem_name.
    """
    if iros:
        exp_result_dir = os.path.join(IROS_DIR, exp_result_dir)
    else:
        exp_result_dir = os.path.join(BASE_RESULTS_DIR, exp_result_dir)

    results_dir = exp_result_dir
    results_file = os.path.join(results_dir, COMPILED_RESULTS_FILENAME)
    if os.path.exists(results_file):
        return results_file, results_dir

    for dir_name in os.listdir(exp_result_dir):
        dir_path = os.path.join(exp_result_dir, dir_name)
        if os.path.isdir(dir_path) and dir_name.startswith(problem_name):
            results_dir = dir_path
            results_file = os.path.join(results_dir, COMPILED_RESULTS_FILENAME)
    return results_file, results_dir


def _replace_zero_values(new_value):
    def map_fn(value):
        return value if value != 0 else new_value
    return map_fn


def _replace_non_int_values(new_value):
    def map_fn(value):
        try:
            return int(value)
        except ValueError:
            return new_value
    return map_fn


def _replace_nan_values(new_value):
    def map_fn(value):
        try:
            return new_value if np.isnan(value) else value
        except TypeError:
            return value
    return map_fn


def import_results_df(results_file: str) -> pd.DataFrame:
    """Import experiment results into a pandas df """
    # Have to disable following pylint errors due to pylint not handling df obj
    # pylint: disable=[cell-var-from-loop]
    # pylint: disable=[unsubscriptable-object]
    # pylint: disable=[unsupported-delete-operation]
    # pylint: disable=[no-member]
    # pylint: disable=[unsupported-assignment-operation]
    df: pd.DataFrame = pd.read_csv(results_file, sep=",")

    # remove unwanted columns, for cleanliness
    for header in IGNORE_KEYS:
        if header in df.columns:
            del df[header]

    # need to handle 'None' and '' values since these columns are NST specific
    for col in [NUM_SIMS_KEY, NESTING_LVL_KEY]:
        df[col] = df[col].map(_replace_non_int_values(0))

    # Replace 0 and non-numberic values with the lowest value.
    # Used to cleanly handle non NST policy results
    for col in [NUM_SIMS_KEY]:
        df[col] = df[col].map(_replace_non_int_values(0))
        col_values = df[col].unique()
        col_values.sort()
        if col_values[0] != 0 or len(col_values) == 1:
            new_value = col_values[0]
        else:
            new_value = col_values[1]
        df[col] = df[col].map(_replace_zero_values(new_value))

    df = df.sort_values(by=[NESTING_LVL_KEY, NUM_SIMS_KEY])

    for col in [ROLLOUT_PI_KEY]:
        if col not in df.columns:
            continue
        df[col] = df[col].map(_replace_nan_values('None'))

    # Add 95% CI column
    def conf_int(row, prefix):
        std = row[f"{prefix}_std"]
        n = row[NUM_EPISODES_COMPLETE_KEY]
        return 1.96 * (std / np.sqrt(n))

    prefix = ""
    for col in df.columns:
        if not col.endswith("_std"):
            continue
        prefix = col.replace("_std", "")
        df[f"{prefix}_CI"] = df.apply(
            lambda row: conf_int(row, prefix), axis=1
        )

    # Add Proportion columns
    col = ""
    for col in df.columns:
        if col not in PROPORTION_KEYS:
            continue
        df[f"proportion_{col}"] = df.apply(
            lambda row: row[col] / row[NUM_EPISODES_COMPLETE_KEY], axis=1
        )

    return df


def load_results(problem_name: str,
                 exp_results_dir: str,
                 iros: bool) -> Tuple[pd.DataFrame, str]:
    """Load results for given problem and exp_results dir.

    This is a convinience function that calls the get_results_file_and_dir
    then import_results_df.

    Assumes exp_result_dir is located in intmcp.config.BASE_RESULTS_DIR.

    Returns Pandas Dataframe containing results and full path to the results
    directory (which is useful for saving result plots)
    """
    results_file, results_dir = get_results_file_and_dir(
        problem_name, exp_results_dir, iros
    )
    return import_results_df(results_file), results_dir


def results_summary(df: pd.DataFrame, include_headers: bool = False) -> str:
    """Generates a simple summary of results """
    num_exps = len(df[EXP_ID_KEY].unique())
    summary = [f"Num Exps = {num_exps}"]
    for (key, name) in [
            (AGENT_ID_KEY, "Agent IDs"),
            (NUM_SIMS_KEY, "Num Sims"),
            (NESTING_LVL_KEY, "Nesting levels"),
            (STEP_LIMIT_KEY, "Step Limits"),
            (POLICY_KEY, "Policy Classes"),
            (ROLLOUT_PI_KEY, "Rollout Policies")
    ]:
        if key not in df.columns:
            continue
        key_vals = df[key].unique()
        key_vals.sort()
        summary.append(f"{name} = {key_vals}")

    if include_headers:
        summary.append(f"Headers = {df.columns.values}")

    return "\n".join(summary)


def get_agent_policies(agent_df: pd.DataFrame
                       ) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Get the list of (policy class, nesting level) from agent df """
    agent_pis = []
    labels = []

    pi_names = list(agent_df[POLICY_KEY].unique())
    pi_names.sort()

    if RANDOM_PI in pi_names:
        pi_names.remove(RANDOM_PI)
        pi_names.insert(0, RANDOM_PI)

    for pi in pi_names:
        pi_df = agent_df[agent_df[POLICY_KEY] == pi]
        nls = pi_df[NESTING_LVL_KEY].unique()
        nls.sort()

        if pi in POLICY_NAME_MAP:
            pi_label = POLICY_NAME_MAP[pi]
        elif 'greedy' in str(pi).lower():
            pi_label = 'Greedy'
        elif NESTED_REASONING_PI in str(pi).lower():
            pi_label = 'FNR'
        else:
            pi_label = str(pi)

        if len(nls) > 1 or pi_label in ("NST", "NR"):
            for nl in nls:
                agent_pis.append((pi, nl))
                labels.append(f"{pi_label} l={nl}")
        else:
            agent_pis.append((pi, nls[0]))
            labels.append(pi_label)

    return agent_pis, labels
