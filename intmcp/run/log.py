"""Utility functions for running INTMCP """
import os
import csv
import pathlib
import logging
from os import listdir
from datetime import datetime
from typing import Optional, Sequence, Any

import pandas as pd
from prettytable import PrettyTable

import intmcp.model as M
import intmcp.log as log_lib
from intmcp.run import stats
from intmcp.config import BASE_RESULTS_DIR


LINE_BREAK = "-"*60
MAJOR_LINE_BREAK = "="*60
EXCEPTION_LIMIT = 10


def get_logger(logger: Optional[logging.Logger] = None) -> logging.Logger:
    """Get logger to use """
    return logging.getLogger() if logger is None else logger


def initial_timestep(timestep: M.JointTimestep,
                     logger: Optional[logging.Logger] = None,
                     render_asci: bool = True):
    """Display message for initial state """
    logger = get_logger(logger)
    state, obs, _, _ = timestep
    logging.log(
        log_lib.INFO1,
        "Episode Map:\nInitial Actual state:\n%s\nInitial obs=%s",
        state.render_asci() if render_asci else state,
        obs.render_asci() if render_asci else obs
    )


def joint_step(step_num: int,
               action: M.JointAction,
               timestep: M.JointTimestep,
               logger: Optional[logging.Logger] = None,
               render_asci: bool = True):
    """Display message for a joint step """
    logger = get_logger(logger)
    next_state, obs, rewards, _ = timestep
    logger.log(
        log_lib.INFO1,
        "%s\nStep=%d\na=%s\no=%s\nr=%s\ns'=%s\n%s",
        LINE_BREAK,
        step_num,
        action,
        obs.render_asci() if render_asci else obs,
        rewards,
        next_state.render_asci() if render_asci else obs,
        LINE_BREAK
    )


def episode_start(episode_num: int,
                  logger: Optional[logging.Logger] = None):
    """Display message for start of an episode """
    logger = get_logger(logger)
    logger.log(
        log_lib.INFO1,
        "%s\nEpisode %d Start\n%s",
        MAJOR_LINE_BREAK,
        episode_num,
        MAJOR_LINE_BREAK
    )


def episode_end(episode_num: int,
                statistics: stats.AgentStatisticsMap,
                logger: Optional[logging.Logger] = None):
    """Display message for start of an episode """
    logger = get_logger(logger)
    logger.log(
        log_lib.INFO1,
        "%s\nEpisode %d Complete\n%s",
        LINE_BREAK,
        episode_num,
        format_as_table(statistics)
    )


def episode_error(episode_num: int,
                  error: Exception,
                  logger: Optional[logging.Logger] = None):
    """Display message for episode error """
    logger = get_logger(logger)
    logger.error(
        "%s\nEpisode %d Error: %s",
        LINE_BREAK,
        episode_num,
        str(error)
    )


def simulation_start(num_episodes: int,
                     time_limit: Optional[int],
                     logger: Optional[logging.Logger] = None):
    """Display message for end of simulations """
    logger = get_logger(logger)
    logger.info(
        "%s\nRunning %d episodes with Time Limit = %s s\n%s",
        MAJOR_LINE_BREAK,
        num_episodes,
        str(time_limit),
        MAJOR_LINE_BREAK
    )


def time_limit_reached(num_episodes: int,
                       time_limit: int,
                       logger: Optional[logging.Logger] = None):
    """Display message for time limit being reached """
    logger = get_logger(logger)
    logger.info(
        "%s\nTime limit of %d s reached after %d episodes",
        MAJOR_LINE_BREAK,
        time_limit,
        num_episodes
    )


def simulation_end(statistics: stats.AgentStatisticsMap,
                   logger: Optional[logging.Logger] = None):
    """Display message for end of simulations """
    logger = get_logger(logger)
    logger.info(
        "%s\nSimulations Complete\n%s\n%s",
        MAJOR_LINE_BREAK,
        format_as_table(statistics),
        MAJOR_LINE_BREAK
    )


def progress(ep_num: int,
             num_episodes: int,
             logger: Optional[logging.Logger] = None):
    """Display progress message """
    logger = get_logger(logger)
    display_freq = max(1, num_episodes // 10)
    if (ep_num+1) % display_freq == 0:
        logger.info("Episode %d / %d complete", ep_num+1, num_episodes)


def make_dir(exp_name: str) -> str:
    """Makes a new experiment results directory at """
    result_dir = os.path.join(BASE_RESULTS_DIR, f"{exp_name}_{datetime.now()}")
    pathlib.Path(result_dir).mkdir(exist_ok=True)
    return result_dir


def compile_results(result_dir: str,
                    extra_output_dir: Optional[str] = None) -> str:
    """Compile all .tsv results files in a directory into a single file.

    If extra_output_dir is provided then will additionally compile_result to
    the extra_output_dir.
    """
    result_filepaths = [
        os.path.join(result_dir, f) for f in listdir(result_dir)
        if os.path.isfile(os.path.join(result_dir, f)) and f.endswith(".csv")
    ]

    concat_results_filepath = os.path.join(result_dir, "compiled_results.csv")

    dfs = map(pd.read_csv, result_filepaths)
    concat_df = pd.concat(dfs)
    concat_df.to_csv(concat_results_filepath)

    if extra_output_dir:
        extra_results_filepath = os.path.join(
            extra_output_dir, "compiled_results.csv"
        )
        concat_df.to_csv(extra_results_filepath)

    return concat_results_filepath


def format_as_table(values: stats.AgentStatisticsMap) -> str:
    """format values as a table """
    table = PrettyTable()

    agent_ids = list(values)
    table.field_names = ["AgentID"] + [str(i) for i in agent_ids]

    for row_name in list(values[agent_ids[0]].keys()):
        row = [row_name]
        for i in agent_ids:
            agent_row_value = values[i][row_name]
            if isinstance(agent_row_value, float):
                row.append(f"{agent_row_value:.4f}")
            else:
                row.append(str(agent_row_value))
        table.add_row(row)

    table.align = 'r'
    table.align["AgentID"] = 'l'   # type: ignore
    return table.get_string()


class CSVWriter:
    """A logging object to write to CSV files.

    Each 'write()' takes an 'OrderedDict', creating one column in the CSV file
    for each dictionary key on the first call. Subsequent calls to 'write()'
    must contain the same dictionary keys.

    Inspired by:
    https://github.com/deepmind/dqn_zoo/blob/master/dqn_zoo/parts.py
    """

    def __init__(self, fname: str):
        dirname = os.path.dirname(fname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self._fname = fname
        self._header_written = False
        self._fieldnames: Sequence[Any] = []

    def write(self, statistics: stats.AgentStatisticsMap) -> None:
        """Appends given statistics as new rows to CSV file.

        1 row per agent entry in the AgentStatisticsMap.
        Assumes all agent's statistics maps share the same headers
        """
        agent_ids = list(statistics)
        if self._fieldnames == []:
            self._fieldnames = list(statistics[agent_ids[0]].keys())

        # Open a file in 'append' mode, so we can continue logging safely to
        # the same file if needed.
        with open(self._fname, 'a') as fout:
            # Always use same fieldnames to create writer, this way a
            # consistency check is performed automatically on each write.
            writer = csv.DictWriter(fout, fieldnames=self._fieldnames)
            # Write a header if this is the very first write.
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            for i in agent_ids:
                writer.writerow(statistics[i])

    def close(self) -> None:
        """Closes the `CsvWriter`."""
