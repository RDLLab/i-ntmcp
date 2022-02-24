"""Module that contains usefull functions for running experiments """
import os
import time
import random
import logging
from pprint import pformat
import multiprocessing as mp
from itertools import product
from collections import namedtuple
from typing import List, Optional, Dict, Type, Tuple, Any

import yaml
import numpy as np

import intmcp.log as log_lib
from intmcp import model as M
from intmcp.config import N_PROCS
from intmcp import envs as env_lib
from intmcp import policy as policy_lib
from intmcp.tree import NestedSearchTree

from intmcp.run import log as run_log
from intmcp.run.runner import run_sims
from intmcp.run import stats as stats_lib
from intmcp.run import render as render_lib


EXP_LOG_LEVEL = logging.INFO+1


# YAML FILE KEYS
ENV_NAME = "env_name"
RUN_KWARGS = "run_kwargs"
MODELS = "models"
POLICIES = "policies"
POLICY_CLASS = "class"
ROLLOUT_POLICIES = "rollout_policies"
GAMMA = "gamma"


ModelParams = namedtuple("ModelParams", ["cls", "kwargs"])
PolicyParams = namedtuple("PolicyParams", ["cls", "kwargs"])
ExpParams = namedtuple(
    "ExpParams",
    ["exp_id", "m_params", "pi_params_list", "r_kwargs", "result_dir"]
)


lock = mp.Lock()


def _init_lock(lck):
    global lock
    lock = lck


def _log_exp_start(exp_params: ExpParams, logger: logging.Logger):
    m_params = exp_params.m_params
    pi_params_list = exp_params.pi_params_list
    r_kwargs = exp_params.r_kwargs

    lock.acquire()
    try:
        logger.info(run_log.LINE_BREAK)
        logger.info(f"Running exp num {exp_params.exp_id} with:")
        logger.info(f"Model class = {m_params.cls}")
        logger.info("Model kwargs:")
        logger.info(pformat(m_params.kwargs))

        for i, params in enumerate(pi_params_list):
            logger.info(f"Agent = {i} Policy class = {params.cls}")
            logger.info("Policy kwargs:")
            logger.info(pformat(params.kwargs))

        logger.info("Run kwargs:")
        logger.info(pformat(r_kwargs))
        logger.info(f"Result dir = {exp_params.result_dir}")
        logger.info(run_log.LINE_BREAK)
    finally:
        lock.release()


def _get_exp_results_fname(exp_params: ExpParams) -> str:
    fname = f"exp_{exp_params.exp_id}.csv"
    return os.path.join(exp_params.result_dir, fname)


def _get_exp_logger(exp_params: ExpParams) -> logging.Logger:
    logger_name = f"exp_{exp_params.exp_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    fname = f"exp_{exp_params.exp_id}.log"
    log_file = os.path.join(exp_params.result_dir, fname)
    file_formatter = logging.Formatter(
        '%(asctime)s %(levelname)s %(message)s',
        '%H:%M:%S'
    )

    filehandler = logging.FileHandler(log_file)
    filehandler.setFormatter(file_formatter)
    filehandler.setLevel(logging.DEBUG)

    stream_formatter = logging.Formatter('%(name)s - %(message)s')
    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(stream_formatter)
    streamhandler.setLevel(
        exp_params.r_kwargs.get("log_level", logging.INFO)
    )

    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logger.propagate = False

    return logger


def get_exp_params_statistics(exp_params: ExpParams
                              ) -> stats_lib.AgentStatisticsMap:
    """Get the relevant statistics for given exp params """
    m_params = exp_params.m_params
    pi_params_list = exp_params.pi_params_list
    r_kwargs = exp_params.r_kwargs
    num_agents = len(pi_params_list)

    def _add_dict(stat_dict, param_dict):
        for k, v in param_dict.items():
            if v is None:
                # Need to do this so Pandas imports 'None' instead of NaN
                v = 'None'
            stat_dict[k] = v

    stats = {}
    policy_headers = set()
    for i in range(num_agents):
        stats[i] = {
            "exp_id": exp_params.exp_id,
            "agent_id": i,
            "model": m_params.cls.__name__,
        }
        _add_dict(stats[i], r_kwargs)

        pi_class, pi_kwargs = pi_params_list[i].cls, pi_params_list[i].kwargs
        stats[i]["policy_name"] = pi_class.__name__
        _add_dict(stats[i], pi_kwargs)
        policy_headers.update(pi_kwargs)

    for i in range(num_agents):
        for header in policy_headers:
            if header not in stats[i]:
                stats[i][header] = 'None'

    return stats


def _load_rollout_policies(env_name: str,
                           pi_class: Type[policy_lib.BasePolicy],
                           rpi_kwargs_list: List) -> List[Optional[Dict]]:
    parsed_kwargs_list: List[Optional[Dict]] = []
    for rpi_kwargs in rpi_kwargs_list:
        if rpi_kwargs == 'None' or rpi_kwargs is None:
            parsed_kwargs_list.append(None)
            continue

        assert pi_class == NestedSearchTree
        parsed_agent_rpi_kwargs = {}
        for agent_id, rpi_tuple in rpi_kwargs.items():
            assert 0 < len(rpi_tuple) <= 2
            rpi_name = rpi_tuple[0]
            kwargs = {} if len(rpi_tuple) == 1 else rpi_tuple[1]

            if rpi_name != 'None' and rpi_name is not None:
                rpi_cls = env_lib.get_policy_class(env_name, rpi_name)
            else:
                rpi_cls = policy_lib.RandomPolicy

            parsed_agent_rpi_kwargs[agent_id] = (rpi_cls, kwargs)
        parsed_kwargs_list.append(parsed_agent_rpi_kwargs)
    return parsed_kwargs_list


def _load_agent_policies(env_name: str,
                         default_gamma: float,
                         pi_dict: Dict) -> List[PolicyParams]:
    pi_class = env_lib.get_policy_class(env_name, pi_dict[POLICY_CLASS])

    pi_kwargs_lists: Dict = {}
    for k, v in pi_dict.items():
        if k == POLICY_CLASS:
            continue
        if k == ROLLOUT_POLICIES:
            pi_kwargs_lists[k] = _load_rollout_policies(env_name, pi_class, v)
            continue

        if not isinstance(v, list):
            v = [v]
        for i, x in enumerate(v):
            if x == 'None':
                v[i] = None

        pi_kwargs_lists[k] = v

    # Add default gamma if it's not already in kwargs list
    pi_kwargs_lists[GAMMA] = pi_kwargs_lists.get(GAMMA, [default_gamma])

    pi_params_list = []
    kwarg_keys = list(pi_kwargs_lists)
    for kwarg_values in product(*pi_kwargs_lists.values()):
        pi_kwargs = dict(zip(kwarg_keys, kwarg_values))
        pi_params_list.append(PolicyParams(pi_class, pi_kwargs))

    return pi_params_list


def _load_exp_policies(env_name: str,
                       default_gamma: float,
                       policies_dict: Dict) -> List[List[PolicyParams]]:
    agent_policy_lists = [[] for _ in policies_dict]    # type: ignore
    for i in policies_dict:
        for agent_pi_dict in policies_dict[i]:
            agent_policy_lists[i].extend(
                _load_agent_policies(env_name, default_gamma, agent_pi_dict)
            )
    return agent_policy_lists


def _load_exp_dict(yaml_path: str) -> Dict:
    with open(yaml_path, 'r') as yaml_file:
        exp_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return exp_dict


def _write_exp_dict(exp_name: str, result_dir: str, exp_dict: Dict):
    outfile_path = os.path.join(result_dir, f"{exp_name}.yaml")
    with open(outfile_path, "w") as outfile:
        yaml.dump(exp_dict, stream=outfile, Dumper=yaml.Dumper)


def _get_exp_kwargs(run_kwargs: Dict[str, Any],
                    model_class: Type[M.POSGModel],
                    model_kwargs: Dict[str, Any],
                    policy_params: List[List[PolicyParams]],
                    result_dir: str,
                    exp_id_init: int = 0
                    ) -> Tuple[int, List[ExpParams]]:
    logging.log(EXP_LOG_LEVEL, "Running experiments with following params")
    logging.log(EXP_LOG_LEVEL, pformat(locals()))

    exp_params_list = []
    exp_id = exp_id_init
    # Reverse lists so that larger experiments are started first. This ensures
    # more efficient use of multiple processes
    for agent_pi_params in product(*policy_params):
        m_params = ModelParams(model_class, {**model_kwargs})

        r_kwargs = {**run_kwargs}

        exp_params = ExpParams(
            exp_id, m_params, agent_pi_params, r_kwargs, result_dir
        )
        exp_params_list.append(exp_params)
        exp_id += 1

    return exp_id, exp_params_list


def run_exp_sim(exp_params: ExpParams) -> str:
    """Run simulations for a single experiment and write results to a file """
    exp_logger = _get_exp_logger(exp_params)
    _log_exp_start(exp_params, exp_logger)

    m_params = exp_params.m_params
    pi_params_list = exp_params.pi_params_list
    r_kwargs = exp_params.r_kwargs

    gamma = r_kwargs["gamma"]
    seed = r_kwargs["seed"]

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Disable this by default since rendering asci in log files each step
    # slows down experiments
    r_kwargs['render_asci'] = r_kwargs.get('render_asci', False)

    model = m_params.cls(**m_params.kwargs)

    policies = []
    for i, pi_params in enumerate(pi_params_list):
        pi_class = pi_params.cls
        kwargs = pi_params.kwargs
        policies.append(
            pi_class.initialize(
                model=model, ego_agent=i, logger=exp_logger, **kwargs
            )
        )

    trackers = stats_lib.get_default_trackers(gamma, policies)

    renderers = render_lib.get_renderers(
        r_kwargs.get("show_pi", None),
        r_kwargs.get("show_belief", None),
        r_kwargs.get("show_tree", None)
    )

    try:
        statistics = run_sims(
            model, policies, trackers, renderers, logger=exp_logger, **r_kwargs
        )
        exp_params_statistics = get_exp_params_statistics(exp_params)
        statistics = stats_lib.combine_statistics(
            [statistics, exp_params_statistics]
        )

        results_fname = _get_exp_results_fname(exp_params)
        csv_writer = run_log.CSVWriter(results_fname)
        csv_writer.write(statistics)
        csv_writer.close()

    except Exception as ex:
        exp_logger.exception("Exception occured: %s", str(ex))
        exp_logger.error(pformat(locals()))
        raise ex

    return results_fname


def run_experiment(exp_params_list: List[ExpParams],
                   n_procs: Optional[int] = None) -> str:
    """Run series of simulations """
    logging.info("Running %d experiments", len(exp_params_list))
    result_dir = exp_params_list[0].result_dir

    if n_procs is None:
        n_procs = os.cpu_count()

    mp_lock = mp.Lock()

    if n_procs == 1:
        _init_lock(mp_lock)
        for exp_params in exp_params_list:
            run_exp_sim(exp_params)
    else:
        with mp.Pool(
                n_procs, initializer=_init_lock, initargs=(mp_lock,)
        ) as p:
            p.map(run_exp_sim, exp_params_list, 1)

    return result_dir


def run_yaml_experiment(yaml_path: str,
                        n_procs: Optional[int],
                        test_run: bool = False,
                        extra_output_dir: Optional[str] = None,
                        log_level: int = EXP_LOG_LEVEL):
    """Run experiment defined in YAML file """
    exp_start_time = time.time()
    if n_procs is None:
        n_procs = N_PROCS-1

    log_lib.config_logger(log_level)

    exp_dict = _load_exp_dict(yaml_path)
    logging.log(
        EXP_LOG_LEVEL,
        "Running experiment from file %s with contents\n%s",
        yaml_path,
        pformat(exp_dict)
    )

    env_name = exp_dict[ENV_NAME]
    model_class = env_lib.get_env_class(env_name)
    run_kwargs = exp_dict[RUN_KWARGS]

    for k, v in run_kwargs.items():
        if v == 'None':
            run_kwargs[k] = None

    policy_params = _load_exp_policies(
        env_name, run_kwargs[GAMMA], exp_dict[POLICIES]
    )

    exp_id_init = 0
    exp_params_list = []
    result_dirs = []
    for model_name, model_kwargs in exp_dict[MODELS].items():
        exp_name = f"{env_name}_{model_name}"
        result_dir = run_log.make_dir(exp_name)
        logging.log(
            EXP_LOG_LEVEL, "Saving results to result_dir=%s", result_dir
        )
        result_dirs.append(result_dir)
        _write_exp_dict(exp_name, result_dir, exp_dict)

        exp_id_init, model_exp_params_list = _get_exp_kwargs(
            run_kwargs,
            model_class,
            model_kwargs,
            policy_params,
            result_dir,
            exp_id_init
        )

        exp_params_list.extend(model_exp_params_list)

        logging.log(
            EXP_LOG_LEVEL,
            "\nexp_name=%s - num_exps=%d",
            exp_name,
            len(model_exp_params_list)
        )

    logging.log(
        EXP_LOG_LEVEL,
        "\nTotal number of experiments to run = %d",
        len(exp_params_list)
    )

    if test_run:
        return

    run_experiment(exp_params_list, n_procs=n_procs)

    logging.log(EXP_LOG_LEVEL, "Compiling results")
    for result_dir in result_dirs:
        run_log.compile_results(result_dir, extra_output_dir)

    logging.log(
        EXP_LOG_LEVEL,
        "Experiment Run time %.2f seconds",
        time.time() - exp_start_time
    )
