"""Common config information for the INTMCP Package """
import os
import os.path as osp


BASE_DIR = osp.dirname(osp.abspath(__file__))
BASE_REPO_DIR = osp.abspath(osp.join(BASE_DIR, os.pardir))
BASE_RESULTS_DIR = osp.join(BASE_REPO_DIR, "results")


if not osp.exists(BASE_RESULTS_DIR):
    os.makedirs(BASE_RESULTS_DIR)


os_cpu_count = os.cpu_count()
if os_cpu_count is not None:
    N_PROCS = max(1, os_cpu_count)
else:
    N_PROCS = 1
