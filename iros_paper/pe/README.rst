IROS 22 Paper: *Pursuit-Evasion*
################################

This directory contains all experiment and results files for the *Pursuit-Evasion* environment. Specifically:

- ``compiled_results.csv`` contains the raw experiment results used in the paper
- ``pe_uct_c_exps_evader.txt`` and ``pe_uct_c_exps_pursuer.txt`` contain details and results of experiments for finding the UCT C hyperparameter value used.
- ``pe_exp_num_sims.yaml`` and ``pe_exp_win_rate.yaml`` define the parameters for the experiments run
- ``pe_winrate_analysis.ipynb`` is an jupyter-notebook file for extracting winrate results from the raw results file.
- ``plot_pe.ipynb`` is an jupyter-notebook file for generating plots from the raw results file.
- ``win_rate_numsims=2048.csv`` contains the specific results used for the win rate table in the paper (this file was generated from ``compiled_results.csv`` using the ``pe_winrate_analysis.ipynb`` notebook).


Running the experiments
~~~~~~~~~~~~~~~~~~~~~~~

**Warning** the experiments can take a while to run (multiple days for the ``pe_exp_num_sims.yaml`` experiment). You can run shorter experiments by adjusting the parameters in the .yaml experiment files (or copy the file and make your own).

The experiments can be run using the ``run_exp.py`` script located in the ``i-ntmcp/run_scripts`` directory.

.. code-block:: bash

   # you can view run options using the --help flag
   python run_scripts/run_exp.py --help

   # to run num sims experiment from i-ntmcp directory
   python run_scripts/run_exp.py iros_paper/pe/pe_exp_num_sims.yaml

   # to run win rate experiment from i-ntmcp directory
   python run_scripts/run_exp.py iros_paper/pe/pe_exp_win_rate.yaml


The results of each experiment run will be saved to a directory in the ``i-ntmcp/results/`` directory.


Problem Size
~~~~~~~~~~~~

For the 8x8 grid

Number of states = (# open locs * # directions)^(num agents) * (max # possible different runner goal locs)
*|S|* = 3*(43*4)**2 = 88752 = ~8.8*10^4

Number of observations = #WallObs * seen/not * heard/not
*|O|* chaser = *|O|* runner = 2^6 = 64

*|A|* runner = *|A|* chaser = 4

Observation branching factor = 4
