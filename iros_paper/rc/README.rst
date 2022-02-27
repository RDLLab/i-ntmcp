IROS 22 Paper: *Runner-Chaser*
################################

This directory contains all experiment and results files for the *Runner-Chaser* environment. The files are organized based on the experiment type being run. Specifically:

- ``i-pomdp-lite_comparison`` directory contains files for the experiments used to compare against I-POMDP Lite
- ``num_sims`` directory contains files for the experiments looking at the performance of I-NTMCP as the number of simulations and nesting levels increased

Within each directory:

- ``compiled_results.csv`` files contain raw experiment results. The experiment the results are attached to can be determined by the name of the parent directory.
- ``run.txt`` files contain high-level stdout output from experiment runs.
- ``X.yaml`` files contain experiment parameter definitions (these are the files passed as input to the experiment run script)
- ``rc_uct_c_exps_evader.txt`` and ``rc_uct_c_exps_pursuer.txt`` contain details and results of experiments for finding the UCT C hyperparameter values used. Note, that the same value for the *7x7* grid is used for both types of experiments.

Each directory also contains jupyter-notebooks for result analysis:

- ``plot_rc_pairwise_by_x.ipynb`` for generating plots from the raw results file for the num sims experiment.
- ``rc_comparison.ipynb`` for generating tables from raw results for the I-POMDP Lite comparison experiments.



Running the experiments
~~~~~~~~~~~~~~~~~~~~~~~

**Warning** the experiments can take a while to run (multiple days for some experiments). You can run shorter experiments by adjusting the parameters in the .yaml experiment files (or copy the file and make your own).

The experiments can be run using the ``run_exp.py`` script located in the ``i-ntmcp/run_scripts`` directory.

.. code-block:: bash

   # This commands are run from the i-ntmcp directory
   # you can view run options using the --help flag
   python run_scripts/run_exp.py --help

   # to run num sims experiment
   python run_scripts/run_exp.py iros_paper/rc/num_sims/rc_7x7_exp_numsims.yaml

   # to run the different I-POMDP Lite comparison experiments
   python run_scripts/run_exp.py iros_paper/rc/i-pomdp-lite_comparison/3x3/rc_3x3_comparison.yaml
   python run_scripts/run_exp.py iros_paper/rc/i-pomdp-lite_comparison/4x4/rc_4x4_comparison.yaml
   python run_scripts/run_exp.py iros_paper/rc/i-pomdp-lite_comparison/7x7/rc_7x7_comparison.yaml


The results of each experiment run will be saved to a directory (with the same name as the .yaml file) in the ``i-ntmcp/results/`` directory.
