I-NTMCP
#######

Implementation of the Interactive Nested Tree Monte-Carlo Planning (I-NTMCP)
algorithm for online planning in I-POMDPs.

This repository contains all the code used for the "Online planning for
Interactive POMDPs using Nested Tree Monte Carlo planning" paper. While we have
done our best to make it as user friendly as possible it should be treated as a
prototype only and for verifying results. We hope to release a more robust
implementation in the future.


Installation
~~~~~~~~~~~~

I-NTMCP is implemented using Python 3.9.

As with any python seperate python library, project, etc, we recommend
installing I-NTMCP in a seperate virtual environment such as those provided by `Conda <https://docs.conda.io/>`_.


1. Download or clone the repo.
2. Install I-NTMCP using PIP by navigating to the ``i-ntmcp`` root directory (the one containing the ``setup.py`` file), and running:


.. code-block:: bash
    pip install -e .

    # use the following to install all dependencies (including for search tree
    visualization, pygraphviz)
    pip install -e .[all]


Running I-NTMCP
~~~~~~~~~~~~~~~

I-NTMCP can be run using the run script provided: ``run_scripts/run.py``. This script takes the following compolsory as arguments:


1. the environment name - ``rc`` for the *Runner-Chaser* domain, and ``pe`` for the *Pursuit-Evasion* domain.
2. the name of the policy/algorithm to use for agent 0
3. the name of the policy/algorithm to use for agent 1


The available policies are:


1. ``NST``, ``NestedSearchTree`` - this is the I-NTMCP algorithm
2. ``random`` - a uniform random policy
3. ``PESPPolicy`` - the shortest path policy for the *Pursuit-Evasion* problem
4. ``RCNestedReasoningPolicy`` - the finite nested reasoning policy for the *Runner-Chaser* problem


The script also takes additional arguments depending on the problem and policy/algorithm being run. These additional arguments can be viewed using the ``--help`` flag.

For example, the different size grids for the *Runner-Chaser* problem can be run using the ``--grid_name`` flag with and either ``cap3``, ``cap4``, or ``cap7``.


IROS paper results and experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

IROS paper results and other files can be found in the ``iros_paper`` directory. For more details for each environment, including instructions for re-running the experiments, see the README files in the ``iros_paper/rc`` and ``iros_paper/pe`` directories.


Authors
~~~~~~~

**Jonathon Schwartz** - Jonathon.schwartz@anu.edu.au (primary author)
**Ruijia Zhou** - Ruijia.zhou@anu.edu.au
**Hanna Kurniawati** - Hanna.kurniawati@anu.edu.au


Licence
~~~~~~~

**MIT** © 2022, Jonathon Schwartz
