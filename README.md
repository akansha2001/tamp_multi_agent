# TAMPURA for decentralized multi-agent settings

Partially Observable Task and motion planning with uncertainty and risk awareness for decentralized multi-agent settings. Uses the framework introduced in [paper](https://arxiv.org/abs/2403.10454) or [website](https://aidan-curtis.github.io/tampura.github.io/).

<!-- ![alt text](figs/tasks.png) -->

## Install

Clone this repository
```
git clone git@github.com:akansha2001/tamp_multi_agent.git
```
In the root of the repository, install IsaacLab. 

### IsaacLab installation

Follow the instructions for installing IsaacLab from [website](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). Use the `pip` installation for IsaacSim and set up a `conda` environment.


Activate the IsaacLab environment. From the root of the repository, run the following commands

```
python -m pip install -e .
pip install pygraphviz
```

# Example Notebook

See `notebooks/grasping_env.ipynb` for a simple usage example.

# Notebooks

The notebooks are available for running in the `notebooks` folder of the repository.

# IsaacLab execution

For sample executions, move the `tamp_uncertainty` folder to `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/` and enter

```
python IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/tamp_uncertainty/search_object.py --num_envs 1
```

from the root of the repository. More information about these scenarios is available on [this repository](https://github.com/akansha2001/tamp_uncertainty.git).

<!-- The robot environments from the paper are in a separate [tampura_environments](https://github.com/aidan-curtis/tampura_environments) repo -->
