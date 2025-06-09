# TAMPURA for decentralized multi-agent settings

Partially Observable Task and motion planning with uncertainty and risk awareness for decentralized multi-agent settings. Uses the framework introduced in [paper](https://arxiv.org/abs/2403.10454) or [website](https://aidan-curtis.github.io/tampura.github.io/).

<!-- ![alt text](figs/tasks.png) -->

## Install

Clone this repository
```
git clone git@github.com:akansha2001/tamp_multi_agent.git
```
In the root of the repository, install IsaacLab. Instructions for the `pip` installation using `venv`:

### IsaacLab installation

Follow the instructions for installing IsaacLab from [website](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). Use the binaries installation for IsaacSim and set up a conda environment.


Activate the IsaacLab environment. From the root of the repository, run the following commands

```
python -m pip install -e .
pip install pygraphviz
```

# Example Notebook

See `notebooks/grasping_env.ipynb` for a simple usage example.

# Notebooks

The notebooks are available for running in the `notebooks` folder of the repository.

<!-- The robot environments from the paper are in a separate [tampura_environments](https://github.com/aidan-curtis/tampura_environments) repo -->
