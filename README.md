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
#### Create a virtual environment
```
# create a virtual environment named env_isaaclab with python3.10
python3.10 -m venv env_isaaclab
# activate the virtual environment
source env_isaaclab/bin/activate
```

#### Install CUDA-enabled PyTorch

For CUDA 12

```
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

#### Upgrade `pip`
```
pip install --upgrade pip
```

#### Install Isaac Sim packages

```
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
```

#### Install IsaacLab

Clone Isaac Lab

```
git clone git@github.com:isaac-sim/IsaacLab.git
```

Install dependencies using `apt`

```
sudo apt install cmake build-essential
```

From the Isaac Lab folder, run the following command

```
./isaaclab.sh --install # or "./isaaclab.sh -i"
```

### Install the TAMPURA package

Activate the IsaacLab venv

```
source env_isaaclab/bin/activate
```

From the root of the repository, run the following commands

```
python -m pip install -e .
pip install pygraphviz
```

# Example Notebook

See `notebooks/grasping_env.ipynb` for a simple usage example.

# Notebooks

The notebooks are available for running in the `notebooks` folder of the repository.

<!-- The robot environments from the paper are in a separate [tampura_environments](https://github.com/aidan-curtis/tampura_environments) repo -->
