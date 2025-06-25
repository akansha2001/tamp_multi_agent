# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import class_uncertainty_ik_abs_env_cfg, search_object_ik_abs_env_cfg, simple_pick_env_cfg, cabinet_pick_env_cfg, multi_clean_env_cfg



##
# Inverse Kinematics - Absolute Pose Control
##



gym.register(
    id="Isaac-Lift-Franka-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": class_uncertainty_ik_abs_env_cfg.FrankaLiftEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Franka-IK-Abs-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": search_object_ik_abs_env_cfg.FrankaLiftEnvCfg,
    },
    disable_env_checker=True,
)


# simple pick

gym.register(
    id="Isaac-Lift-Cube-Franka-IK-Abs-simple-pick",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": simple_pick_env_cfg.FrankaSimplePickEnvCfgik,
    },
    disable_env_checker=True,
)

# cabinet pick

gym.register(
    id="Isaac-Lift-Cube-Franka-IK-Abs-cabinet-pick",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": cabinet_pick_env_cfg.FrankaCabinetPickEnvCfgik,
    },
    disable_env_checker=True,
)

# multi agent clean


gym.register(
    id="Isaac-Lift-Cube-Franka-IK-Abs-multi-clean",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": multi_clean_env_cfg.FrankaMultiCleanEnvCfgik,
    },
    disable_env_checker=True,
)

