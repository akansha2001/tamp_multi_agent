# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an environment with a pick and lift state machine.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

    ./isaaclab.sh -p scripts/environments/state_machine/lift_cube_sm.py --num_envs 32

"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick, clean, place state machines.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# add arguments to record videos
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=5000, help="Interval between video recordings (in steps).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch
from collections.abc import Sequence

import warp as wp

from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData

import isaaclab_tasks  # noqa: F401
from multi_clean_env_cfg import FrankaMultiCleanEnvCfgik
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# initialize warp
wp.init()

TEMP_POS = [0.7, 0, 0.055]
GOAL_POS = [0.5, 0, 0.055]

class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)

class PickPlaceCleanSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)
    APPROACH_ABOVE_GOAL = wp.constant(5)
    APPROACH_GOAL = wp.constant(6)
    RELEASE_OBJECT = wp.constant(7)
    LIFT_EE = wp.constant(8)
    APPROACH_LOC_1 = wp.constant(9)
    APPROACH_LOC_2 = wp.constant(10)
    
class PickPlaceCleanSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.6)
    GRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(1.0)
    APPROACH_ABOVE_GOAL = wp.constant(0.5)
    APPROACH_GOAL = wp.constant(0.6)
    RELEASE_OBJECT = wp.constant(0.3)
    LIFT_EE = wp.constant(1.0)
    APPROACH_LOC_1 = wp.constant(0.4)
    APPROACH_LOC_2 = wp.constant(0.4)
    
    TIMEOUT = wp.constant(10.0)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold

@wp.kernel
def infer_pick_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickPlaceCleanSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickPlaceCleanSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PickPlaceCleanSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickPlaceCleanSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickPlaceCleanSmWaitTime.APPROACH_ABOVE_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickPlaceCleanSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickPlaceCleanSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickPlaceCleanSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickPlaceCleanSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickPlaceCleanSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= PickPlaceCleanSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickPlaceCleanSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickPlaceCleanSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickPlaceCleanSmWaitTime.LIFT_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickPlaceCleanSmState.LIFT_OBJECT
                sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]

@wp.kernel
def infer_place_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickPlaceCleanSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= PickPlaceCleanSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PickPlaceCleanSmState.APPROACH_ABOVE_GOAL
            sm_wait_time[tid] = 0.0
    elif state == PickPlaceCleanSmState.APPROACH_ABOVE_GOAL:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickPlaceCleanSmWaitTime.APPROACH_ABOVE_GOAL:
                # move to next state and reset wait time
                sm_state[tid] = PickPlaceCleanSmState.APPROACH_GOAL
                sm_wait_time[tid] = 0.0
    elif state == PickPlaceCleanSmState.APPROACH_GOAL:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickPlaceCleanSmWaitTime.APPROACH_GOAL:
                # move to next state and reset wait time
                sm_state[tid] = PickPlaceCleanSmState.RELEASE_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickPlaceCleanSmState.RELEASE_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickPlaceCleanSmWaitTime.RELEASE_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickPlaceCleanSmState.LIFT_EE
            sm_wait_time[tid] = 0.0
    elif state == PickPlaceCleanSmState.LIFT_EE:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickPlaceCleanSmWaitTime.LIFT_EE:
                # move to next state and reset wait time
                sm_state[tid] = PickPlaceCleanSmState.LIFT_EE # new
                sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]

class PickPlaceCleanSm:
    """A simple state machine in a robot's task space to pick and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.01):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute_pick(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor) -> torch.Tensor:
        """Compute the desired state of the robot's end-effector and the gripper."""
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_pick_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.position_threshold,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)
    def compute_place(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor) -> torch.Tensor:
        """Compute the desired state of the robot's end-effector and the gripper."""
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_place_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.position_threshold,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)
    def compute_clean(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor) -> torch.Tensor:
        """Compute the desired state of the robot's end-effector and the gripper."""
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 1] = 0.2 # y axis
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_place_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.position_threshold,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


# TAMPURA
import random
from dataclasses import dataclass, field
from typing import List,Dict,Any

import copy
import itertools 
import time
from tampura.policies.policy import save_config, RolloutHistory, save_run_data

import numpy as np
import os
from tampura.environment import TampuraEnv
from tampura.spec import ProblemSpec
from tampura.structs import (
    AbstractBelief,
    ActionSchema,
    StreamSchema,
    AliasStore,
    Belief,
    NoOp,
    Predicate,
    State,
    effect_from_execute_fn,
    Observation,
    AbstractBeliefSet,
)
import logging 
from tampura.symbolic import OBJ, Atom, ForAll, Not, Exists, Or, And, OneOf, Eq, eval_expr
from tampura.policies.tampura_policy import TampuraPolicy
from tampura.config.config import load_config, setup_logger

# TODO: different training and execution scenarios, study the MDPs
# 0: human, 1: random, 2: inactive
TRAIN = 1
# 0: human, 1: random, 2: inactive, 3: nominal
EXEC = 0

from pick_successes import SIMPLE_PICK_EGO_SIM, CABINET_PICK_EGO_SIM

ROB = "robot_"
REG = "region_"
MUG = "mug"
REGIONS = [f"{REG}{MUG}",f"{REG}stable_mug"]
ACTION_NAMES = ["pick_action","place_action","clean_action","nothing_action"]

# problem specification: try with just one robot to demonstrate how overall cost increases
ROBOTS=[f"{ROB}1",f"{ROB}2"]
OBJ_REGIONS={MUG:REGIONS[0]}

# higher num_samples needed to learn true transition model

# Test 
# GOAL = Atom("holding",[ROBOTS[0],MUG])
GOAL = And([Atom("clean",[REGIONS[0]]),Atom("in_obj",[MUG,REGIONS[0]])])
# GOAL = Atom("holding",[ROBOTS[0],MUG]) # test to see if place other pick ego works

# State of the environment
@dataclass
class EnvState(State):
    holding: Dict[str,List[str]] = field(default_factory=lambda: {})
    obj_regions: Dict[str,str] = field(default_factory=lambda:{})
    clean: List[str] = field(default_factory=lambda: [])
    next_actions: List[str] = field(default_factory=lambda: [])
    sim_env: Any = field(default_factory=lambda: None)
    sim_env_cfg: Any = field(default_factory=lambda: None)
    robot_1_offset: Any = field(default_factory=lambda: None)
    robot_2_offset: Any = field(default_factory=lambda: None)
    ppc_sm: Any = field(default_factory=lambda: None)
            
        
# Observation space
@dataclass
class EnvObservation(Observation):
    holding: Dict[str,List[str]] = field(default_factory=lambda: {})
    obj_regions: Dict[str,str] = field(default_factory=lambda:{})
    clean: List[str] = field(default_factory=lambda: [])
    next_actions: List[str] = field(default_factory=lambda: [])

# Belief space
class EnvBelief(Belief):
    def __init__(self, holding={},obj_regions={},clean=[],next_actions=[]):
        # true state
        self.holding = holding.copy()
        self.obj_regions = obj_regions.copy()
        self.clean = clean.copy()
        self.next_actions = next_actions.copy()
        

    def update(self, a, o, s):
        
        # dictionary mutations are IN-PLACE!!! use .copy()!!
        holding = self.holding.copy() 
        obj_regions = self.obj_regions.copy()
        clean = self.clean.copy()
        next_actions = self.next_actions.copy()
        
        
        # get argument index for ego agent
        
        a_other_name,a_ego_name = a.name.split("*")
        
        if a_other_name == "clean_other": # robot, region
            nargs_other = 2
        elif a_other_name == "nothing_other":
            nargs_other = 1
        else: # robot, object, region
            nargs_other = 3
            
        a_ego_args = a.args[nargs_other:]
       
        
        
        if a_other_name == "pick_other" or a_other_name == "place_other":
    
            holding[a.args[0]] = o.holding[a.args[0]]
            obj_regions[a.args[1]] = o.obj_regions[a.args[1]]
        
        elif a_other_name == "clean_other":
            
            clean = o.clean
        
        if a_ego_name == "pick_ego" or a_ego_name == "place_ego":
            
            holding[a_ego_args[0]] = o.holding[a_ego_args[0]]
            obj_regions[a_ego_args[1]] = o.obj_regions[a_ego_args[1]]
        
        elif a_ego_name == "clean_ego":
            
            clean = o.clean
              
        next_actions = o.next_actions
            
        return EnvBelief(holding=holding,clean=clean,obj_regions=obj_regions,next_actions=next_actions)

    def abstract(self, store: AliasStore):
        
        ab = []
        
        # true state
        for rob in self.holding.keys():
            ab += [Atom("holding",[rob,obj]) for obj in self.holding[rob]]
        for obj in self.obj_regions.keys():
            if self.obj_regions[obj] !="":
                ab += [Atom("in_obj",[obj,self.obj_regions[obj]])]
        for clean_region in self.clean:
            ab += [Atom("clean",[clean_region])]
        # next actions
        if self.next_actions != []:
            for next_action in self.next_actions:
                
                name,args = next_action.split("-")
                args=list(args.split("%"))
                
                rob=args[0]
                if Atom("is_ego",[rob]) not in store.certified: # not the ego agent
                    ab += [Atom(name,args)]
            
        return AbstractBelief(ab)

    # def vectorize(self):
    #     return np.array([int(obj in self.holding) for obj in OBJECTS])
      

def get_next_actions_execute(a, b, store): 
    
    a_other_name,a_ego_name = a.name.split("*")
    
    if a_other_name == "clean_other": # robot, region
        n_args = 2
    elif a_other_name == "nothing_other":
        n_args = 1
    else: # robot, object, region
        n_args = 3
    
    if EXEC == 0: # human  
        print("ego attempts action ..")
        print(a_ego_name)
        print(a.args[n_args:])
        print("from")
        print(b.abstract(store).items)
    
    
    next_actions=[]
    others = []
    for entity in store.als_type:
        if store.als_type[entity]=="robot":
            if Atom("is_ego",[entity]) not in store.certified:
                others.append(entity)
    
    # depending on whether or not ego succeeded, different actions are applicable
    # however other agent does not know whether ego succeeded, so all actions should technically be applicable!
    # pruned out by preconditions! 
    for rob in others: # one list of outcomes per robot
        
        applicable_actions_rob=[]
        # nothing is always applicable
        applicable_actions_rob.append(Atom("nothing_action",[rob]))
        
        for reg in REGIONS:
            
            applicable_actions_rob.append(Atom("clean_action",[rob,reg]))
            
            for obj in OBJ_REGIONS.keys():
                applicable_actions_rob.append(Atom("pick_action",[rob,obj,reg]))
                applicable_actions_rob.append(Atom("place_action",[rob,obj,reg]))
        
        if EXEC == 0: # human        
        
            while True:
                
                for i,act in enumerate(applicable_actions_rob):
                    print(str(i)+". "+act.pred_name+str(act.args))
                    
                choice = input("choose an action \n")
                if int(choice)>=0 and int(choice)<len(applicable_actions_rob):
                    break
                else:
                    print("invalid choice, enter again")
            
            observed_action_rob = applicable_actions_rob[int(choice)] 
            
        elif EXEC == 1: # random 
            
            observed_action_rob = random.choice(applicable_actions_rob)
            
        elif EXEC == 2: #inactive
            
            observed_action_rob = Atom("nothing_action",[rob])
            
          
        print(observed_action_rob)
        
        name=observed_action_rob.pred_name
        args=observed_action_rob.args
        
        if name == "nothing_action":
            a_other=name+"-"+rob
        elif name == "clean_action": 
            a_other=name+"-"+rob+"%"+args[1]
        else: # pick and place
            a_other=name+"-"+rob+"%"+args[1]+"%"+args[2]
            
        next_actions.append(a_other)
    
    return next_actions # for all the agents
def get_next_actions_effects(a, b, store): # human operator : tedious, kind of works
    
    
    a_other_name,a_ego_name = a.name.split("*")
    
    if a_other_name == "clean_other": # robot, region
        n_args = 2
    elif a_other_name == "nothing_other":
        n_args = 1
    else: # robot, object, region
        n_args = 3
    
    if TRAIN == 0: # human  
        print("ego attempts action ..")
        print(a_ego_name)
        print(a.args[n_args:])
        print("from")
        print(b.abstract(store).items)
        
    
    next_actions=[]
    others = []
    for entity in store.als_type:
        if store.als_type[entity]=="robot":
            if Atom("is_ego",[entity]) not in store.certified:
                others.append(entity)
    
    for rob in others: # one list of outcomes per robot
        
        applicable_actions_rob=[]
        # nothing is always applicable
        applicable_actions_rob.append(Atom("nothing_action",[rob]))
        
        for reg in REGIONS:
            
            applicable_actions_rob.append(Atom("clean_action",[rob,reg]))
            
            for obj in OBJ_REGIONS.keys():
                applicable_actions_rob.append(Atom("pick_action",[rob,obj,reg]))
                applicable_actions_rob.append(Atom("place_action",[rob,obj,reg]))

        if TRAIN == 0: # human 
            while True:
                
                for i,act in enumerate(applicable_actions_rob):
                    print(str(i)+". "+act.pred_name+str(act.args))
                    
                choice = input("choose an action \n")
                if int(choice)>=0 and int(choice)<len(applicable_actions_rob):
                    break
                else:
                    print("invalid choice, enter again")
            
            observed_action_rob = applicable_actions_rob[int(choice)] 
            print(observed_action_rob)
        elif TRAIN == 1: # random 
            observed_action_rob = random.choice(applicable_actions_rob)
        elif TRAIN == 2: # inactive agent
            observed_action_rob = Atom("nothing_action",[rob])
        
        name=observed_action_rob.pred_name
        args=observed_action_rob.args
        
        if name == "nothing_action":
            a_other=name+"-"+rob
        elif name == "clean_action": 
            a_other=name+"-"+rob+"%"+args[1]
        else: # pick and place
            a_other=name+"-"+rob+"%"+args[1]+"%"+args[2]
            
        next_actions.append(a_other)
    
    
    return next_actions # for all the agents


# execution
def pick_execute(s, args):
    
    # simulation
    sim_env = s.sim_env 
    sim_env_cfg = s.sim_env_cfg
    robot_1_offset = s.robot_1_offset
    robot_2_offset = s.robot_2_offset
    ppc_sm = s.ppc_sm
        
    
    # robot_1
    # -- end-effector frame
    ee_frame_sensor_1 = sim_env.unwrapped.scene["ee_frame_1"]
    tcp_rest_position_1 = ee_frame_sensor_1.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_1_offset
    tcp_rest_orientation_1 = ee_frame_sensor_1.data.target_quat_w[..., 0, :].clone()
    
    # robot_2
    # -- end-effector frame
    ee_frame_sensor_2 = sim_env.unwrapped.scene["ee_frame_2"]
    tcp_rest_position_2 = ee_frame_sensor_2.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_2_offset
    tcp_rest_orientation_2 = ee_frame_sensor_2.data.target_quat_w[..., 0, :].clone()
    
    
    # create action buffers (position + quaternion)
    actions = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
    
    actions[:,0:7] = torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1)
    actions[:,7] = GripperState.OPEN
    
    actions[:,8:15] = torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1)
    actions[:,15] = GripperState.OPEN
    
    
    
    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros((sim_env.unwrapped.num_envs, 4), device=sim_env.unwrapped.device)
    desired_orientation[:, 1] = 1.0
    # create state machine
    ppc_sm = PickPlaceCleanSm(
        sim_env_cfg.sim.dt * sim_env_cfg.decimation, sim_env.unwrapped.num_envs, sim_env.unwrapped.device, position_threshold=0.01
    )
    
    
    
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            dones = sim_env.step(actions)[-2]

            # observations
    
            # robot_1
            # -- end-effector frame
            ee_frame_sensor_1 = sim_env.unwrapped.scene["ee_frame_1"]
            tcp_rest_position_1 = ee_frame_sensor_1.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_1_offset
            tcp_rest_orientation_1 = ee_frame_sensor_1.data.target_quat_w[..., 0, :].clone()
            
            # robot_2
            # -- end-effector frame
            ee_frame_sensor_2 = sim_env.unwrapped.scene["ee_frame_2"]
            tcp_rest_position_2 = ee_frame_sensor_2.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins  - robot_2_offset
            tcp_rest_orientation_2 = ee_frame_sensor_2.data.target_quat_w[..., 0, :].clone()
            
            
            # -- object frame
            object_data: RigidObjectData = sim_env.unwrapped.scene["object"].data
            object_position_1 = object_data.root_pos_w - sim_env.unwrapped.scene.env_origins - robot_1_offset
            object_position_2 = object_data.root_pos_w - sim_env.unwrapped.scene.env_origins - robot_2_offset

            
            # -- target object frame
            desired_position_1 = torch.tensor([[0.2,0.0,0.35]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)
            desired_position_2 = torch.tensor([[0.2,0.0,0.35]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)
            

            if args[0] == ROBOTS[0]: # other agent is robot_1  

                actions = ppc_sm.compute_pick(
                    torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1),
                    torch.cat([object_position_1, desired_orientation], dim=-1),
                    torch.cat([desired_position_1, desired_orientation], dim=-1),
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,8:15] = torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1)
                actions_buffer[:,15] = GripperState.CLOSE
                actions_buffer[:,:8] = actions
                
                
                actions = actions_buffer
                
            else: # other agent is robot_2

                actions = ppc_sm.compute_pick(
                    torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1),
                    torch.cat([object_position_2, desired_orientation], dim=-1),
                    torch.cat([desired_position_2, desired_orientation], dim=-1),
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,:7] = torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1)
                actions_buffer[:,7] = GripperState.CLOSE
                actions_buffer[:,8:] = actions
                
                actions = actions_buffer
            
            
                    
            if ppc_sm.sm_state == PickPlaceCleanSmState.LIFT_OBJECT and ppc_sm.sm_wait_time >= PickPlaceCleanSmWaitTime.LIFT_OBJECT:
                ppc_sm.sm_state = PickPlaceCleanSmState.REST
                s.holding[args[0]] = [args[1]]
                s.obj_regions[args[1]] = ""
                
                break
            else: 
                if ppc_sm.sm_wait_time > PickPlaceCleanSmWaitTime.TIMEOUT:
                    
                    break
                
    s.ppc_sm = ppc_sm
    return s
def place_execute(s, args):
    
    # simulation
    sim_env = s.sim_env 
    sim_env_cfg = s.sim_env_cfg
    robot_1_offset = s.robot_1_offset
    robot_2_offset = s.robot_2_offset
    ppc_sm = s.ppc_sm
    
    if args[2] == REGIONS[0]:
        goal = torch.tensor([GOAL_POS]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device) - sim_env.unwrapped.scene.env_origins 
    else:
        goal = torch.tensor([TEMP_POS]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device) - sim_env.unwrapped.scene.env_origins 
    
    if args[0] == ROBOTS[0]:
        goal = goal - robot_1_offset
    else:
        goal = goal - robot_2_offset   
    
    # robot_1
    # -- end-effector frame
    ee_frame_sensor_1 = sim_env.unwrapped.scene["ee_frame_1"]
    tcp_rest_position_1 = ee_frame_sensor_1.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_1_offset
    tcp_rest_orientation_1 = ee_frame_sensor_1.data.target_quat_w[..., 0, :].clone()
    
    # robot_2
    # -- end-effector frame
    ee_frame_sensor_2 = sim_env.unwrapped.scene["ee_frame_2"]
    tcp_rest_position_2 = ee_frame_sensor_2.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_2_offset
    tcp_rest_orientation_2 = ee_frame_sensor_2.data.target_quat_w[..., 0, :].clone()
    
    
    # create action buffers (position + quaternion)
    actions = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
    
    actions[:,0:7] = torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1)
    actions[:,7] = GripperState.CLOSE
    
    actions[:,8:15] = torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1)
    actions[:,15] = GripperState.CLOSE
    
    
    
    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros((sim_env.unwrapped.num_envs, 4), device=sim_env.unwrapped.device)
    desired_orientation[:, 1] = 1.0
    # create state machine
    ppc_sm = PickPlaceCleanSm(
        sim_env_cfg.sim.dt * sim_env_cfg.decimation, sim_env.unwrapped.num_envs, sim_env.unwrapped.device, position_threshold=0.01
    )
    
    
    
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            dones = sim_env.step(actions)[-2]

            # observations
    
            # robot_1
            # -- end-effector frame
            ee_frame_sensor_1 = sim_env.unwrapped.scene["ee_frame_1"]
            tcp_rest_position_1 = ee_frame_sensor_1.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_1_offset
            tcp_rest_orientation_1 = ee_frame_sensor_1.data.target_quat_w[..., 0, :].clone()
            
            # robot_2
            # -- end-effector frame
            ee_frame_sensor_2 = sim_env.unwrapped.scene["ee_frame_2"]
            tcp_rest_position_2 = ee_frame_sensor_2.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins  - robot_2_offset
            tcp_rest_orientation_2 = ee_frame_sensor_2.data.target_quat_w[..., 0, :].clone()
            
            
            # -- object frame
            object_data: RigidObjectData = sim_env.unwrapped.scene["object"].data
            object_position_1 = object_data.root_pos_w - sim_env.unwrapped.scene.env_origins - robot_1_offset
            object_position_2 = object_data.root_pos_w - sim_env.unwrapped.scene.env_origins - robot_2_offset

            
            # -- target object frame
            desired_position_1 = torch.tensor([[0.2,0.0,0.35]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)
            desired_position_2 = torch.tensor([[0.2,0.0,0.35]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)
            

            if args[0] == ROBOTS[0]: # other agent is robot_1  

                actions = ppc_sm.compute_place(
                    torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1),
                    torch.cat([goal, desired_orientation], dim=-1),
                    torch.cat([desired_position_1, desired_orientation], dim=-1),
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,8:15] = torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1)
                actions_buffer[:,15] = GripperState.CLOSE
                actions_buffer[:,:8] = actions
                
                
                actions = actions_buffer
                
            else: # other agent is robot_2

                actions = ppc_sm.compute_place(
                    torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1),
                    torch.cat([goal, desired_orientation], dim=-1),
                    torch.cat([desired_position_2, desired_orientation], dim=-1),
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,:7] = torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1)
                actions_buffer[:,7] = GripperState.CLOSE
                actions_buffer[:,8:] = actions
                
                actions = actions_buffer
            
            # print statements => stable behavior??
            print(ppc_sm.sm_state)
            print(ppc_sm.sm_wait_time)       
            if ppc_sm.sm_state == PickPlaceCleanSmState.LIFT_EE and ppc_sm.sm_wait_time >= PickPlaceCleanSmWaitTime.LIFT_EE:
                ppc_sm.sm_state = PickPlaceCleanSmState.REST
                s.holding[args[0]] = []
                s.obj_regions[args[1]] = args[2]
                
                break
            else: 
                if ppc_sm.sm_wait_time > PickPlaceCleanSmWaitTime.TIMEOUT:
                    
                    break
                
    s.ppc_sm = ppc_sm
    return s
def clean_execute(s, args):
    
    # simulation
    sim_env = s.sim_env 
    sim_env_cfg = s.sim_env_cfg
    robot_1_offset = s.robot_1_offset
    robot_2_offset = s.robot_2_offset
    ppc_sm = s.ppc_sm
    
    
    
    if args[1] == REGIONS[0]:
        goal_pos = GOAL_POS.copy()
    else:
        goal_pos = TEMP_POS.copy()
        
    goal_pos[1] -= 0.1
    goal_pos[2] = 0.0 
    
    
    goal = torch.tensor([goal_pos]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device) - sim_env.unwrapped.scene.env_origins 
    if args[0] == ROBOTS[0]:
        goal = goal - robot_1_offset
    else:
        goal = goal - robot_2_offset     
    
    # robot_1
    # -- end-effector frame
    ee_frame_sensor_1 = sim_env.unwrapped.scene["ee_frame_1"]
    tcp_rest_position_1 = ee_frame_sensor_1.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_1_offset
    tcp_rest_orientation_1 = ee_frame_sensor_1.data.target_quat_w[..., 0, :].clone()
    
    # robot_2
    # -- end-effector frame
    ee_frame_sensor_2 = sim_env.unwrapped.scene["ee_frame_2"]
    tcp_rest_position_2 = ee_frame_sensor_2.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_2_offset
    tcp_rest_orientation_2 = ee_frame_sensor_2.data.target_quat_w[..., 0, :].clone()
    
    
    # create action buffers (position + quaternion)
    actions = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
    
    actions[:,0:7] = torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1)
    actions[:,7] = GripperState.CLOSE
    
    actions[:,8:15] = torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1)
    actions[:,15] = GripperState.CLOSE
    
    
    
    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros((sim_env.unwrapped.num_envs, 4), device=sim_env.unwrapped.device)
    desired_orientation[:, 1] = 1.0
    # create state machine
    ppc_sm = PickPlaceCleanSm(
        sim_env_cfg.sim.dt * sim_env_cfg.decimation, sim_env.unwrapped.num_envs, sim_env.unwrapped.device, position_threshold=0.01
    )
    
    
    
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            dones = sim_env.step(actions)[-2]

            # observations
    
            # robot_1
            # -- end-effector frame
            ee_frame_sensor_1 = sim_env.unwrapped.scene["ee_frame_1"]
            tcp_rest_position_1 = ee_frame_sensor_1.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_1_offset
            tcp_rest_orientation_1 = ee_frame_sensor_1.data.target_quat_w[..., 0, :].clone()
            
            # robot_2
            # -- end-effector frame
            ee_frame_sensor_2 = sim_env.unwrapped.scene["ee_frame_2"]
            tcp_rest_position_2 = ee_frame_sensor_2.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins  - robot_2_offset
            tcp_rest_orientation_2 = ee_frame_sensor_2.data.target_quat_w[..., 0, :].clone()
            
            
            # -- object frame
            object_data: RigidObjectData = sim_env.unwrapped.scene["object"].data
            object_position_1 = object_data.root_pos_w - sim_env.unwrapped.scene.env_origins - robot_1_offset
            object_position_2 = object_data.root_pos_w - sim_env.unwrapped.scene.env_origins - robot_2_offset

            
            # -- target object frame
            desired_position_1 = torch.tensor([[0.2,0.0,0.35]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)
            desired_position_2 = torch.tensor([[0.2,0.0,0.35]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)
            

            if args[0] == ROBOTS[0]: # other agent is robot_1  

                actions = ppc_sm.compute_clean(
                    torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1),
                    torch.cat([goal, desired_orientation], dim=-1),
                    torch.cat([desired_position_1, desired_orientation], dim=-1),
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,8:15] = torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1)
                actions_buffer[:,15] = GripperState.CLOSE
                actions_buffer[:,:8] = actions
                
                
                actions = actions_buffer
                
            else: # other agent is robot_2

                actions = ppc_sm.compute_clean(
                    torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1),
                    torch.cat([goal, desired_orientation], dim=-1),
                    torch.cat([desired_position_2, desired_orientation], dim=-1),
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,:7] = torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1)
                actions_buffer[:,7] = GripperState.CLOSE
                actions_buffer[:,8:] = actions
                
                actions = actions_buffer
            
            
            # print statements => stable behavior??
            print(ppc_sm.sm_state)
            print(ppc_sm.sm_wait_time)        
            if ppc_sm.sm_state == PickPlaceCleanSmState.LIFT_EE and ppc_sm.sm_wait_time >= PickPlaceCleanSmWaitTime.LIFT_EE:
                ppc_sm.sm_state = PickPlaceCleanSmState.REST
                s.clean.append(args[1])
                
                break
            else: 
                if ppc_sm.sm_wait_time > PickPlaceCleanSmWaitTime.TIMEOUT:
                    
                    break
                
    s.ppc_sm = ppc_sm
    return s
    
def pretend_pick_execute(s, args):
    
    # simulation
    sim_env = s.sim_env 
    sim_env_cfg = s.sim_env_cfg
    robot_1_offset = s.robot_1_offset
    robot_2_offset = s.robot_2_offset
    ppc_sm = s.ppc_sm
        
    
    # robot_1
    # -- end-effector frame
    ee_frame_sensor_1 = sim_env.unwrapped.scene["ee_frame_1"]
    tcp_rest_position_1 = ee_frame_sensor_1.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_1_offset
    tcp_rest_orientation_1 = ee_frame_sensor_1.data.target_quat_w[..., 0, :].clone()
    
    # robot_2
    # -- end-effector frame
    ee_frame_sensor_2 = sim_env.unwrapped.scene["ee_frame_2"]
    tcp_rest_position_2 = ee_frame_sensor_2.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_2_offset
    tcp_rest_orientation_2 = ee_frame_sensor_2.data.target_quat_w[..., 0, :].clone()
    
    
    # create action buffers (position + quaternion)
    actions = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
    
    actions[:,0:7] = torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1)
    actions[:,7] = GripperState.OPEN
    
    actions[:,8:15] = torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1)
    actions[:,15] = GripperState.OPEN
    
    
    
    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros((sim_env.unwrapped.num_envs, 4), device=sim_env.unwrapped.device)
    desired_orientation[:, 1] = 1.0
    # create state machine
    ppc_sm = PickPlaceCleanSm(
        sim_env_cfg.sim.dt * sim_env_cfg.decimation, sim_env.unwrapped.num_envs, sim_env.unwrapped.device, position_threshold=0.01
    )
    
    
    
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            dones = sim_env.step(actions)[-2]

            # observations
    
            # robot_1
            # -- end-effector frame
            ee_frame_sensor_1 = sim_env.unwrapped.scene["ee_frame_1"]
            tcp_rest_position_1 = ee_frame_sensor_1.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_1_offset
            tcp_rest_orientation_1 = ee_frame_sensor_1.data.target_quat_w[..., 0, :].clone()
            
            # robot_2
            # -- end-effector frame
            ee_frame_sensor_2 = sim_env.unwrapped.scene["ee_frame_2"]
            tcp_rest_position_2 = ee_frame_sensor_2.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins  - robot_2_offset
            tcp_rest_orientation_2 = ee_frame_sensor_2.data.target_quat_w[..., 0, :].clone()
            
            
            # -- object frame
            object_data: RigidObjectData = sim_env.unwrapped.scene["object"].data
            object_position_1 = object_data.root_pos_w - sim_env.unwrapped.scene.env_origins - robot_1_offset
            object_position_2 = object_data.root_pos_w - sim_env.unwrapped.scene.env_origins - robot_2_offset

            
            # -- target object frame
            desired_position_1 = torch.tensor([[0.2,0.0,0.35]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)
            desired_position_2 = torch.tensor([[0.2,0.0,0.35]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)
            

            if args[0] == ROBOTS[0]: # other agent is robot_1  

                actions = ppc_sm.compute_pick(
                    torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1),
                    torch.cat([object_position_1, desired_orientation], dim=-1),
                    torch.cat([desired_position_1, desired_orientation], dim=-1),
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,8:15] = torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1)
                actions_buffer[:,15] = GripperState.CLOSE
                actions_buffer[:,:8] = actions
                
                
                actions = actions_buffer
                
            else: # other agent is robot_2

                actions = ppc_sm.compute_pick(
                    torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1),
                    torch.cat([object_position_2, desired_orientation], dim=-1),
                    torch.cat([desired_position_2, desired_orientation], dim=-1),
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,:7] = torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1)
                actions_buffer[:,7] = GripperState.CLOSE
                actions_buffer[:,8:] = actions
                
                actions = actions_buffer
            
            
                    
            if ppc_sm.sm_state == PickPlaceCleanSmState.APPROACH_ABOVE_OBJECT and ppc_sm.sm_wait_time >= PickPlaceCleanSmWaitTime.APPROACH_ABOVE_OBJECT: # pretend
                ppc_sm.sm_state = PickPlaceCleanSmState.REST
                
                
                break
            else: 
                if ppc_sm.sm_wait_time > PickPlaceCleanSmWaitTime.TIMEOUT:
                    
                    break
                
    s.ppc_sm = ppc_sm
    return s
def pretend_place_execute(s, args):
    
    # simulation
    sim_env = s.sim_env 
    sim_env_cfg = s.sim_env_cfg
    robot_1_offset = s.robot_1_offset
    robot_2_offset = s.robot_2_offset
    ppc_sm = s.ppc_sm
    
    if args[2] == REGIONS[0]:
        goal = torch.tensor([GOAL_POS]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device) - sim_env.unwrapped.scene.env_origins 
    else:
        goal = torch.tensor([TEMP_POS]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device) - sim_env.unwrapped.scene.env_origins 
    
    if args[0] == ROBOTS[0]:
        goal = goal - robot_1_offset
    else:
        goal = goal - robot_2_offset   
    
    # robot_1
    # -- end-effector frame
    ee_frame_sensor_1 = sim_env.unwrapped.scene["ee_frame_1"]
    tcp_rest_position_1 = ee_frame_sensor_1.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_1_offset
    tcp_rest_orientation_1 = ee_frame_sensor_1.data.target_quat_w[..., 0, :].clone()
    
    # robot_2
    # -- end-effector frame
    ee_frame_sensor_2 = sim_env.unwrapped.scene["ee_frame_2"]
    tcp_rest_position_2 = ee_frame_sensor_2.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_2_offset
    tcp_rest_orientation_2 = ee_frame_sensor_2.data.target_quat_w[..., 0, :].clone()
    
    
    # create action buffers (position + quaternion)
    actions = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
    
    actions[:,0:7] = torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1)
    actions[:,7] = GripperState.CLOSE
    
    actions[:,8:15] = torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1)
    actions[:,15] = GripperState.CLOSE
    
    
    
    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros((sim_env.unwrapped.num_envs, 4), device=sim_env.unwrapped.device)
    desired_orientation[:, 1] = 1.0
    # create state machine
    ppc_sm = PickPlaceCleanSm(
        sim_env_cfg.sim.dt * sim_env_cfg.decimation, sim_env.unwrapped.num_envs, sim_env.unwrapped.device, position_threshold=0.01
    )
    
    
    
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            dones = sim_env.step(actions)[-2]

            # observations
    
            # robot_1
            # -- end-effector frame
            ee_frame_sensor_1 = sim_env.unwrapped.scene["ee_frame_1"]
            tcp_rest_position_1 = ee_frame_sensor_1.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_1_offset
            tcp_rest_orientation_1 = ee_frame_sensor_1.data.target_quat_w[..., 0, :].clone()
            
            # robot_2
            # -- end-effector frame
            ee_frame_sensor_2 = sim_env.unwrapped.scene["ee_frame_2"]
            tcp_rest_position_2 = ee_frame_sensor_2.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins  - robot_2_offset
            tcp_rest_orientation_2 = ee_frame_sensor_2.data.target_quat_w[..., 0, :].clone()
            
            
            # -- object frame
            object_data: RigidObjectData = sim_env.unwrapped.scene["object"].data
            object_position_1 = object_data.root_pos_w - sim_env.unwrapped.scene.env_origins - robot_1_offset
            object_position_2 = object_data.root_pos_w - sim_env.unwrapped.scene.env_origins - robot_2_offset

            
            # -- target object frame
            desired_position_1 = torch.tensor([[0.2,0.0,0.35]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)
            desired_position_2 = torch.tensor([[0.2,0.0,0.35]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)
            

            if args[0] == ROBOTS[0]: # other agent is robot_1  

                actions = ppc_sm.compute_place(
                    torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1),
                    torch.cat([goal, desired_orientation], dim=-1),
                    torch.cat([desired_position_1, desired_orientation], dim=-1),
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,8:15] = torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1)
                actions_buffer[:,15] = GripperState.CLOSE
                actions_buffer[:,:8] = actions
                
                
                actions = actions_buffer
                
            else: # other agent is robot_2

                actions = ppc_sm.compute_place(
                    torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1),
                    torch.cat([goal, desired_orientation], dim=-1),
                    torch.cat([desired_position_2, desired_orientation], dim=-1),
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,:7] = torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1)
                actions_buffer[:,7] = GripperState.CLOSE
                actions_buffer[:,8:] = actions
                
                actions = actions_buffer
            
            # print statements => stable behavior??
            print(ppc_sm.sm_state)
            print(ppc_sm.sm_wait_time)       
            if ppc_sm.sm_state == PickPlaceCleanSmState.APPROACH_ABOVE_GOAL and ppc_sm.sm_wait_time >= PickPlaceCleanSmWaitTime.APPROACH_ABOVE_GOAL: # pretend
                ppc_sm.sm_state = PickPlaceCleanSmState.REST
                
                
                break
            else: 
                if ppc_sm.sm_wait_time > PickPlaceCleanSmWaitTime.TIMEOUT:
                    
                    break
                
    s.ppc_sm = ppc_sm
    return s
def pretend_clean_execute(s, args):
    
    # simulation
    sim_env = s.sim_env 
    sim_env_cfg = s.sim_env_cfg
    robot_1_offset = s.robot_1_offset
    robot_2_offset = s.robot_2_offset
    ppc_sm = s.ppc_sm
    
    
    
    if args[1] == REGIONS[0]:
        goal_pos = GOAL_POS.copy()
    else:
        goal_pos = TEMP_POS.copy()
        
    goal_pos[1] -= 0.1
    goal_pos[2] = 0.0 
    
    
    goal = torch.tensor([goal_pos]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device) - sim_env.unwrapped.scene.env_origins 
    if args[0] == ROBOTS[0]:
        goal = goal - robot_1_offset
    else:
        goal = goal - robot_2_offset     
    
    # robot_1
    # -- end-effector frame
    ee_frame_sensor_1 = sim_env.unwrapped.scene["ee_frame_1"]
    tcp_rest_position_1 = ee_frame_sensor_1.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_1_offset
    tcp_rest_orientation_1 = ee_frame_sensor_1.data.target_quat_w[..., 0, :].clone()
    
    # robot_2
    # -- end-effector frame
    ee_frame_sensor_2 = sim_env.unwrapped.scene["ee_frame_2"]
    tcp_rest_position_2 = ee_frame_sensor_2.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_2_offset
    tcp_rest_orientation_2 = ee_frame_sensor_2.data.target_quat_w[..., 0, :].clone()
    
    
    # create action buffers (position + quaternion)
    actions = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
    
    actions[:,0:7] = torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1)
    actions[:,7] = GripperState.CLOSE
    
    actions[:,8:15] = torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1)
    actions[:,15] = GripperState.CLOSE
    
    
    
    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros((sim_env.unwrapped.num_envs, 4), device=sim_env.unwrapped.device)
    desired_orientation[:, 1] = 1.0
    # create state machine
    ppc_sm = PickPlaceCleanSm(
        sim_env_cfg.sim.dt * sim_env_cfg.decimation, sim_env.unwrapped.num_envs, sim_env.unwrapped.device, position_threshold=0.01
    )
    
    
    
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            dones = sim_env.step(actions)[-2]

            # observations
    
            # robot_1
            # -- end-effector frame
            ee_frame_sensor_1 = sim_env.unwrapped.scene["ee_frame_1"]
            tcp_rest_position_1 = ee_frame_sensor_1.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_1_offset
            tcp_rest_orientation_1 = ee_frame_sensor_1.data.target_quat_w[..., 0, :].clone()
            
            # robot_2
            # -- end-effector frame
            ee_frame_sensor_2 = sim_env.unwrapped.scene["ee_frame_2"]
            tcp_rest_position_2 = ee_frame_sensor_2.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins  - robot_2_offset
            tcp_rest_orientation_2 = ee_frame_sensor_2.data.target_quat_w[..., 0, :].clone()
            
            
            # -- object frame
            object_data: RigidObjectData = sim_env.unwrapped.scene["object"].data
            object_position_1 = object_data.root_pos_w - sim_env.unwrapped.scene.env_origins - robot_1_offset
            object_position_2 = object_data.root_pos_w - sim_env.unwrapped.scene.env_origins - robot_2_offset

            
            # -- target object frame
            desired_position_1 = torch.tensor([[0.2,0.0,0.35]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)
            desired_position_2 = torch.tensor([[0.2,0.0,0.35]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)
            

            if args[0] == ROBOTS[0]: # other agent is robot_1  

                actions = ppc_sm.compute_clean(
                    torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1),
                    torch.cat([goal, desired_orientation], dim=-1),
                    torch.cat([desired_position_1, desired_orientation], dim=-1),
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,8:15] = torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1)
                actions_buffer[:,15] = GripperState.CLOSE
                actions_buffer[:,:8] = actions
                
                
                actions = actions_buffer
                
            else: # other agent is robot_2

                actions = ppc_sm.compute_clean(
                    torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1),
                    torch.cat([goal, desired_orientation], dim=-1),
                    torch.cat([desired_position_2, desired_orientation], dim=-1),
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,:7] = torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1)
                actions_buffer[:,7] = GripperState.CLOSE
                actions_buffer[:,8:] = actions
                
                actions = actions_buffer
            
            
            # print statements => stable behavior??
            print(ppc_sm.sm_state)
            print(ppc_sm.sm_wait_time)        
            if ppc_sm.sm_state == PickPlaceCleanSmState.APPROACH_ABOVE_GOAL and ppc_sm.sm_wait_time >= PickPlaceCleanSmWaitTime.APPROACH_ABOVE_GOAL: # pretend
                ppc_sm.sm_state = PickPlaceCleanSmState.REST
                
                break
            else: 
                if ppc_sm.sm_wait_time > PickPlaceCleanSmWaitTime.TIMEOUT:
                    
                    break
                
    s.ppc_sm = ppc_sm
    return s
       

# other agents actions
    
def pick_other_execute_fn(a, b, s, store):
    
    obj_regions = b.obj_regions.copy()
    holding = b.holding.copy()
    # simulation
    sim_env = s.sim_env 
    sim_env_cfg = s.sim_env_cfg
    robot_1_offset = s.robot_1_offset
    robot_2_offset = s.robot_2_offset
    ppc_sm = s.ppc_sm
        
    if EXEC == 0: # human
        a_other_name,a_ego_name = a.name.split("*")
        print(a_other_name)
        print(a.args[:3])
        while True:
            print("Was pick executed?")
            choice = input("0:No / 1:Yes")
            if int(choice)==0 or int(choice)==1:
                break
        picked = int(choice) == 1
    
    elif EXEC == 1: # random
        
        picked = random.random()<0.9
    
    
        
    if EXEC == 0 or EXEC == 1: # human or random; no change to state for nominal!
        
        if picked: # simulate pick and check result
            
            s = pick_execute(s, a.args[:3])
            
        else: 
            
            s = pretend_pick_execute(s, a.args[:3])
            
    if s.obj_regions[a.args[1]] == "" and s.holding[a.args[0]] == [a.args[1]]: # picked
        print("picked")
        obj_regions[a.args[1]] = ""
        holding[a.args[0]] = [a.args[1]]
    else:
        print("not picked")
    
    return s, EnvObservation(obj_regions=obj_regions,holding=holding)
def pick_other_effects_fn(a, b, store):
    
    obj_regions = b.obj_regions.copy()
    holding = b.holding.copy()
    
    if TRAIN == 0: # human
        a_other_name,a_ego_name = a.name.split("*")
        print(a_other_name)
        print(a.args[:3])
        while True:
            print("Was pick executed?")
            choice = input("0:No / 1:Yes")
            if int(choice)==0 or int(choice)==1:
                break
        picked = int(choice)==1
        print(picked)
    elif TRAIN == 1: # random
        picked = random.random()<=0.9
                            
    if picked: 
        obj_regions[a.args[1]] = ""
        holding[a.args[0]] = [a.args[1]]
    
    return obj_regions,holding
    
def place_other_execute_fn(a, b, s, store):
    
    obj_regions = b.obj_regions.copy()
    holding = b.holding.copy()
    # simulation
    sim_env = s.sim_env 
    sim_env_cfg = s.sim_env_cfg
    robot_1_offset = s.robot_1_offset
    robot_2_offset = s.robot_2_offset
    ppc_sm = s.ppc_sm
            
    
    
    if EXEC == 0: # human
        a_other_name,a_ego_name = a.name.split("*")
        print(a_other_name)
        print(a.args[:3])
        while True:
            print("Was place executed?")
            choice = input("0:No / 1:Yes")
            if int(choice)==0 or int(choice)==1:
                break
        placed = int(choice) == 1
    
    elif EXEC == 1: # random
        
        placed = random.random()<0.9
    
    
        
    if EXEC ==0 or EXEC == 1: # human or random; no change to state for nominal!
        if placed: 
            s = place_execute(s,a.args[:3])
        else:
            s = pretend_place_execute(s,a.args[:3])
            

    
    if s.obj_regions[a.args[1]] == a.args[2] and s.holding[a.args[0]] == []: # placed
        print("placed")
        obj_regions[a.args[1]] = a.args[2]
        holding[a.args[0]] = []
    else:
        print("not placed")
    
    return s, EnvObservation(obj_regions=obj_regions,holding=holding)
def place_other_effects_fn(a, b, store):
    
    obj_regions = b.obj_regions.copy()
    holding = b.holding.copy()
    
    if TRAIN == 0: # human
        a_other_name,a_ego_name = a.name.split("*")
        print(a_other_name)
        print(a.args[:3])
        while True:
            print("Was place executed?")
            choice = input("0:No / 1:Yes")
            if int(choice)==0 or int(choice)==1:
                break
        placed = int(choice)==1
        print(placed)
    elif TRAIN == 1: # random
        placed = random.random()<=0.9
    
    if placed: 
        obj_regions[a.args[1]] = a.args[2]
        holding[a.args[0]] = []
    
    return obj_regions,holding
    
def clean_other_execute_fn(a, b, s, store):
    
    clean = b.clean.copy()
    # simulation
    sim_env = s.sim_env 
    sim_env_cfg = s.sim_env_cfg
    robot_1_offset = s.robot_1_offset
    robot_2_offset = s.robot_2_offset
    ppc_sm = s.ppc_sm
            
    
    if EXEC == 0: # human
        a_other_name,a_ego_name = a.name.split("*")
        print(a_other_name)
        print(a.args[:2])
        while True:
            print("Was clean executed?")
            choice = input("0:No / 1:Yes")
            if int(choice)==0 or int(choice)==1:
                break
        cleaned = int(choice) == 1
    
    elif EXEC == 1: # random
        
        cleaned = random.random()<0.9
    
    
        
    if EXEC ==0 or EXEC == 1: # human or random; no change to state for nominal!
        if cleaned: 
            s = clean_execute(s, a.args[:2])
        else: 
            s = pretend_clean_execute(s, a.args[:2])
            
           

    if a.args[1] in s.clean:
        print("cleaned")
        clean.append(a.args[1])  
    else:
        print("not cleaned")  
    
    
    return s, EnvObservation(clean=clean)
def clean_other_effects_fn(a, b, store):
    
    clean = b.clean.copy()
    
    if TRAIN == 0: # human
        a_other_name,a_ego_name = a.name.split("*")
        print(a_other_name)
        print(a.args[:2])
        while True:
            print("Was clean executed?")
            choice = input("0:No / 1:Yes")
            if int(choice)==0 or int(choice)==1:
                break
        cleaned = int(choice)==1
        print(cleaned)
    elif TRAIN == 1: # random
        cleaned = random.random()<=0.9
    
    if cleaned: 
        
        clean.append(a.args[1])
    
    return clean

# joint actions
def joint_execute_fn(a, b, s, store):
    
    holding = b.holding.copy()
    clean = b.clean.copy()
    obj_regions = b.obj_regions.copy()
    next_actions = b.next_actions.copy()
    # simulation
    sim_env = s.sim_env 
    sim_env_cfg = s.sim_env_cfg
    robot_1_offset = s.robot_1_offset
    robot_2_offset = s.robot_2_offset
    ppc_sm = s.ppc_sm
            
    
    a_other_name,a_ego_name = a.name.split("*")
    
    if a_other_name == "clean_other": # robot, region
        nargs = 2
    elif a_other_name == "nothing_other":
        nargs = 1
    else: # robot, object, region
        nargs = 3
    
    args_ego = a.args[nargs:]
    
    
    # remove ego's previous action
    for na in s.next_actions: 
        name,args = na.split("-")
        args=args.split("%")
        if args[0] == args_ego[0]:
            s.next_actions.remove(na)
    
    
        
    if a_other_name == "pick_other":
        s,obs = pick_other_execute_fn(a, b, s, store)
    elif a_other_name == "place_other":
        s,obs = place_other_execute_fn(a, b, s, store)
    elif a_other_name == "clean_other":
        s,obs = clean_other_execute_fn(a, b, s, store)
    else: 
        pass
        
    if a_ego_name == "pick_ego":
        if s.obj_regions[args_ego[1]] == args_ego[2]: # feasibility check
            if random.random()<0.9:
                s = pick_execute(s, args_ego)
            else:
                s = pretend_pick_execute(s, args_ego)
               
    elif a_ego_name == "place_ego": # no need for feasibility check
        if random.random()<0.9:
            s = place_execute(s, args_ego) 
        else:
            s = pretend_place_execute(s, args_ego)
            
    elif a_ego_name == "clean_ego":
        if args_ego[1] not in s.obj_regions.values(): # feasibility check
            if random.random()<0.9:
                s = clean_execute(s, args_ego)
            else:
                s = pretend_clean_execute(s, args_ego)
    else: 
        pass
    
    
    # next action of ego
    if a_ego_name == "pick_ego" or a_ego_name == "place_ego":
        next_action = a_ego_name[:-3]+"action"+"-"+args_ego[0]+"%"+args_ego[1]+"%"+args_ego[2]
    elif a_ego_name=="clean_ego":
        next_action = a_ego_name[:-3]+"action"+"-"+args_ego[0]+"%"+args_ego[1] 
    else:
        next_action = a_ego_name[:-3]+"action"+"-"+args_ego[0]
        
    # add ego's next action for the other agent     
    s.next_actions.append(next_action)   
    
    next_actions = s.next_actions.copy() 
    
    for na in next_actions: # replace other agents' previous action with noop (temporary, till observation is received)
        name,args = na.split("-")
        args=args.split("%")
        if args[0] != args_ego[0]:
            next_actions.remove(na)
            next_actions.append("nothing_action-"+args[0])
                
    return s, EnvObservation(holding=s.holding,clean=s.clean,obj_regions=s.obj_regions,next_actions=next_actions)  
def joint_effects_fn(a, b, store):

    holding = b.holding.copy()
    clean = b.clean.copy()
    obj_regions = b.obj_regions.copy()
    next_actions = b.next_actions.copy()
    
    a_other_name,a_ego_name = a.name.split("*")
    
    if a_other_name == "clean_other": # robot, region
        nargs = 2
    elif a_other_name == "nothing_other":
        nargs = 1
    else: # robot, object, region
        nargs = 3
    
    args_ego = a.args[nargs:]
 
    
    if a_other_name == "pick_other":
        obj_regions,holding = pick_other_effects_fn(a, b, store)
    elif a_other_name == "place_other": 
        obj_regions,holding = place_other_effects_fn(a, b, store)
    elif a_other_name == "clean_other":
        clean = clean_other_effects_fn(a, b, store)
    else: 
        pass
        

    if a_ego_name == "pick_ego":
        if obj_regions[args_ego[1]] == args_ego[2]: 
            if random.random()<0.9:
                holding[args_ego[0]] = [args_ego[1]]
                obj_regions[args_ego[1]] = ""
    elif a_ego_name == "place_ego":
        if random.random()<0.9:
            holding[args_ego[0]]=[]
            obj_regions[args_ego[1]]=args_ego[2]
    elif a_ego_name == "clean_ego":
        if args_ego[1] not in obj_regions.values():
            if random.random()<0.9:
                clean.append(args_ego[1])
    else: 
        pass
    
    
    # resulting state
    b_temp = copy.deepcopy(b)
    next_actions = get_next_actions_effects(a, b_temp, store) # get next actions from previous belief
    
    o = EnvObservation(holding=holding,clean=clean,obj_regions=obj_regions,next_actions=next_actions)    
    new_belief=b.update(a,o,store)
    return AbstractBeliefSet.from_beliefs([new_belief], store)        
            

# rest of the ego-actions have deterministic effects! 

# Set up environment dynamics
class ToyDiscrete(TampuraEnv):
    
    def initialize(self,ego=f"{ROB}1",s=EnvState()):
        
        self.ego=ego
        
        store = AliasStore()
        
        for rob in ROBOTS:
            
            store.set(rob, rob, "robot")
        # store.set(ego,ego,"robot")
            
        for region in REGIONS:
            store.set(region, region, "region")
        
        store.set(MUG, MUG, "physical")
        
        store.certified.append(Atom("stable",[MUG,REGIONS[0]]))
        store.certified.append(Atom("stable",[MUG,REGIONS[1]]))
        
        store.certified.append(Atom("is_ego",[ego]))

        holding = s.holding
        obj_regions = s.obj_regions
        next_actions = s.next_actions
        clean = s.clean

        b = EnvBelief(holding=holding,clean=clean,obj_regions=obj_regions,
                      next_actions=next_actions)

        return b, store

    def get_problem_spec(self) -> ProblemSpec:
        
        actions_other = ACTION_NAMES
        
        others=[]
        for rob in ROBOTS:
            if rob != self.ego:
                others.append(rob)

        predicates = [
            Predicate("is_ego",["robot"]),
            Predicate("holding", ["robot","physical"]),
            Predicate("stable",["physical","region"]),
            Predicate("in_obj",["physical","region"]),
            Predicate("clean",["region"]),
        ] 
        action_predicates = [Predicate("pick_action",["robot","physical","region"]),
                             Predicate("place_action",["robot","physical","region"]),
                             Predicate("clean_action",["robot","region"]),
                             Predicate("nothing_action",["robot"])]
        
        predicates += action_predicates
        
        possible_outcomes = [[Atom("clean_action",[rob,reg]) for reg in REGIONS]+
                            [Atom("pick_action",[rob,obj,reg]) for obj in OBJ_REGIONS.keys() for reg in REGIONS] + 
                            [Atom("place_action",[rob,obj,reg])for obj in OBJ_REGIONS.keys() for reg in REGIONS] +
                            [Atom("nothing_action",[rob])] for rob in others]
        
        

        
        # modify preconditions, effects and execute functions for observation
        action_schemas_ego = [
            
            # ego-agent
            ActionSchema(
                name="pick_ego",
                inputs=["?rob1","?obj1","?reg1"],
                input_types=["robot","physical","region"],
                preconditions=[Atom("is_ego",["?rob1"]), # is the ego agent
                               Atom("in_obj",["?obj1","?reg1"]), # object is in region from where pick is attempted
                               Not(Exists(Atom("holding",["?rob1","?obj"]),["?obj"],["physical"])), # robot hand is free
                               ],
                verify_effects=[OneOf([Atom("holding",["?rob1","?obj1"]),Atom("in_obj",["?obj1","?reg1"])])], 
            ),
            
            
            ActionSchema(
                name="place_ego",
                inputs=["?rob1","?obj1","?reg1"],
                input_types=["robot","physical","region"],
                preconditions=[Atom("is_ego",["?rob1"]), # is the ego agent
                               Atom("holding",["?rob1","?obj1"]), # robot is holding the object that is to be placed 
                               Not(Atom("in_obj",["?obj1","?reg1"])),
                               Atom("stable",["?obj1","?reg1"]), # region where place is attempted is stable
                               ],
                verify_effects=[OneOf([Atom("holding",["?rob1","?obj1"]),Atom("in_obj",["?obj1","?reg1"])])],  
            ),
            
            ActionSchema(
                name="clean_ego",
                inputs=["?rob1","?reg1"],
                input_types=["robot","region"],
                preconditions=[Atom("is_ego",["?rob1"]),
                               Not(Exists(Atom("in_obj",["?obj","?reg1"]),["?obj"],["physical"])), # region is free
                               Not(Atom("clean",["?reg1"])), # region is unclean
                               Not(Exists(Atom("holding",["?rob1","?obj"]),["?obj"],["physical"])), # robot hand is free
                               ],
                verify_effects=[Atom("clean",["?reg1"])]
            ),
            
            ActionSchema(
                name="nothing_ego",
                inputs=["?rob1"],
                input_types=["robot"],
                preconditions=[Atom("is_ego",["?rob1"])],
            ),
        ]
        
        action_schemas_other = [
            
            # other agents
            ActionSchema(
                name="pick_other",
                inputs=["?rob2","?obj2","?reg3"],
                input_types=["robot","physical","region"],
                preconditions=[Not(Atom("is_ego",["?rob2"])), # is not the ego agent
                               Atom("pick_action",["?rob2","?obj2","?reg3"]), # other agents' turn
                               Atom("in_obj",["?obj2","?reg3"]), # object is in region from where pick is attempted
                               Not(Exists(Atom("holding",["?rob2","?obj"]),["?obj"],["physical"])), # robot hand is free
                               ],
                verify_effects=[OneOf([Atom("holding",["?rob2","?obj2"]),Atom("in_obj",["?obj2","?reg3"])])],
            ),
            
            
            ActionSchema(
                name="place_other",
                inputs=["?rob2","?obj2","?reg3"],
                input_types=["robot","physical","region"],
                preconditions=[Not(Atom("is_ego",["?rob2"])), # is not the ego agent
                               Atom("place_action",["?rob2","?obj2","?reg3"]), # other agents' turn
                               Not(Atom("in_obj",["?obj2","?reg3"])), # object is in region where place is attempted
                               Atom("holding",["?rob2","?obj2"]), # robot is holding the object that is to be placed 
                               Not(Atom("in_obj",["?obj2","?reg3"])),
                               Atom("stable",["?obj2","?reg3"]), # region where place is attempted is stable
                               ],
                verify_effects=[OneOf([Atom("holding",["?rob2","?obj2"]),Atom("in_obj",["?obj2","?reg3"])])],
            ),
            
            ActionSchema(
                name="clean_other",
                inputs=["?rob2","?reg3"],
                input_types=["robot","region"],
                preconditions=[Not(Atom("is_ego",["?rob2"])), # is not the ego agent
                               Atom("clean_action",["?rob2","?reg3"]), # other agents' turn
                               Not(Atom("clean",["?reg3"])),
                               Not(Exists(Atom("in_obj",["?obj","?reg3"]),["?obj"],["physical"])),
                               Not(Exists(Atom("holding",["?rob2","?obj"]),["?obj"],["physical"])), # robot hand is free
                               ],
                verify_effects=[Atom("clean",["?reg3"])],
            ),
            
            ActionSchema(
                name="nothing_other",
                inputs=["?rob2"],
                input_types=["robot"],
                preconditions=[Not(Atom("is_ego",["?rob2"])),
                               Atom("nothing_action",["?rob2"])],
                verify_effects=[],
            )
            
            
        ]
        
        
        
        action_schemas = []
        
        for as_other in action_schemas_other:
            
            as_other_name = as_other.name
            
            for as_ego in action_schemas_ego:
                
                as_ego_name = as_ego.name
                schema = ActionSchema()
                
                # prohibited
                if (as_other_name == "pick_other" and (as_ego_name == "pick_ego" or as_ego_name == "place_ego")) or \
                    (as_other_name == "place_other" and as_ego_name == "place_ego" ) or \
                        (as_other_name == "clean_other" and as_ego_name == "clean_ego") : # not possible under beliefs
                    
                    continue
                
                # special cases
                # assumption: other agent acts before ego agent
                # place_other,clean_ego: works only if other attempts place in region for which clean is not attempted by ego
                elif as_other_name == "place_other" and as_ego_name == "clean_ego":
                    
                    schema.name = as_other_name+"*"+as_ego_name
                    schema.inputs = as_other.inputs + as_ego.inputs
                    schema.input_types = as_other.input_types + as_ego.input_types
                    schema.preconditions = as_other.preconditions + as_ego.preconditions + [Not(Eq("?reg1","?reg3"))] # other does not try to place in region ego is trying to clean
                    schema.effects = as_other.effects + as_ego.effects
                    schema.verify_effects = as_other.verify_effects + as_ego.verify_effects + [OneOf(po) for po in possible_outcomes]
                
                # dependent action: pick other, clean ego
                elif as_other_name == "place_other" and as_ego_name == "pick_ego":
                    
                    schema.name = as_other_name+"*"+as_ego_name
                    schema.inputs = as_other.inputs + as_ego.inputs
                    schema.input_types = as_other.input_types + as_ego.input_types
                    schema.preconditions = as_other.preconditions + [Atom("is_ego",["?rob1"]), Eq("?obj1","?obj2"), Eq("?reg1","?reg3"),Not(Exists(Atom("holding",["?rob1","?obj"]),["?obj"],["physical"]))]
                    schema.effects = as_other.effects + as_ego.effects
                    schema.verify_effects = [OneOf(po) for po in possible_outcomes]+ [OneOf([Atom("in_obj",["?obj1","?reg1"]),Atom("holding",["?rob1","?obj1"]),Atom("holding",["?rob2","?obj2"])])] # special ueffs
                    
                # dependent action: pick other, clean ego
                elif as_other_name == "pick_other" and as_ego_name == "clean_ego":
                    
                    schema.name = as_other_name+"*"+as_ego_name
                    schema.inputs = as_other.inputs + as_ego.inputs
                    schema.input_types = as_other.input_types + as_ego.input_types
                    schema.preconditions = as_other.preconditions + [Atom("is_ego",["?rob1"]), Not(Atom("clean",["?reg1"])),Not(Exists(Atom("holding",["?rob1","?obj"]),["?obj"],["physical"]))]
                    schema.effects = as_other.effects + as_ego.effects
                    schema.verify_effects = as_other.verify_effects + as_ego.verify_effects + [OneOf(po) for po in possible_outcomes]
                    
                    
                # regular cases
                else: 
                    
                    schema.name = as_other_name+"*"+as_ego_name
                    schema.inputs = as_other.inputs + as_ego.inputs
                    schema.input_types = as_other.input_types + as_ego.input_types
                    schema.preconditions = as_other.preconditions + as_ego.preconditions
                    schema.effects = as_other.effects + as_ego.effects
                    schema.verify_effects = as_other.verify_effects + as_ego.verify_effects + [OneOf(po) for po in possible_outcomes]
                    
                schema.execute_fn = joint_execute_fn
                schema.effects_fn = joint_effects_fn

                action_schemas.append(schema)
                    
        # print([a.name+" "+str(a.inputs) for a in action_schemas])
        
        reward = GOAL

        spec = ProblemSpec(
            predicates=predicates,
            action_schemas=action_schemas,
            reward=reward,
        )

        return spec

# Planner

def main():
    
     ############################  TAMPURA  ###################################
        # Initialize environment
    cfg = {}

    # Set some print options to print out abstract belief, action, observation, and reward
    cfg["task"] = "class_uncertain"
    cfg["planner"] = "tampura_policy"
    cfg["global_seed"] = 0
    cfg["vis"] = False
    cfg["flat_width"] = 1
    cfg["pwa"] = 0.2
    cfg["pwk"] = 3.0
    cfg["envelope_threshold"] = 0.015
    cfg["gamma"] = 0.95
    cfg["decision_strategy"] = "prob"
    cfg["learning_strategy"] = "bayes_optimistic"
    cfg["load"] = "null"
    cfg["real_camera"] = False
    cfg["real_execute"] = False
    cfg["symk_selection"] = "unordered"
    cfg["symk_direction"] = "fw"
    cfg["symk_simple"] = True
    cfg["from_scratch"] = True
    cfg["flat_sample"] = True # disable progressive widening

    # Set some print options to print out abstract belief, action, observation, and reward
    cfg["print_options"] = "ab,a,o,r"
    cfg["vis_graph"] = True
    
    cfg['batch_size'] = 500
    cfg['num_samples'] = 500
    
    cfg["max_steps"] = 15
    cfg["num_skeletons"] = 10
    cfg['envelope_threshold'] = 10e-10 # enforce reuse
    
    cfg["flat_sample"] = False # TODO: check; may cause progressive widening
    cfg['save_dir'] = os.getcwd()+"/runs/run{}".format(time.time())
    
    log_dir = "/home/am/Videos/"  
    
    # parse configuration
    sim_env_cfg: FrankaMultiCleanEnvCfgik = parse_env_cfg(
        "Isaac-Lift-Cube-Franka-IK-Abs-multi-clean",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    sim_env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-multi-clean", cfg=sim_env_cfg,render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "IsaacLab"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": False,
        }
        print("[INFO] Recording videos during training.")
        sim_env = gym.wrappers.RecordVideo(sim_env, **video_kwargs)
    # reset environment at start
    # reset environment at start
    sim_env.reset()
    
    
    
    
    if EXEC != 3: # not nominal

        if TRAIN == 0: # human
            cfg['envelope_threshold'] = 10e-10 # enforce reuse
            cfg['num_skeletons'] = 10 # optimism about cooperation
            cfg["batch_size"] = 10  
            cfg["num_samples"] = 10
        elif TRAIN == 1: # random
            cfg['envelope_threshold'] = 10e-10 
            cfg['num_skeletons'] = 100
            cfg['batch_size'] = 500
            cfg['num_samples'] = 500
        elif TRAIN == 2:
            cfg['envelope_threshold'] = 10e-5 # other agent consistent but ego may fail
            cfg['num_skeletons'] = 10 # exhaust all combinations
            cfg["batch_size"] = 100  
            cfg["num_samples"] = 2000 
        
        

        # state
        s = EnvState(holding={ROBOTS[0]:[],ROBOTS[1]:[]},clean=[REGIONS[1]],
                    obj_regions={MUG:REGIONS[0]},
                    next_actions=["nothing_action-"+ROBOTS[0],"nothing_action-"+ROBOTS[1]])
        # s = EnvState(holding={ROBOTS[0]:[],ROBOTS[1]:[]},clean=[REGIONS[1]],
        #             obj_regions={MUG:REGIONS[0]},
        #             next_actions=["nothing_action-"+ROBOTS[0],"pick_action-"+ROBOTS[1]+"%"+MUG+"%"+REGIONS[0]]) # TEST
        s.sim_env = sim_env
        s.sim_env_cfg = sim_env_cfg
        s.robot_1_offset = torch.tensor([[0.0, -0.20, 0.0]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)
        s.robot_2_offset = torch.tensor([[0.0, 0.20, 0.0]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)
        s.ppc_sm = PickPlaceCleanSm(sim_env_cfg.sim.dt * sim_env_cfg.decimation, sim_env.unwrapped.num_envs, sim_env.unwrapped.device, position_threshold=0.01)

            
        # for robot1
        # Initialize 
        env = ToyDiscrete(config=cfg)
        b0, store= env.initialize(ego=ROBOTS[0],s=s)


        # Set up logger to print info
        setup_logger(cfg["save_dir"], logging.INFO)

        # Initialize the policy
        planner = TampuraPolicy(config = cfg, problem_spec = env.problem_spec)

        env.state = s

        b=b0

        assert env.problem_spec.verify(store)

        save_config(planner.config, planner.config["save_dir"])

        history = RolloutHistory(planner.config)

        st = time.time()
        for step in range(100):
            
            # robot 1 acts
            env.state = s
            b.next_actions = s.next_actions # important!!
            a_b = b.abstract(store)
            reward = env.problem_spec.get_reward(a_b, store)
            
            if reward:
                print("goal achieved")
                break
            
            logging.info("\n" + ("=" * 10) + "t=" + str(step) + ("=" * 10))
            if "s" in planner.print_options:
                logging.info("State: " + str(s))
            if "b" in planner.print_options:
                logging.info("Belief: " + str(b))
            if "ab" in planner.print_options:
                logging.info("Abstract Belief: " + str(a_b))
            if "r" in planner.print_options:
                logging.info("Reward: " + str(reward))
            
            
            action, info, store = planner.get_action(b, store) 
            

            if action.name == "no-op": # symk planning failure returns no-op; else we would get "nothing_ego"
                bp = copy.deepcopy(b)
                observation = None
                
                # replace previous action with nothing 
                for ac in s.next_actions:
                    name,args = ac.split("-")
                    args=args.split("%")
                    if args[0] == ROBOTS[1]:
                        s.next_actions.remove(ac)
                        s.next_actions.append("nothing_action-"+args[0])
                
                continue # skip the rest of the loop (asking for next action) and repeat MDP solving step
                
            else:
                
                if "a" in planner.print_options:
                    logging.info("Action: " + str(action))
                observation= env.step(action, b, store) # should call execute function
                bp = b.update(action, observation, store)

                if planner.config["vis"]:
                    env.vis_updated_belief(bp, store)

            a_bp = bp.abstract(store)
            history.add(s, b, a_b, action, observation, reward, info, store, time.time() - st)

            reward = env.problem_spec.get_reward(a_bp, store)
            
            
            if "o" in planner.print_options:
                logging.info("Observation: " + str(observation))
            if "sp" in planner.print_options:
                logging.info("Next State: " + str(env.state))
            if "bp" in planner.print_options:
                logging.info("Next Belief: " + str(bp))
            if "abp" in planner.print_options:
                logging.info("Next Abstract Belief: " + str(a_bp))
            if "rp" in planner.print_options:
                logging.info("Next Reward: " + str(reward))

            # update the belief
            b = bp
            # update the state as modified by ego!
            s = env.state

            # remove previous action (nothing)
            for ac in s.next_actions:
                name,args = ac.split("-")
                args=args.split("%")
                if args[0] == ROBOTS[1]:
                    s.next_actions.remove(ac)
                    
            
            # get current action
                        
            next_actions = get_next_actions_execute(action,b,store)
            for ac in next_actions:
                s.next_actions.append(ac)
                
        
            # true outcome evaluated in functions
            
        # history.add(env.state, bp, a_bp, None, None, reward, info, store, time.time() - st)
            
        logging.info("=" * 20)

        env.wrapup()

        # if not planner.config["real_execute"]:
        #     save_run_data(history, planner.config["save_dir"])
                    
    else: # nominal
        
        # train random
        
        # state
        s = EnvState(holding={ROBOTS[0]:[],ROBOTS[1]:[]},clean=[REGIONS[1]],
                    obj_regions={MUG:REGIONS[0]},
                    next_actions=["nothing_action-"+ROBOTS[0],"nothing_action-"+ROBOTS[1]])
        s.sim_env = sim_env
        s.sim_env_cfg = sim_env_cfg
        s.robot_1_offset = torch.tensor([[0.0, -0.20, 0.0]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)
        s.robot_2_offset = torch.tensor([[0.0, 0.20, 0.0]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)
        s.ppc_sm = PickPlaceCleanSm(sim_env_cfg.sim.dt * sim_env_cfg.decimation, sim_env.unwrapped.num_envs, sim_env.unwrapped.device, position_threshold=0.01)

        save_dir = os.getcwd()+"/runs/run{}".format(time.time())
        # for robot1
        # Initialize 
        save_dir_1 = save_dir + "/planner1" 
        cfg1 = cfg.copy()
        cfg1['save_dir'] = save_dir_1
        env1 = ToyDiscrete(config=cfg1)
        b01, store1= env1.initialize(ego=ROBOTS[0],s=s)
        # for robot2
        # Initialize 
        save_dir_2 = save_dir + "/planner2"
        cfg2 = cfg.copy()
        cfg2['save_dir'] = save_dir_2
        env2 = ToyDiscrete(config=cfg2)
        b02, store2= env2.initialize(ego=ROBOTS[1],s=s)

        # Set up logger to print info
        setup_logger(cfg1["save_dir"], logging.INFO)
        setup_logger(cfg2["save_dir"], logging.INFO)

        # Initialize the policy

        planner1 = TampuraPolicy(config = cfg1, problem_spec = env1.problem_spec)
        planner2 = TampuraPolicy(config = cfg2, problem_spec = env2.problem_spec)

        env1.state = s
        env2.state = s
        
        b1=b01
        b2=b02

        assert env1.problem_spec.verify(store1)
        assert env2.problem_spec.verify(store2)

        save_config(planner1.config, planner1.config["save_dir"])
        save_config(planner2.config, planner2.config["save_dir"])

        history1 = RolloutHistory(planner1.config)
        history2 = RolloutHistory(planner2.config)

        st = time.time()
        for step in range(100):

            # robot 1 acts
            env1.state = env2.state # important!!
            s1 = env1.state
            b1.next_actions = s1.next_actions # important!!
            a_b1 = b1.abstract(store1)
            reward1 = env1.problem_spec.get_reward(a_b1, store1)
            
            if reward1:
                print("goal achieved")
                break  
            
            logging.info("\n robot 1 ")
            logging.info("\n" + ("=" * 10) + "t=" + str(step) + ("=" * 10))
            if "s" in planner1.print_options:
                logging.info("State: " + str(s1))
            if "b" in planner1.print_options:
                logging.info("Belief: " + str(b1))
            if "ab" in planner1.print_options:
                logging.info("Abstract Belief: " + str(a_b1))
            if "r" in planner1.print_options:
                logging.info("Reward: " + str(reward1))
            
            
            action1, info1, store1 = planner1.get_action(b1, store1) # should only call effects functions!!??
            
            
            
            if action1.name == "no-op": # symk planning failure returns no-op; else we would get "nothing_ego"
                bp1 = copy.deepcopy(b1)
                observation1 = None
                
                # replace previous action with nothing 
                for ac in s.next_actions:
                    name,args = ac.split("-")
                    args=args.split("%")
                    if args[0] != ROBOTS[0]: # other agent
                        s.next_actions.remove(ac)
                        s.next_actions.append("nothing_action-"+args[0])
                
                continue 
            else:
                if "a" in planner1.print_options:
                    logging.info("Action: " + str(action1))

                observation1= env1.step(action1, b1, store1) # should call execute function
                bp1 = b1.update(action1, observation1, store1)

                if planner1.config["vis"]:
                    env1.vis_updated_belief(bp1, store1)

            a_bp1 = bp1.abstract(store1)
            history1.add(s1, b1, a_b1, action1, observation1, reward1, info1, store1, time.time() - st)

            reward1 = env1.problem_spec.get_reward(a_bp1, store1)
            
            if "o" in planner1.print_options:
                logging.info("Observation: " + str(observation1))
            if "sp" in planner1.print_options:
                logging.info("Next State: " + str(env1.state))
            if "bp" in planner1.print_options:
                logging.info("Next Belief: " + str(bp1))
            if "abp" in planner1.print_options:
                logging.info("Next Abstract Belief: " + str(a_bp1))
            if "rp" in planner1.print_options:
                logging.info("Next Reward: " + str(reward1))

            # update the belief
            b1 = bp1
            
            # robot 2 acts
            env2.state = env1.state # important!!
            s2 = env2.state
            b2.next_actions = s2.next_actions # important!!
            a_b2 = b2.abstract(store2)
            reward2 = env2.problem_spec.get_reward(a_b2, store2)
            
            if reward2:
                print("goal achieved")
                break  

            logging.info("\n robot 2 ")
            logging.info("\n" + ("=" * 10) + "t=" + str(step) + ("=" * 10))
            if "s" in planner2.print_options:
                logging.info("State: " + str(s2))
            if "b" in planner1.print_options:
                logging.info("Belief: " + str(b2))
            if "ab" in planner1.print_options:
                logging.info("Abstract Belief: " + str(a_b2))
            if "r" in planner1.print_options:
                logging.info("Reward: " + str(reward2))
            
            
            action2, info2, store2 = planner2.get_action(b2, store2) # should only call effects functions!!??
            

            if action2.name == "no-op": # symk planning failure returns no-op; else we would get "nothing_ego"
                bp2 = copy.deepcopy(b2)
                observation2 = None
                
                # replace previous action with nothing 
                for ac in s.next_actions:
                    name,args = ac.split("-")
                    args=args.split("%")
                    if args[0] != ROBOTS[1]:
                        s.next_actions.remove(ac)
                        s.next_actions.append("nothing_action-"+args[0])
                
                continue 
            else:
                if "a" in planner2.print_options:
                    logging.info("Action: " + str(action2))
                observation2= env2.step(action2, b2, store2) # should call execute function
                bp2 = b2.update(action2, observation2, store2)

                if planner2.config["vis"]:
                    env2.vis_updated_belief(bp2, store2)

            a_bp2 = bp2.abstract(store2)
            history2.add(s2, b2, a_b2, action2, observation2, reward2, info2, store2, time.time() - st)

            reward2 = env2.problem_spec.get_reward(a_bp2, store2)
            
            if "o" in planner2.print_options:
                logging.info("Observation: " + str(observation2))
            if "sp" in planner2.print_options:
                logging.info("Next State: " + str(env2.state))
            if "bp" in planner2.print_options:
                logging.info("Next Belief: " + str(bp2))
            if "abp" in planner2.print_options:
                logging.info("Next Abstract Belief: " + str(a_bp2))
            if "rp" in planner2.print_options:
                logging.info("Next Reward: " + str(reward2))

            # update the belief
            b2 = bp2

        history1.add(env1.state, bp1, a_bp1, None, None, reward1, info1, store1, time.time() - st)
        history2.add(env2.state, bp2, a_bp2, None, None, reward2, info2, store2, time.time() - st)
            
        logging.info("=" * 20)

        env1.wrapup()
        env2.wrapup()

        # if not planner1.config["real_execute"]:
        #     save_run_data(history1, planner1.config["save_dir"])

        # if not planner2.config["real_execute"]:
        #     save_run_data(history2, planner2.config["save_dir"])
        
    s.sim_env.close()
    

    

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
