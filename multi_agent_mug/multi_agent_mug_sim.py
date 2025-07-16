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
from isaaclab.sensors import FrameTransformer


from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData

import isaaclab_tasks  # noqa: F401
from multi_mug_env_cfg import FrankaMultiMugEnvCfgik
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# initialize warp
wp.init()

class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)

class MugSmState:
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
    APPROACH_INFRONT_HANDLE = wp.constant(9)
    APPROACH_HANDLE = wp.constant(10)
    GRASP_HANDLE = wp.constant(11)
    OPEN_DRAWER = wp.constant(12)
    RELEASE_HANDLE = wp.constant(13)
    CLOSE_DRAWER = wp.constant(14)
    
class MugSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.6)
    GRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(1.0)
    APPROACH_ABOVE_GOAL = wp.constant(0.5)
    APPROACH_GOAL = wp.constant(0.1) # low
    RELEASE_OBJECT = wp.constant(0.1) # low
    LIFT_EE = wp.constant(1.0)
    APPROACH_INFRONT_HANDLE = wp.constant(1.25)
    APPROACH_HANDLE = wp.constant(1.0)
    GRASP_HANDLE = wp.constant(1.0)
    OPEN_DRAWER = wp.constant(2.0)
    RELEASE_HANDLE = wp.constant(0.2)
    CLOSE_DRAWER = wp.constant(3.0)
    
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
    if state == MugSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= MugSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = MugSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == MugSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= MugSmWaitTime.APPROACH_ABOVE_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = MugSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == MugSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= MugSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = MugSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == MugSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= MugSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = MugSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == MugSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= MugSmWaitTime.LIFT_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = MugSmState.LIFT_OBJECT
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
    if state == MugSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= MugSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = MugSmState.RELEASE_OBJECT
            sm_wait_time[tid] = 0.0
    # elif state == MugSmState.APPROACH_ABOVE_GOAL:
    #     des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
    #     gripper_state[tid] = GripperState.CLOSE
    #     if distance_below_threshold(
    #         wp.transform_get_translation(ee_pose[tid]),
    #         wp.transform_get_translation(des_ee_pose[tid]),
    #         position_threshold,
    #     ):
    #         # wait for a while
    #         if sm_wait_time[tid] >= MugSmWaitTime.APPROACH_ABOVE_GOAL:
    #             # move to next state and reset wait time
    #             sm_state[tid] = MugSmState.APPROACH_GOAL
    #             sm_wait_time[tid] = 0.0
    # elif state == MugSmState.APPROACH_GOAL:
    #     des_ee_pose[tid] = object_pose[tid]
    #     gripper_state[tid] = GripperState.CLOSE
    #     if distance_below_threshold(
    #         wp.transform_get_translation(ee_pose[tid]),
    #         wp.transform_get_translation(des_ee_pose[tid]),
    #         position_threshold,
    #     ):
    #         if sm_wait_time[tid] >= MugSmWaitTime.APPROACH_GOAL:
    #             # move to next state and reset wait time
    #             sm_state[tid] = MugSmState.RELEASE_OBJECT
    #             sm_wait_time[tid] = 0.0
    elif state == MugSmState.RELEASE_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= MugSmWaitTime.RELEASE_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = MugSmState.LIFT_EE
            sm_wait_time[tid] = 0.0
    elif state == MugSmState.LIFT_EE:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= MugSmWaitTime.LIFT_EE:
                # move to next state and reset wait time
                sm_state[tid] = MugSmState.LIFT_EE # new
                sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]

@wp.kernel
def infer_open_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    handle_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    handle_approach_offset: wp.array(dtype=wp.transform),
    handle_grasp_offset: wp.array(dtype=wp.transform),
    drawer_opening_rate: wp.array(dtype=wp.transform),
    position_threshold: float,
    object_pose: wp.array(dtype=wp.transform),
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == MugSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= MugSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = MugSmState.APPROACH_INFRONT_HANDLE
            sm_wait_time[tid] = 0.0
    elif state == MugSmState.APPROACH_INFRONT_HANDLE:
        des_ee_pose[tid] = wp.transform_multiply(handle_approach_offset[tid], handle_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= MugSmWaitTime.APPROACH_INFRONT_HANDLE:
                # move to next state and reset wait time
                sm_state[tid] = MugSmState.APPROACH_HANDLE
                sm_wait_time[tid] = 0.0
    elif state == MugSmState.APPROACH_HANDLE:
        des_ee_pose[tid] = handle_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= MugSmWaitTime.APPROACH_HANDLE:
                # move to next state and reset wait time
                sm_state[tid] = MugSmState.GRASP_HANDLE
                sm_wait_time[tid] = 0.0
    elif state == MugSmState.GRASP_HANDLE:
        des_ee_pose[tid] = wp.transform_multiply(handle_grasp_offset[tid], handle_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= MugSmWaitTime.GRASP_HANDLE:
            # move to next state and reset wait time
            sm_state[tid] = MugSmState.OPEN_DRAWER
            sm_wait_time[tid] = 0.0
    elif state == MugSmState.OPEN_DRAWER:
        des_ee_pose[tid] = wp.transform_multiply(drawer_opening_rate[tid], handle_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= MugSmWaitTime.OPEN_DRAWER:
            # move to next state and reset wait time
            sm_state[tid] = MugSmState.RELEASE_HANDLE
            sm_wait_time[tid] = 0.0
    elif state == MugSmState.RELEASE_HANDLE:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= MugSmWaitTime.RELEASE_HANDLE:
            # move to next state and reset wait time
            sm_state[tid] = MugSmState.LIFT_EE
            sm_wait_time[tid] = 0.0
    elif state == MugSmState.LIFT_EE:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= MugSmWaitTime.LIFT_EE:
                # move to next state and reset wait time
                sm_state[tid] = MugSmState.LIFT_EE # new
                sm_wait_time[tid] = 0.0
    
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]

@wp.kernel
def infer_close_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    handle_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    handle_approach_offset: wp.array(dtype=wp.transform),
    handle_grasp_offset: wp.array(dtype=wp.transform),
    drawer_closing_rate: wp.array(dtype=wp.transform),
    position_threshold: float,
    object_pose: wp.array(dtype=wp.transform),
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == MugSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= MugSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = MugSmState.APPROACH_INFRONT_HANDLE
            sm_wait_time[tid] = 0.0
    elif state == MugSmState.APPROACH_INFRONT_HANDLE:
        des_ee_pose[tid] = wp.transform_multiply(handle_approach_offset[tid], handle_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= MugSmWaitTime.APPROACH_INFRONT_HANDLE:
                # move to next state and reset wait time
                sm_state[tid] = MugSmState.APPROACH_HANDLE
                sm_wait_time[tid] = 0.0
    elif state == MugSmState.APPROACH_HANDLE:
        des_ee_pose[tid] = handle_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= MugSmWaitTime.APPROACH_HANDLE:
                # move to next state and reset wait time
                sm_state[tid] = MugSmState.GRASP_HANDLE
                sm_wait_time[tid] = 0.0
    elif state == MugSmState.GRASP_HANDLE:
        des_ee_pose[tid] = wp.transform_multiply(handle_grasp_offset[tid], handle_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= MugSmWaitTime.GRASP_HANDLE:
            # move to next state and reset wait time
            sm_state[tid] = MugSmState.CLOSE_DRAWER
            sm_wait_time[tid] = 0.0
    elif state == MugSmState.CLOSE_DRAWER:
        des_ee_pose[tid] = wp.transform_multiply(drawer_closing_rate[tid], handle_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= MugSmWaitTime.CLOSE_DRAWER:
            # move to next state and reset wait time
            sm_state[tid] = MugSmState.RELEASE_HANDLE
            sm_wait_time[tid] = 0.0
    elif state == MugSmState.RELEASE_HANDLE:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= MugSmWaitTime.RELEASE_HANDLE:
            # move to next state and reset wait time
            sm_state[tid] = MugSmState.LIFT_EE
            sm_wait_time[tid] = 0.0
    elif state == MugSmState.LIFT_EE:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= MugSmWaitTime.LIFT_EE:
                # move to next state and reset wait time
                sm_state[tid] = MugSmState.LIFT_EE # new
                sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]

@wp.kernel
def infer_transit_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    position_threshold: float,
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == MugSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= MugSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = MugSmState.APPROACH_GOAL
            sm_wait_time[tid] = 0.0
    elif state == MugSmState.APPROACH_GOAL:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= MugSmWaitTime.APPROACH_GOAL:
                # move to next state and reset wait time
                sm_state[tid] = MugSmState.APPROACH_GOAL
                sm_wait_time[tid] = 0.0
    
    
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]

@wp.kernel
def infer_transfer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    position_threshold: float,
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == MugSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= MugSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = MugSmState.APPROACH_GOAL
            sm_wait_time[tid] = 0.0
    elif state == MugSmState.APPROACH_GOAL:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= MugSmWaitTime.APPROACH_GOAL:
                # move to next state and reset wait time
                sm_state[tid] = MugSmState.APPROACH_GOAL
                sm_wait_time[tid] = 0.0
    
    
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]

class MugSm:
    """A simple state machine in a robot's task space.

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
        self.offset[:, 2] = 0.2
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # approach infront of the handle
        self.handle_approach_offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.handle_approach_offset[:, 0] = -0.1
        self.handle_approach_offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # handle grasp offset
        self.handle_grasp_offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.handle_grasp_offset[:, 0] = 0.02
        self.handle_grasp_offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # drawer opening rate
        self.drawer_opening_rate = torch.zeros((self.num_envs, 7), device=self.device)
        self.drawer_opening_rate[:, 0] = -0.015
        self.drawer_opening_rate[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # drawer opening rate
        self.drawer_closing_rate = torch.zeros((self.num_envs, 7), device=self.device)
        self.drawer_closing_rate[:, 0] = 0.015*5
        self.drawer_closing_rate[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)
        self.handle_approach_offset_wp = wp.from_torch(self.handle_approach_offset, wp.transform)
        self.handle_grasp_offset_wp = wp.from_torch(self.handle_grasp_offset, wp.transform)
        self.drawer_opening_rate_wp = wp.from_torch(self.drawer_opening_rate, wp.transform)
        self.drawer_closing_rate_wp = wp.from_torch(self.drawer_closing_rate, wp.transform)
        
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
        self.offset[:, 0] = -0.3
        self.offset[:, 2] = 0.3
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
    def compute_open(self, ee_pose: torch.Tensor, handle_pose: torch.Tensor, object_pose: torch.Tensor):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        handle_pose = handle_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        handle_pose_wp = wp.from_torch(handle_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_open_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                handle_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.handle_approach_offset_wp,
                self.handle_grasp_offset_wp,
                self.drawer_opening_rate_wp,
                self.position_threshold,
                object_pose
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)
    def compute_close(self, ee_pose: torch.Tensor, handle_pose: torch.Tensor, object_pose: torch.Tensor):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        handle_pose = handle_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        handle_pose_wp = wp.from_torch(handle_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_close_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                handle_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.handle_approach_offset_wp,
                self.handle_grasp_offset_wp,
                self.drawer_closing_rate_wp,
                self.position_threshold,
                object_pose,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)
    def compute_transit(self, ee_pose: torch.Tensor, object_pose: torch.Tensor) -> torch.Tensor:
        """Compute the desired state of the robot's end-effector and the gripper."""

        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_transit_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.position_threshold,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)
    def compute_transfer(self, ee_pose: torch.Tensor, object_pose: torch.Tensor) -> torch.Tensor:
        """Compute the desired state of the robot's end-effector and the gripper."""

        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_transfer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
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
from tampura.symbolic import OBJ, Atom, ForAll, Not, Exists, Or, And, OneOf, eval_expr
from tampura.policies.tampura_policy import TampuraPolicy
from tampura.config.config import load_config, setup_logger

from pick_successes import SIMPLE_PICK_EGO_SIM, CABINET_PICK_EGO_SIM


# TODO: different training and execution scenarios, study the MDPs
# TODO: change action templates for open open, etc.
# 0: human, 1: random, 2: inactive
TRAIN = 2
# 0: human, 1: random, 2: inactive, 3: nominal
EXEC = 2
# biased towards door related operations
DOOR_BIAS = False # TODO

ROB = "robot_"
REG = "region_"
MUG = "mug"
DOOR = "door"
REGIONS = [f"{REG}{MUG}",f"{REG}{DOOR}",f"{REG}stable_mug"]
ACTION_NAMES = ["transit_action","transfer_action","pick_action","place_action","open_action","close_action","nothing_action"]

# problem specification: try with just one robot to demonstrate how overall cost increases
ROBOTS=[f"{ROB}1",f"{ROB}2"]
ROB_REGIONS = {ROBOTS[0]:REGIONS[-1],ROBOTS[1]:REGIONS[-1]} # long horizon: combinatorial explosion
# ROB_REGIONS = {ROBOTS[0]:REGIONS[1],ROBOTS[1]:REGIONS[0]} # short horizon: kind of works?
OBJ_REGIONS={MUG:REGIONS[0]}
# probability of success for open/close by ego
OPEN_EGO = 0.9
CLOSE_EGO = 0.9
# higher num_samples needed to learn true transition model

# Test 
GOAL = And([Exists(Atom("holding",["?rob",MUG]),["?rob"],["robot"]),Not(Atom("open",[DOOR]))])

REGIONS_LOC = {REGIONS[0]:[0.25,0.0,1.0],REGIONS[1]:[0.25,0.0,0.75],REGIONS[2]:[0.25,0.0,1.005]}
PICK_POS = [0.4, 0.0, 0.75]

# State of the environment
@dataclass
class EnvState(State):
    holding: Dict[str,List[str]] = field(default_factory=lambda: {})
    open_door: bool = field(default_factory=lambda: False)
    rob_regions: Dict[str,str] = field(default_factory=lambda:{})
    obj_regions: Dict[str,str] = field(default_factory=lambda:{})
    next_actions: List[str] = field(default_factory=lambda: [])
    sim_env: Any = field(default_factory=lambda: None)
    sim_env_cfg: Any = field(default_factory=lambda: None)
    robot_1_offset: Any = field(default_factory=lambda: None)
    robot_2_offset: Any = field(default_factory=lambda: None)
    mug_sm: Any = field(default_factory=lambda: None)
    
# Observation space
@dataclass
class EnvObservation(Observation):
    holding: Dict[str,List[str]] = field(default_factory=lambda: {})
    open_door: bool = field(default_factory=lambda: False)
    rob_regions: Dict[str,str] = field(default_factory=lambda:{})
    obj_regions: Dict[str,str] = field(default_factory=lambda:{})
    next_actions: List[str] = field(default_factory=lambda: [])

# Belief space
class EnvBelief(Belief):
    def __init__(self, holding={},open_door=False,rob_regions={},obj_regions={},next_actions=[]):
        # true state
        self.holding = holding
        self.open_door = open_door
        self.rob_regions = rob_regions
        self.obj_regions = obj_regions
        self.next_actions = next_actions
        

    def update(self, a, o, s):
        
        # dictionary mutations are IN-PLACE!!! use .copy()!!
        holding = self.holding.copy() 
        open_door = self.open_door
        rob_regions = self.rob_regions.copy()
        obj_regions = self.obj_regions.copy()
        next_actions = self.next_actions.copy()
        
        
        # get argument index for ego agent
        
        a_other_name,a_ego_name = a.name.split("*")
        
        if a_other_name == "transfer_other":
            nargs_other = 4
        elif a_other_name == "nothing_other" or a_other_name == "open_other" or a_other_name == "close_other":
            nargs_other = 1
        else:
            nargs_other = 3
            
        a_ego_args = a.args[nargs_other:]
        
        # the previous values of variables change depending on the action
        
        # action_other
        if a_other_name == "pick_other" or a_other_name == "place_other":
            holding[a.args[0]] = o.holding[a.args[0]]
            obj_regions[a.args[1]] = o.obj_regions[a.args[1]]
        elif a_other_name == "transit_other" or a_other_name == "transfer_other":
            rob_regions[a.args[0]] = o.rob_regions[a.args[0]]
        elif a_other_name == "open_other" or a_other_name == "close_other":
            open_door = o.open_door
        else: 
            pass
        
        # action ego
        if a_ego_name == "pick_ego" or a_ego_name == "place_ego":
            holding[a_ego_args[0]] = o.holding[a_ego_args[0]]
            obj_regions[a_ego_args[1]] = o.obj_regions[a_ego_args[1]]
        elif a_ego_name == "transit_ego" or a_ego_name == "transfer_ego":
            rob_regions[a_ego_args[0]] = o.rob_regions[a_ego_args[0]]
        elif a_ego_name == "open_ego" or a_ego_name == "close_ego":
            open_door = o.open_door
        else: 
            pass
           
        next_actions = o.next_actions
            
        return EnvBelief(holding=holding,open_door=open_door,rob_regions=rob_regions,obj_regions=obj_regions,next_actions=next_actions)

    def abstract(self, store: AliasStore):
        
        ab = []
        
        # true state
        for rob in self.holding.keys():
            ab += [Atom("holding",[rob,obj]) for obj in self.holding[rob]]
        for rob in self.rob_regions.keys():
            ab += [Atom("in_rob",[rob,self.rob_regions[rob]])]
        for obj in self.obj_regions.keys():
            if self.obj_regions[obj] !="":
                ab += [Atom("in_obj",[obj,self.obj_regions[obj]])]
        if self.open_door:
            ab += [Atom("open",[DOOR])]
        
        # next actions
        if self.next_actions != []:
            for next_action in self.next_actions:
                
                name,args = next_action.split("-")
                args=list(args.split("%"))
                
                rob=args[0]
                if Atom("is_ego",[rob]) not in store.certified:
                    ab += [Atom(name,args)]
            
        return AbstractBelief(ab)

    # def vectorize(self):
    #     return np.array([int(obj in self.holding) for obj in OBJECTS])
      
def get_next_actions_execute(a, b, store): # human operator : tedious, kind of works
    
    a_other_name,a_ego_name = a.name.split("*")
    if a_other_name == "transfer_other":
        n_args=4
    elif a_other_name == "open_other" or a_other_name == "close_other" or a_other_name == "nothing_other":
        n_args = 1
    else:
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
    
    for rob in others: 
        
        
        applicable_actions_rob=[]
        
        applicable_actions_rob.append(Atom("nothing_action",[rob]))
        applicable_actions_rob.append(Atom("open_action",[rob]))
        applicable_actions_rob.append(Atom("close_action",[rob]))
        
        for reg in REGIONS:
            
            applicable_actions_rob.append(Atom("transit_action",[rob,reg]))
            
            for obj in OBJ_REGIONS.keys():
                
                applicable_actions_rob.append(Atom("transfer_action",[rob,obj,reg]))                    
                applicable_actions_rob.append(Atom("place_action",[rob,obj,reg]))
                
                if Atom("pick_action",[rob,obj]) not in applicable_actions_rob:
                    applicable_actions_rob.append(Atom("pick_action",[rob,obj]))
                
        
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
        
        if name=="transit_action":
            a_other=name+"-"+rob+"%"+args[1]
        elif name == "transfer_action":
            a_other=name+"-"+rob+"%"+args[1]+"%"+args[2]
        elif name == "pick_action":
            a_other=name+"-"+rob+"%"+args[1]
        elif name == "place_action":
            a_other=name+"-"+rob+"%"+args[1]
        else: # open, close, nothing
            a_other=name+"-"+rob
            
        next_actions.append(a_other)
            
    return next_actions # for all the agents
def get_next_actions_effects(a, b, store): # human operator : tedious, kind of works
    
    
    a_other_name,a_ego_name = a.name.split("*")
    
    if a_other_name == "transfer_other":
        n_args=4
    elif a_other_name == "open_other" or a_other_name == "close_other" or a_other_name == "nothing_other":
        n_args = 1
    else:
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
        
        applicable_actions_rob.append(Atom("nothing_action",[rob]))
        applicable_actions_rob.append(Atom("open_action",[rob]))
        applicable_actions_rob.append(Atom("close_action",[rob]))
        
        for reg in REGIONS:
            
            applicable_actions_rob.append(Atom("transit_action",[rob,reg]))
            
            for obj in OBJ_REGIONS.keys():
                
                applicable_actions_rob.append(Atom("transfer_action",[rob,obj,reg]))                    
                applicable_actions_rob.append(Atom("place_action",[rob,obj,reg]))
                
                if Atom("pick_action",[rob,obj]) not in applicable_actions_rob:
                    applicable_actions_rob.append(Atom("pick_action",[rob,obj]))
                
                
        
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
            
            if DOOR_BIAS:
                
                if b.rob_regions[rob]!=REGIONS[1]: # if not at door, go to door
                    observed_action_rob=Atom("transit_action",[rob,REGIONS[1]])
                else:
                    
                    if b.open_door: #  close if ego is not attempting close or holding object or attempts pick
                        if a_ego_name == "pick_ego" or b.holding[a.args[n_args]]!=[]:
                            observed_action_rob=Atom("close_action",[rob])
                        else:
                            observed_action_rob=Atom("nothing_action",[rob])                        
                    else: # door is closed
                        if a_ego_name == "open_ego":
                            observed_action_rob=Atom("nothing_action",[rob])
                        else:
                            if b.holding[a.args[n_args]]==[]:
                                observed_action_rob=Atom("open_action",[rob])
                            else:
                                observed_action_rob=Atom("nothing_action",[rob])
                                
                    # print(observed_action_rob)
                    
            else:
                observed_action_rob = random.choice(applicable_actions_rob)
                
            
            
        elif TRAIN == 2: # inactive agent
            observed_action_rob = Atom("nothing_action",[rob])
        
                        
        name=observed_action_rob.pred_name
        args=observed_action_rob.args
        
        if name=="transit_action":
            a_other=name+"-"+rob+"%"+args[1]
        elif name == "transfer_action":
            a_other=name+"-"+rob+"%"+args[1]+"%"+args[2]
        elif name == "pick_action":
            a_other=name+"-"+rob+"%"+args[1]
        elif name == "place_action":
            a_other=name+"-"+rob+"%"+args[1]
        else: # open, close, nothing
            a_other=name+"-"+rob
            
        next_actions.append(a_other)
    
    
    return next_actions # for all the agents

# execution

def transit_execute(s, args):
    
    # simulation
    sim_env = s.sim_env 
    sim_env_cfg = s.sim_env_cfg
    robot_1_offset = s.robot_1_offset
    robot_2_offset = s.robot_2_offset
    mug_sm = s.mug_sm
        
    
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
    mug_sm = MugSm(
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
            
            
            desired_position_1 = torch.tensor([REGIONS_LOC[args[2]]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)  
            desired_position_2 = torch.tensor([REGIONS_LOC[args[2]]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)  
            
           
            if args[0] == ROBOTS[0]: # other agent is robot_1  

                actions = mug_sm.compute_transit(
                    torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1),
                    torch.cat([desired_position_1, desired_orientation], dim=-1),
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,8:15] = torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1)
                actions_buffer[:,15] = GripperState.CLOSE
                actions_buffer[:,:8] = actions
                
                
                actions = actions_buffer
                
            else: # other agent is robot_2

                actions = mug_sm.compute_transit(
                    torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1),
                    torch.cat([desired_position_2, desired_orientation], dim=-1),
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,:7] = torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1)
                actions_buffer[:,7] = GripperState.CLOSE
                actions_buffer[:,8:] = actions
                
                actions = actions_buffer
            
            
            print(mug_sm.sm_state) 
            print(mug_sm.sm_wait_time)      
            if mug_sm.sm_state == MugSmState.APPROACH_GOAL and mug_sm.sm_wait_time >= MugSmWaitTime.APPROACH_GOAL: # pretend
                mug_sm.sm_state = MugSmState.REST
                s.rob_regions[args[0]]=args[2]
                
                break
            else: 
                if mug_sm.sm_wait_time > MugSmWaitTime.TIMEOUT:
                    
                    break
                
    s.mug_sm = mug_sm
    return s
def transfer_execute(s, args):
    
    # simulation
    sim_env = s.sim_env 
    sim_env_cfg = s.sim_env_cfg
    robot_1_offset = s.robot_1_offset
    robot_2_offset = s.robot_2_offset
    mug_sm = s.mug_sm
        
    
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
    mug_sm = MugSm(
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

            
            desired_position_1 = torch.tensor([REGIONS_LOC[args[2]]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)  
            desired_position_2 = torch.tensor([REGIONS_LOC[args[2]]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)  
            

            if args[0] == ROBOTS[0]: # other agent is robot_1  
                # desired_position_1[:,1] += -0.05
                actions = mug_sm.compute_transfer(
                    torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1),
                    torch.cat([desired_position_1, desired_orientation], dim=-1),
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,8:15] = torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1)
                actions_buffer[:,15] = GripperState.CLOSE
                actions_buffer[:,:8] = actions
                
                
                actions = actions_buffer
                
            else: # other agent is robot_2
                # desired_position_2[:,1] += 0.05
                actions = mug_sm.compute_transfer(
                    torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1),
                    torch.cat([desired_position_2, desired_orientation], dim=-1),
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,:7] = torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1)
                actions_buffer[:,7] = GripperState.CLOSE
                actions_buffer[:,8:] = actions
                
                actions = actions_buffer
            
            
            
            print(mug_sm.sm_state) 
            print(mug_sm.sm_wait_time)        
            if mug_sm.sm_state == MugSmState.APPROACH_GOAL and mug_sm.sm_wait_time >= MugSmWaitTime.APPROACH_GOAL: # pretend
                mug_sm.sm_state = MugSmState.REST
                s.rob_regions[args[0]]=args[2]
                
                break
            else: 
                if mug_sm.sm_wait_time > MugSmWaitTime.TIMEOUT:
                    
                    break
                
    s.mug_sm = mug_sm
    return s

def pick_execute(s, args):
    
    # simulation
    sim_env = s.sim_env 
    sim_env_cfg = s.sim_env_cfg
    robot_1_offset = s.robot_1_offset
    robot_2_offset = s.robot_2_offset
    mug_sm = s.mug_sm
        
    
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
    mug_sm = MugSm(
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
            desired_position_1 = torch.tensor([PICK_POS]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)
            desired_position_2 = torch.tensor([PICK_POS]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)
            

            if args[0] == ROBOTS[0]: # other agent is robot_1  

                actions = mug_sm.compute_pick(
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

                actions = mug_sm.compute_pick(
                    torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1),
                    torch.cat([object_position_2, desired_orientation], dim=-1),
                    torch.cat([desired_position_2, desired_orientation], dim=-1),
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,:7] = torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1)
                actions_buffer[:,7] = GripperState.CLOSE
                actions_buffer[:,8:] = actions
                
                actions = actions_buffer
            
            
            print(mug_sm.sm_state)
            print(mug_sm.sm_wait_time)       
            if mug_sm.sm_state == MugSmState.LIFT_OBJECT and mug_sm.sm_wait_time >= MugSmWaitTime.LIFT_OBJECT:
                mug_sm.sm_state = MugSmState.REST
                s.holding[args[0]] = [args[1]]
                s.obj_regions[args[1]] = ""
                
                break
            else: 
                if mug_sm.sm_wait_time > MugSmWaitTime.TIMEOUT:
                    
                    break
                
    s.mug_sm = mug_sm
    return s
def place_execute(s, args):
    
    # simulation
    sim_env = s.sim_env 
    sim_env_cfg = s.sim_env_cfg
    robot_1_offset = s.robot_1_offset
    robot_2_offset = s.robot_2_offset
    mug_sm = s.mug_sm
    
    if args[2] == REGIONS[0]: # place in region_mug
        goal = torch.tensor([[0.4,0.0,1.0]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device) - sim_env.unwrapped.scene.env_origins 
        if args[0] == ROBOTS[0]:
            goal = goal - robot_1_offset
        else:
            goal = goal - robot_2_offset 
    else: # place in region_stable_mug
        
        if args[0] == ROBOTS[0]:
            goal = torch.tensor([0.4,-0.2,0.75]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device) - sim_env.unwrapped.scene.env_origins 
        else:
            goal = torch.tensor([0.4,0.2,0.75]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device) - sim_env.unwrapped.scene.env_origins 
      
 
      
    
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
    mug_sm = MugSm(
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

                actions = mug_sm.compute_place(
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

                actions = mug_sm.compute_place(
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
            print(mug_sm.sm_state)
            print(mug_sm.sm_wait_time)       
            if mug_sm.sm_state == MugSmState.LIFT_EE and mug_sm.sm_wait_time >= MugSmWaitTime.LIFT_EE:
                mug_sm.sm_state = MugSmState.REST
                s.holding[args[0]] = []
                s.obj_regions[args[1]] = args[2]
                
                break
            else: 
                if mug_sm.sm_wait_time > MugSmWaitTime.TIMEOUT:
                    
                    break
                
    s.mug_sm = mug_sm
    return s
def open_execute(s, args):
    
    # simulation
    sim_env = s.sim_env 
    sim_env_cfg = s.sim_env_cfg
    robot_1_offset = s.robot_1_offset
    robot_2_offset = s.robot_2_offset
    mug_sm = s.mug_sm
        
    
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
    
    desired_position = torch.tensor([REGIONS_LOC[REGIONS[1]]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)  
            
    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros((sim_env.unwrapped.num_envs, 4), device=sim_env.unwrapped.device)
    desired_orientation[:, 1] = 1.0
    # create state machine
    mug_sm = MugSm(
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
            
            
            # -- cabinet frame
            cabinet_frame_tf: FrameTransformer = sim_env.unwrapped.scene["cabinet_frame"]
            cabinet_position_1 = cabinet_frame_tf.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_1_offset
            cabinet_position_2 = cabinet_frame_tf.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_2_offset
            cabinet_orientation = cabinet_frame_tf.data.target_quat_w[..., 0, :].clone()

            
            

            if args[0] == ROBOTS[0]: # other agent is robot_1  

                actions = mug_sm.compute_open(
                    torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1),
                    torch.cat([cabinet_position_1, cabinet_orientation], dim=-1),
                    torch.cat([desired_position, desired_orientation], dim=-1),
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,8:15] = torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1)
                actions_buffer[:,15] = GripperState.CLOSE
                actions_buffer[:,:8] = actions
                
                
                actions = actions_buffer
                
            else: # other agent is robot_2

                actions = mug_sm.compute_open(
                    torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1),
                    torch.cat([cabinet_position_2, cabinet_orientation], dim=-1),
                    torch.cat([desired_position, desired_orientation], dim=-1),
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,:7] = torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1)
                actions_buffer[:,7] = GripperState.CLOSE
                actions_buffer[:,8:] = actions
                
                actions = actions_buffer
            
            
            # print statements => stable behavior??
            print(mug_sm.sm_state)
            print(mug_sm.sm_wait_time)         
            if mug_sm.sm_state == MugSmState.LIFT_EE and mug_sm.sm_wait_time >= MugSmWaitTime.LIFT_EE:
                mug_sm.sm_state = MugSmState.REST
                s.open_door = True
                
                break
            else: 
                if mug_sm.sm_wait_time > MugSmWaitTime.TIMEOUT:
                    
                    break
                
    s.mug_sm = mug_sm
    return s
def close_execute(s, args):
    
    # simulation
    sim_env = s.sim_env 
    sim_env_cfg = s.sim_env_cfg
    robot_1_offset = s.robot_1_offset
    robot_2_offset = s.robot_2_offset
    mug_sm = s.mug_sm
        
    
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
    
    desired_position = torch.tensor([REGIONS_LOC[REGIONS[1]]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)  
    
    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros((sim_env.unwrapped.num_envs, 4), device=sim_env.unwrapped.device)
    desired_orientation[:, 1] = 1.0
    # create state machine
    mug_sm = MugSm(
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
            
            
            # -- cabinet frame
            cabinet_frame_tf: FrameTransformer = sim_env.unwrapped.scene["cabinet_frame"]
            cabinet_position_1 = cabinet_frame_tf.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_1_offset
            cabinet_position_2 = cabinet_frame_tf.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_2_offset
            cabinet_orientation = cabinet_frame_tf.data.target_quat_w[..., 0, :].clone()


            if args[0] == ROBOTS[0]: # other agent is robot_1  

                actions = mug_sm.compute_close(
                    torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1),
                    torch.cat([cabinet_position_1, cabinet_orientation], dim=-1),
                    torch.cat([desired_position, desired_orientation], dim=-1),
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,8:15] = torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1)
                actions_buffer[:,15] = GripperState.CLOSE
                actions_buffer[:,:8] = actions
                
                
                actions = actions_buffer
                
            else: # other agent is robot_2

                actions = mug_sm.compute_close(
                    torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1),
                    torch.cat([cabinet_position_2, cabinet_orientation], dim=-1),
                    torch.cat([desired_position, desired_orientation], dim=-1),                    
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,:7] = torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1)
                actions_buffer[:,7] = GripperState.CLOSE
                actions_buffer[:,8:] = actions
                
                actions = actions_buffer
            
            
            # print statements => stable behavior??
            print(mug_sm.sm_state)
            print(mug_sm.sm_wait_time)         
            if mug_sm.sm_state == MugSmState.LIFT_EE and mug_sm.sm_wait_time >= MugSmWaitTime.LIFT_EE:
                mug_sm.sm_state = MugSmState.REST
                s.open_door = False
                
                break
            else: 
                if mug_sm.sm_wait_time > MugSmWaitTime.TIMEOUT:
                    
                    break
                
    s.mug_sm = mug_sm
    return s
    
def pretend_pick_execute(s, args):
    
    # simulation
    sim_env = s.sim_env 
    sim_env_cfg = s.sim_env_cfg
    robot_1_offset = s.robot_1_offset
    robot_2_offset = s.robot_2_offset
    mug_sm = s.mug_sm
        
    
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
    mug_sm = MugSm(
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

                actions = mug_sm.compute_pick(
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

                actions = mug_sm.compute_pick(
                    torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1),
                    torch.cat([object_position_2, desired_orientation], dim=-1),
                    torch.cat([desired_position_2, desired_orientation], dim=-1),                   
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,:7] = torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1)
                actions_buffer[:,7] = GripperState.CLOSE
                actions_buffer[:,8:] = actions
                
                actions = actions_buffer
            
            
                    
            if mug_sm.sm_state == MugSmState.APPROACH_ABOVE_OBJECT and mug_sm.sm_wait_time >= MugSmWaitTime.APPROACH_ABOVE_OBJECT: # pretend
                mug_sm.sm_state = MugSmState.REST
                
                
                break
            else: 
                if mug_sm.sm_wait_time > MugSmWaitTime.TIMEOUT:
                    
                    break
                
    s.mug_sm = mug_sm
    return s
def pretend_place_execute(s, args):
    
    # simulation
    sim_env = s.sim_env 
    sim_env_cfg = s.sim_env_cfg
    robot_1_offset = s.robot_1_offset
    robot_2_offset = s.robot_2_offset
    mug_sm = s.mug_sm
    
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
    mug_sm = MugSm(
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

                actions = mug_sm.compute_place(
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

                actions = mug_sm.compute_place(
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
            print(mug_sm.sm_state)
            print(mug_sm.sm_wait_time)       
            if mug_sm.sm_state == MugSmState.APPROACH_ABOVE_GOAL and mug_sm.sm_wait_time >= MugSmWaitTime.APPROACH_ABOVE_GOAL:
                mug_sm.sm_state = MugSmState.REST
                
                
                break
            else: 
                if mug_sm.sm_wait_time > MugSmWaitTime.TIMEOUT:
                    
                    break
                
    s.mug_sm = mug_sm
    return s
def pretend_open_execute(s, args):
    
    # simulation
    sim_env = s.sim_env 
    sim_env_cfg = s.sim_env_cfg
    robot_1_offset = s.robot_1_offset
    robot_2_offset = s.robot_2_offset
    mug_sm = s.mug_sm
        
    
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
    
    desired_position = torch.tensor([REGIONS_LOC[REGIONS[1]]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)  
            
    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros((sim_env.unwrapped.num_envs, 4), device=sim_env.unwrapped.device)
    desired_orientation[:, 1] = 1.0
    # create state machine
    mug_sm = MugSm(
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
            
            
            # -- cabinet frame
            cabinet_frame_tf: FrameTransformer = sim_env.unwrapped.scene["cabinet_frame"]
            cabinet_position_1 = cabinet_frame_tf.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_1_offset
            cabinet_position_2 = cabinet_frame_tf.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_2_offset
            cabinet_orientation = cabinet_frame_tf.data.target_quat_w[..., 0, :].clone()

            
            

            if args[0] == ROBOTS[0]: # other agent is robot_1  

                actions = mug_sm.compute_open(
                    torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1),
                    torch.cat([cabinet_position_1, cabinet_orientation], dim=-1),
                    torch.cat([desired_position, desired_orientation], dim=-1),
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,8:15] = torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1)
                actions_buffer[:,15] = GripperState.CLOSE
                actions_buffer[:,:8] = actions
                
                
                actions = actions_buffer
                
            else: # other agent is robot_2

                actions = mug_sm.compute_open(
                    torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1),
                    torch.cat([cabinet_position_2, cabinet_orientation], dim=-1),
                    torch.cat([desired_position, desired_orientation], dim=-1),
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,:7] = torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1)
                actions_buffer[:,7] = GripperState.CLOSE
                actions_buffer[:,8:] = actions
                
                actions = actions_buffer
            
            
            # print statements => stable behavior??
            print(mug_sm.sm_state)
            print(mug_sm.sm_wait_time)         
            if mug_sm.sm_state == MugSmState.APPROACH_HANDLE and mug_sm.sm_wait_time >= MugSmWaitTime.APPROACH_HANDLE:
                mug_sm.sm_state = MugSmState.REST
                
                break
            else: 
                if mug_sm.sm_wait_time > MugSmWaitTime.TIMEOUT:
                    
                    break
                
    s.mug_sm = mug_sm
    return s
def pretend_close_execute(s, args):
    
    # simulation
    sim_env = s.sim_env 
    sim_env_cfg = s.sim_env_cfg
    robot_1_offset = s.robot_1_offset
    robot_2_offset = s.robot_2_offset
    mug_sm = s.mug_sm
        
    
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
    
    desired_position = torch.tensor([REGIONS_LOC[REGIONS[1]]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)  
    
    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros((sim_env.unwrapped.num_envs, 4), device=sim_env.unwrapped.device)
    desired_orientation[:, 1] = 1.0
    # create state machine
    mug_sm = MugSm(
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
            
            
            # -- cabinet frame
            cabinet_frame_tf: FrameTransformer = sim_env.unwrapped.scene["cabinet_frame"]
            cabinet_position_1 = cabinet_frame_tf.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_1_offset
            cabinet_position_2 = cabinet_frame_tf.data.target_pos_w[..., 0, :].clone() - sim_env.unwrapped.scene.env_origins - robot_2_offset
            cabinet_orientation = cabinet_frame_tf.data.target_quat_w[..., 0, :].clone()


            if args[0] == ROBOTS[0]: # other agent is robot_1  

                actions = mug_sm.compute_close(
                    torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1),
                    torch.cat([cabinet_position_1, cabinet_orientation], dim=-1),
                    torch.cat([desired_position, desired_orientation], dim=-1),
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,8:15] = torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1)
                actions_buffer[:,15] = GripperState.CLOSE
                actions_buffer[:,:8] = actions
                
                
                actions = actions_buffer
                
            else: # other agent is robot_2

                actions = mug_sm.compute_close(
                    torch.cat([tcp_rest_position_2, tcp_rest_orientation_2], dim=-1),
                    torch.cat([cabinet_position_2, cabinet_orientation], dim=-1),
                    torch.cat([desired_position, desired_orientation], dim=-1),                    
                )
            
                actions_buffer = torch.zeros(sim_env.unwrapped.action_space.shape, device=sim_env.unwrapped.device)
                
                actions_buffer[:,:7] = torch.cat([tcp_rest_position_1, tcp_rest_orientation_1], dim=-1)
                actions_buffer[:,7] = GripperState.CLOSE
                actions_buffer[:,8:] = actions
                
                actions = actions_buffer
            
            
            # print statements => stable behavior??
            print(mug_sm.sm_state)
            print(mug_sm.sm_wait_time)         
            if mug_sm.sm_state == MugSmState.APPROACH_HANDLE and mug_sm.sm_wait_time >= MugSmWaitTime.APPROACH_HANDLE:
                mug_sm.sm_state = MugSmState.REST
                
                break
            else: 
                if mug_sm.sm_wait_time > MugSmWaitTime.TIMEOUT:
                    
                    break
                
    s.mug_sm = mug_sm
    return s
    

# other agents actions
def transit_transfer_other_execute_fn(a, b, s, store):
    
    rob_regions = b.rob_regions.copy()    
    a_other_name,a_ego_name = a.name.split("*")
    if EXEC == 0: # human
        
        print(a_other_name)
        print(a.args[:3])
            
        while True:
            print("True region of movement?")
            for i,reg in enumerate(REGIONS):
                print(str(i)+". "+reg)
            choice = input("Pick region")
            if int(choice)>=0 and int(choice)<len(REGIONS):
                break
            
        reg = REGIONS[int(choice)]
    
    elif EXEC == 1: # random
        
        if random.random()<0.9:
            reg = a.args[2]
        else:
            reg = random.choice(REGIONS)
    
        
    if EXEC ==0 or EXEC == 1: # human or random; no change to state for nominal!
        a.args[2] = reg
        if a_other_name == "transit_other":
            s = transit_execute(s,a.args[:3])
        else: 
            s = transfer_execute(s,a.args[:4])


    rob_regions[a.args[0]] = s.rob_regions[a.args[0]]
    
    return s, EnvObservation(rob_regions=rob_regions)
def transit_transfer_other_effects_fn(a, b, store):
    
    rob_regions = b.rob_regions.copy()
    
    if TRAIN == 0: # human
        
        a_other_name,a_ego_name = a.name.split("*")
        print(a_other_name)
        print(a.args[:3])
        while True:
            print("True region of movement?")
            for i,reg in enumerate(REGIONS):
                print(str(i)+". "+reg)
            choice = input("Pick region")
            if int(choice)>=0 and int(choice)<len(REGIONS):
                break
        reg = REGIONS[int(choice)]
        print(reg)
        
    elif TRAIN == 1: # random
        
        if DOOR_BIAS:
            reg = a.args[2]
        else:
            if random.random()<0.9:
                reg = a.args[2]
            else:
                reg = random.choice(REGIONS)
            
            
     
    rob_regions[a.args[0]] = reg
    
    return rob_regions


def pick_other_execute_fn(a, b, s, store):
    
    obj_regions = b.obj_regions.copy()
    holding = b.holding.copy()
    
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
    
    
        
    if EXEC ==0 or EXEC == 1: # human or random; no change to state for nominal!
        
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
        if placed: # simulate pick and check result
            
            s = place_execute(s, a.args[:3])
            
        else: 
            
            s = pretend_place_execute(s, a.args[:3])

    
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

def open_other_execute_fn(a, b, s, store):
    
    open_door = b.open_door
    
    if EXEC == 0: # human
        a_other_name,a_ego_name = a.name.split("*")
        print(a_other_name)
        print(a.args[:2])
        while True:
            print("Was open executed?")
            choice = input("0:No / 1:Yes")
            if int(choice)==0 or int(choice)==1:
                break
        opened = int(choice) == 1
    
    elif EXEC == 1: # random
        
        opened = random.random()<0.9
    
    
        
    if EXEC ==0 or EXEC == 1: # human or random; no change to state for nominal!
        if opened: # simulate pick and check result
            
            s = open_execute(s, a.args[0])
            
        else: 
            
            s = pretend_open_execute(s, a.args[0])

    if s.open_door:
        print("opened")
        open_door = True  
    else:
        print("not opened")  
    
    
    return s, EnvObservation(open_door=open_door)
def open_other_effects_fn(a, b, store):
    
    open_door = b.open_door
    
    if TRAIN == 0: # human
        a_other_name,a_ego_name = a.name.split("*")
        print(a_other_name)
        print(a.args[:2])
        while True:
            print("Was open executed?")
            choice = input("0:No / 1:Yes")
            if int(choice)==0 or int(choice)==1:
                break
        opened = int(choice)==1
        print(opened)
    elif TRAIN == 1: # random
        
        if DOOR_BIAS:
            opened=True
        else:
            opened = random.random()<=0.9
    
    if opened: 
        open_door = True
    
    return open_door

def close_other_execute_fn(a, b, s, store):

    open_door = b.open_door
    
    if EXEC == 0: # human
        a_other_name,a_ego_name = a.name.split("*")
        print(a_other_name)
        print(a.args[:2])
        while True:
            print("Was close executed?")
            choice = input("0:No / 1:Yes")
            if int(choice)==0 or int(choice)==1:
                break
        closed = int(choice) == 1
    
    elif EXEC == 1: # random
        
        closed = random.random()<0.9
    
    
        
    if EXEC == 0 or EXEC == 1: # human or random; no change to state for nominal!
        if closed: # simulate pick and check result
            
            s = close_execute(s, a.args[0])
            
        else: 
            
            s = pretend_close_execute(s, a.args[0])

    if not s.open_door:
        print("closed")
        open_door = False  
    else:
        print("not closed")  
    
    
    return s, EnvObservation(open_door=open_door)
def close_other_effects_fn(a, b, store):

    open_door = b.open_door
    
    if TRAIN == 0: # human
        a_other_name,a_ego_name = a.name.split("*")
        print(a_other_name)
        print(a.args[:2])
        while True:
            print("Was close executed?")
            choice = input("0:No / 1:Yes")
            if int(choice)==0 or int(choice)==1:
                break
        closed = int(choice)==1
        print(closed)
    elif TRAIN == 1: # random
        
        if DOOR_BIAS:
            closed=True
        else:
            closed = random.random()<=0.9
    
    if closed: 
        open_door = False
    
    return open_door

# joint actions
def joint_execute_fn(a, b, s, store):
    
    holding = b.holding.copy()
    open_door = b.open_door
    rob_regions = b.rob_regions.copy()
    obj_regions = b.obj_regions.copy()
    next_actions = b.next_actions.copy()
    
    a_other_name, a_ego_name = a.name.split("*")
        
    if a_other_name == "transfer_other":
        nargs = 4
    elif a_other_name == "open_other" or a_other_name == "close_other" or a_other_name == "nothing_other":
        nargs = 1
    else:
        nargs = 3
    
    args_ego = a.args[nargs:]
    
    # remove ego's previous action
    for na in s.next_actions: 
        name,args = na.split("-")
        args=args.split("%")
        if args[0] == args_ego[0]:
            s.next_actions.remove(na)
    

    # other agent
    if a_other_name == "transit_other" or a_other_name == "transfer_other":
        s,obs = transit_transfer_other_execute_fn(a, b, s, store)
    elif a_other_name == "pick_other":
        s,obs = pick_other_execute_fn(a, b, s, store)
    elif a_other_name == "place_other":
        s,obs = place_other_execute_fn(a, b, s, store)
    elif a_other_name == "open_other":
        s,obs = open_other_execute_fn(a, b, s, store)
    elif a_other_name == "close_other":
        s,obs = close_other_execute_fn(a, b, s, store)
    else:
        pass
    
    # ego agent
    if a_ego_name == "transit_ego" or a_ego_name == "transfer_ego":
    
        if random.random()<0.9:
            
            # rob_regions[args_ego[0]] = args_ego[2]
            if a_ego_name == "transit_ego":
                s=transit_execute(s,args_ego)
            else:
                s=transfer_execute(s,args_ego)
                        
        next_action = a_ego_name[:-3]+"action"+"-"+args_ego[0]+"%"+args_ego[2]
        if a_ego_name == "transfer_ego":
            next_action = a_ego_name[:-3]+"action"+"-"+args_ego[0]+"%"+ args_ego[3]+ "%"+args_ego[2]
            
    elif a_ego_name == "pick_ego":
        
        if (a.args[3] == REGIONS[0] and s.open_door) or a.args[3] != REGIONS[0]: # feasibility check
            if a.args[2] == REGIONS[0]: # cabinet
                pick_ego_sim = CABINET_PICK_EGO_SIM
            else:
                pick_ego_sim = SIMPLE_PICK_EGO_SIM
            if random.random()<pick_ego_sim: # 90% success
                s = pick_execute(s, args_ego)
            else:
                s = pretend_pick_execute(s, args_ego)
                
                
        next_action = a_ego_name[:-3]+"action"+"-"+args_ego[0]+"%"+args_ego[1]
    
    elif a_ego_name == "place_ego":
        
        if (a.args[3] == REGIONS[0] and s.open_door) or a.args[3] != REGIONS[0]: # feasibility check
            
            if random.random()<0.9: # 90% success
                s = place_execute(s, args_ego)
            else:
                s = pretend_place_execute(s, args_ego)
                
        next_action = a_ego_name[:-3]+"action"+"-"+args_ego[0]+"%"+args_ego[1]
        
    elif a_ego_name == "open_ego":
        
        if not s.open_door: # feasibility check
            if random.random()<OPEN_EGO:
                s = open_execute(s,args_ego)
            else:
                s = pretend_open_execute(s,args_ego)
        
        next_action = a_ego_name[:-3]+"action"+"-"+args_ego[0]
    
    elif a_ego_name == "close_ego":
        
        if s.open_door: # feasibility check
            if random.random()<CLOSE_EGO:
                s = close_execute(s,args_ego)
            else:
                s = pretend_close_execute(s,args_ego)
        
        next_action = a_ego_name[:-3]+"action"+"-"+args_ego[0]
                
        
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
     
    return s, EnvObservation(holding=s.holding,open_door=s.open_door,rob_regions=s.rob_regions,obj_regions=s.obj_regions,next_actions=next_actions)  
def joint_effects_fn(a, b, store):

    holding = b.holding.copy()
    open_door = b.open_door
    rob_regions = b.rob_regions.copy()
    obj_regions = b.obj_regions.copy()
    next_actions = b.next_actions.copy()
    
    a_other_name,a_ego_name = a.name.split("*")
    
    if a_other_name == "transfer_other":
        n_args = 4
    elif a_other_name == "open_other" or a_other_name == "close_other" or a_other_name == "nothing_other":
        n_args = 1
    else:
        n_args = 3
        
    
    args_ego = a.args[n_args:]
    
    # other agent
    if a_other_name == "transit_other" or a_other_name == "transfer_other":
        rob_regions = transit_transfer_other_effects_fn(a, b, store)
    elif a_other_name == "pick_other":
        obj_regions,holding = pick_other_effects_fn(a, b, store)
    elif a_other_name == "place_other":
        obj_regions,holding = place_other_effects_fn(a, b, store)
    elif a_other_name == "open_other":
        open_door = open_other_effects_fn(a, b, store)
    elif a_other_name == "close_other":
        open_door = close_other_effects_fn(a, b, store)
    else:
        pass 
    
    # resulting state
    o = EnvObservation(holding=holding,open_door=open_door,rob_regions=rob_regions,obj_regions=obj_regions,next_actions=next_actions)    
    b_temp=b.update(a,o,store)
    next_actions = get_next_actions_effects(a, b_temp, store) # get next actions from previous belief
    
    # ego agent 
    if a_ego_name == "transit_ego" or a_ego_name == "transfer_ego":
    
        if random.random()<0.9:

            rob_regions[args_ego[0]] = args_ego[2]
    
            
    elif a_ego_name == "pick_ego":
        
        if (a.args[3] == REGIONS[0] and open_door) or a.args[3] != REGIONS[0]: # feasibility check
            if a.args[2] == REGIONS[0]: # cabinet
                pick_ego_sim = CABINET_PICK_EGO_SIM
            else:
                pick_ego_sim = SIMPLE_PICK_EGO_SIM
            if random.random()<pick_ego_sim: # 90% success
                
                holding[args_ego[0]] = [args_ego[1]]
                obj_regions[args_ego[1]] = ""
                
    
    elif a_ego_name == "place_ego":
        
        if (a.args[3] == REGIONS[0] and open_door) or a.args[3] != REGIONS[0]: # feasibility check
            
            if random.random()<0.9: # 90% success
                
                holding[args_ego[0]] = []
                obj_regions[args_ego[1]] = args_ego[2]
                
        
    elif a_ego_name == "open_ego":
        
        if not open_door: # feasibility check
            if random.random()<OPEN_EGO:
                
                open_door = True
        
    
    elif a_ego_name == "close_ego":
        
        if open_door: # feasibility check
            if random.random()<CLOSE_EGO:
                
                open_door = False
                        
        
    else:
        
        pass
    
    
    o = EnvObservation(holding=holding,open_door=open_door,rob_regions=rob_regions,obj_regions=obj_regions,next_actions=next_actions)    
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
        store.set(DOOR, DOOR, "door")
        
        store.certified.append(Atom("stable",[MUG,REGIONS[0]]))
        store.certified.append(Atom("stable",[MUG,REGIONS[2]]))
        
        store.certified.append(Atom("is_ego",[ego]))

        holding = s.holding.copy()
        open_door = s.open_door
        rob_regions = s.rob_regions.copy()
        obj_regions = s.obj_regions.copy()
        next_actions = s.next_actions.copy()

        b = EnvBelief(holding=holding,open_door=open_door,rob_regions=rob_regions,obj_regions=obj_regions,
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
            Predicate("in_rob",["robot","region"]),
            Predicate("in_obj",["physical","region"]),
            Predicate("open",["door"]),
        ] 
        action_predicates = [Predicate("transit_action",["robot","region"]),Predicate("transfer_action",["robot","physical","region"]),Predicate("pick_action",["robot","physical"]),
                             Predicate("place_action",["robot","physical"]),Predicate("open_action",["robot"]),Predicate("close_action",["robot"]),Predicate("nothing_action",["robot"])]
        
        predicates += action_predicates
        
        possible_outcomes = [[Atom("transit_action",[rob,reg]) for reg in REGIONS]+[Atom("transfer_action",[rob,obj,reg]) for obj in OBJ_REGIONS.keys() for reg in REGIONS] +
                            [Atom("pick_action",[rob,obj]) for obj in OBJ_REGIONS.keys()] + [Atom("place_action",[rob,obj])for obj in OBJ_REGIONS.keys()] +
                            [Atom("open_action",[rob]),Atom("close_action",[rob]),Atom("nothing_action",[rob])] for rob in others]
        
       
        
        # modify preconditions, effects and execute functions for observation
        action_schemas_ego = [
            
            # ego-agent
            ActionSchema(
                name="pick_ego",
                inputs=["?rob1","?obj1","?reg1"],
                input_types=["robot","physical","region"],
                preconditions=[Atom("is_ego",["?rob1"]), # is the ego agent
                               Or([Not(Atom("in_rob",["?rob1",REGIONS[0]])),And([Atom("in_rob",["?rob1",REGIONS[0]]),Atom("open",[DOOR])])]), # TODO: modify!! accesibility of mug: derived predicate
                               Atom("in_obj",["?obj1","?reg1"]), # object is in region from where pick is attempted
                               Atom("in_rob",["?rob1","?reg1"]), # robot is in region from where pick is attempted
                               Atom("stable",["?obj1","?reg1"]),
                               Not(Exists(Atom("holding",["?rob1","?obj"]),["?obj"],["physical"])), # robot hand is free
                               ],
                verify_effects=[OneOf([Atom("holding",["?rob1","?obj1"]),Atom("in_obj",["?obj1","?reg1"])])], 

            ),
            
            
            ActionSchema(
                name="place_ego",
                inputs=["?rob1","?obj1","?reg1"],
                input_types=["robot","physical","region"],
                preconditions=[Atom("is_ego",["?rob1"]), # is the ego agent
                               Or([Not(Atom("in_rob",["?rob1",REGIONS[0]])),And([Atom("in_rob",["?rob1",REGIONS[0]]),Atom("open",[DOOR])])]), # TODO: modify!! accessibility of region
                               Atom("in_rob",["?rob1","?reg1"]), # robot is in region where place is attempted
                               Atom("holding",["?rob1","?obj1"]), # robot is holding the object that is to be placed 
                               Not(Atom("in_obj",["?obj1","?reg1"])),
                               Atom("stable",["?obj1","?reg1"]), # region where place is attempted is stable
                               ],
                verify_effects=[OneOf([Atom("holding",["?rob1","?obj1"]),Atom("in_obj",["?obj1","?reg1"])])],  
            ),
            

            ActionSchema(
                name="transit_ego",
                inputs=["?rob1","?reg1","?reg2"],
                input_types=["robot","region","region"],
                preconditions=[Atom("is_ego",["?rob1"]), # is the ego agent
                               Atom("in_rob",["?rob1","?reg1"]),
                               Not(Atom("in_rob",["?rob1","?reg2"])),
                               Not(Exists(Atom("holding",["?rob1","?obj"]),["?obj"],["physical"])), # robot hand is free
                               ],
                verify_effects=[OneOf([Atom("in_rob",["?rob1","?reg1"]),Atom("in_rob",["?rob1","?reg2"])])],
            ),
            ActionSchema(
                name="transfer_ego",
                inputs=["?rob1","?reg1","?reg2","?obj1"],
                input_types=["robot","region","region","physical"],
                preconditions=[Atom("is_ego",["?rob1"]), # is the ego agent
                               Atom("in_rob",["?rob1","?reg1"]),
                               Not(Atom("in_rob",["?rob1","?reg2"])),
                               Atom("holding",["?rob1","?obj1"])],
                verify_effects=[OneOf([Atom("in_rob",["?rob1","?reg1"]),Atom("in_rob",["?rob1","?reg2"])])],
            ),
            ActionSchema(
                name="open_ego",
                inputs=["?rob1"],
                input_types=["robot"],
                preconditions=[Atom("is_ego",["?rob1"]), # is the ego agent
                               Not(Atom("open",[DOOR])),
                               Atom("in_rob",["?rob1",REGIONS[1]]),
                               Not(Exists(Atom("holding",["?rob1","?obj"]),["?obj"],["physical"]))],
                verify_effects=[Atom("open",[DOOR])],
                
            ),
            ActionSchema(
                name="close_ego",
                inputs=["?rob1"],
                input_types=["robot"],
                preconditions=[Atom("is_ego",["?rob1"]), # is the ego agent
                               Atom("open",[DOOR]),
                               Atom("in_rob",["?rob1",REGIONS[1]]),
                               Not(Exists(Atom("holding",["?rob1","?obj"]),["?obj"],["physical"]))],
                verify_effects=[Not(Atom("open",[DOOR]))],
        
            ),
            
            ActionSchema(
                name="nothing_ego",
                inputs=["?rob1"],
                input_types=["robot"],
                preconditions=[Atom("is_ego",["?rob1"])],
                effects=[],
            ),
        ]
        
        action_schemas_other = [
            
            # other agents
            ActionSchema(
                name="pick_other",
                inputs=["?rob2","?obj2","?reg3"],
                input_types=["robot","physical","region"],
                preconditions=[Not(Atom("is_ego",["?rob2"])), # is not the ego agent
                               Atom("pick_action",["?rob2","?obj2"]), # other agents' turn
                               Or([Not(Atom("in_rob",["?rob2",REGIONS[0]])),And([Atom("in_rob",["?rob2",REGIONS[0]]),Atom("open",[DOOR])])]), # accesibility of mug: derived predicate
                               Atom("in_obj",["?obj2","?reg3"]), # object is in region from where pick is attempted
                               Atom("in_rob",["?rob2","?reg3"]), # robot is in region from where pick is attempted
                               Atom("stable",["?obj2","?reg3"]),
                               Not(Exists(Atom("holding",["?rob2","?obj"]),["?obj"],["physical"])), # robot hand is free
                               ],
                verify_effects=[OneOf([Atom("holding",["?rob2","?obj2"]),Atom("in_obj",["?obj2","?reg3"])])]+[OneOf(po) for po in possible_outcomes],
            ),
            
            
            ActionSchema(
                name="place_other",
                inputs=["?rob2","?obj2","?reg3"],
                input_types=["robot","physical","region"],
                preconditions=[Not(Atom("is_ego",["?rob2"])), # is not the ego agent
                               Atom("place_action",["?rob2","?obj2"]), # other agents' turn
                               Or([Not(Atom("in_rob",["?rob2",REGIONS[0]])),And([Atom("in_rob",["?rob2",REGIONS[0]]),Atom("open",[DOOR])])]), # accessibility of region
                               Not(Atom("in_obj",["?obj2","?reg3"])), # object is in region where place is attempted
                               Atom("in_rob",["?rob2","?reg3"]), # robot is in region where place is attempted
                               Atom("holding",["?rob2","?obj2"]), # robot is holding the object that is to be placed 
                               Not(Atom("in_obj",["?obj2","?reg3"])),
                               Atom("stable",["?obj2","?reg3"]), # region where place is attempted is stable
                               ],
                verify_effects=[OneOf([Atom("holding",["?rob2","?obj2"]),Atom("in_obj",["?obj2","?reg3"])])]+[OneOf(po) for po in possible_outcomes],
            ),
            
            ActionSchema(
                name="transit_other",
                inputs=["?rob2","?reg3","?reg4"],
                input_types=["robot","region","region"],
                preconditions=[Not(Atom("is_ego",["?rob2"])), # is not the ego agent
                               Atom("transit_action",["?rob2","?reg4"]), # other agents' turn
                               Atom("in_rob",["?rob2","?reg3"]),
                               Not(Atom("in_rob",["?rob2","?reg4"])),
                               Not(Exists(Atom("holding",["?rob2","?obj"]),["?obj"],["physical"])), # robot hand is free
                               ],
                verify_effects=[OneOf([Atom("in_rob",["?rob2",reg]) for reg in REGIONS])]+[OneOf(po) for po in possible_outcomes],
            ),
            
            ActionSchema(
                name="transfer_other",
                inputs=["?rob2","?reg3","?reg4","?obj2"],
                input_types=["robot","region","region","physical"],
                preconditions=[Not(Atom("is_ego",["?rob2"])), # is not the ego agent
                               Atom("transfer_action",["?rob2","?obj2","?reg4"]), # other agents' turn
                               Atom("in_rob",["?rob2","?reg3"]),
                               Not(Atom("in_rob",["?rob2","?reg4"])),
                               Atom("holding",["?rob2","?obj2"]),
                               ],
                verify_effects=[OneOf([Atom("in_rob",["?rob2",reg]) for reg in REGIONS])]+[OneOf(po) for po in possible_outcomes],
            ),
            ActionSchema(
                name="open_other",
                inputs=["?rob2"],
                input_types=["robot"],
                preconditions=[Not(Atom("is_ego",["?rob2"])), # is not the ego agent
                               Atom("open_action",["?rob2"]), # other agents' turn
                               Not(Atom("open",[DOOR])),
                               Atom("in_rob",["?rob2",REGIONS[1]]),
                               Not(Exists(Atom("holding",["?rob2","?obj"]),["?obj"],["physical"]))],
                verify_effects=[Atom("open",[DOOR])]+[OneOf(po) for po in possible_outcomes], # TODO: modify     
            ),
            ActionSchema(
                name="close_other",
                inputs=["?rob2"],
                input_types=["robot"],
                preconditions=[Not(Atom("is_ego",["?rob2"])), # is not the ego agent
                               Atom("close_action",["?rob2"]), # other agents' turn
                               Atom("open",[DOOR]),
                               Atom("in_rob",["?rob2",REGIONS[1]]),
                               Not(Exists(Atom("holding",["?rob2","?obj"]),["?obj"],["physical"]))],
                verify_effects=[Not(Atom("open",[DOOR]))]+[OneOf(po) for po in possible_outcomes], # TODO: modify
            ),
            ActionSchema(
                name="nothing_other",
                inputs=["?rob2"],
                input_types=["robot"],
                preconditions=[Not(Atom("is_ego",["?rob2"])),
                               Atom("nothing_action",["?rob2"])],
                verify_effects=[OneOf(po) for po in possible_outcomes],
            )
            
            
        ]
        
        
        
        action_schemas = []
        
        for as_other in action_schemas_other:
            
            as_other_name = as_other.name
            
            for as_ego in action_schemas_ego:
                
                as_ego_name = as_ego.name
                schema = ActionSchema()
                
                if (as_other_name == "transfer_other" and (as_ego_name == "transfer_ego" or as_ego_name == "pick_ego" or as_ego_name == "place_ego")) or \
                    (as_other_name == "pick_other" and (as_ego_name == "transfer_ego" or as_ego_name == "pick_ego" or as_ego_name == "place_ego")) or \
                        (as_other_name == "open_other" and as_ego_name == "open_ego") or (as_other_name == "close_other" and as_ego_name == "close_ego") or \
                            (as_other_name == "place_other" and (as_ego_name == "place_ego" or as_ego_name == "transfer_ego")): # not possible under beliefs
                    
                    continue
                
                # special cases
                # assumption: other agent acts before ego agent
                # assumption: pick is confusible with place in the sense nothing happens and vice versa
                # transit, transfer regions are confusible, nothing may happen (same region)
                # open, close are confusible with each other in the sense nothing happens
                # noop observation is deterministic
                
                # case 1: place, pick
                elif as_other_name == "place_other" and as_ego_name == "pick_ego":
                    
                    schema.name = as_other_name+"*"+as_ego_name
                    schema.inputs = as_other.inputs + as_ego.inputs
                    schema.input_types = as_other.input_types + as_ego.input_types
                    schema.preconditions = as_other.preconditions + [Atom("is_ego",["?rob1"]),Or([Not(Atom("in_rob",["?rob1",REGIONS[0]])),And([Atom("in_rob",["?rob1",REGIONS[0]]),Atom("open",[DOOR])])]), # TODO: modify!! accesibility of mug: derived predicate
                                                                     Atom("stable",["?obj1","?reg1"]),Atom("in_rob",["?rob1","?reg1"]), Not(Exists(Atom("holding",["?rob1","?obj"]),["?obj"],["physical"]))]
                    
                    schema.verify_effects = [OneOf([Atom("holding",["?rob1","?obj1"]),Atom("holding",["?rob2","?obj1"]),Atom("in_obj",["?obj1","?reg1"])])] + [OneOf(po) for po in possible_outcomes] # special UEff
                    
                # case 2: open, pick
                elif as_other_name == "open_other" and as_ego_name == "pick_ego":
                    
                    schema.name = as_other_name+"*"+as_ego_name
                    schema.inputs = as_other.inputs + as_ego.inputs
                    schema.input_types = as_other.input_types + as_ego.input_types
                    schema.preconditions = as_other.preconditions + [Atom("is_ego",["?rob1"]), Atom("in_obj",["?obj1","?reg1"]), Atom("in_rob",["?rob1","?reg1"]), Atom("stable",["?obj1","?reg1"]),
                                                                     Not(Exists(Atom("holding",["?rob1","?obj"]),["?obj"],["physical"]))] # Pre for open + special Pre
                    schema.verify_effects = as_other.verify_effects + as_ego.verify_effects # UEff cartesian product (feasibility learned through probabilities)
                    
                    
                # case 3: open, place
                elif as_other_name == "open_other" and as_ego_name == "place_ego":
                    
                    schema.name = as_other_name+"*"+as_ego_name
                    schema.inputs = as_other.inputs + as_ego.inputs
                    schema.input_types = as_other.input_types + as_ego.input_types
                    schema.preconditions = as_other.preconditions + [Atom("is_ego",["?rob1"]), Atom("holding",["?rob1","?obj1"]), Atom("in_rob",["?rob1","?reg1"]), 
                                                                     Atom("stable",["?obj1","?reg1"])] # Pre for open + special Pre
                    schema.verify_effects = as_other.verify_effects + as_ego.verify_effects
                    
                # case 4: open, close
                elif as_other_name == "open_other" and as_ego_name == "close_ego":
                    
                    schema.name = as_other_name+"*"+as_ego_name
                    schema.inputs = as_other.inputs + as_ego.inputs
                    schema.input_types = as_other.input_types + as_ego.input_types
                    schema.preconditions = as_other.preconditions + [Atom("is_ego",["?rob1"]),Atom("in_rob",["?rob1",REGIONS[1]]), Not(Exists(Atom("holding",["?rob1","?obj"]),["?obj"],["physical"]))] # Pre for open + special Pre
                    schema.verify_effects = [OneOf(po) for po in possible_outcomes] + as_ego.verify_effects # UEff not cartesian pdt
                    
                # case 5: close, pick
                elif as_other_name == "close_other" and as_ego_name == "pick_ego":
                    
                    schema.name = as_other_name+"*"+as_ego_name
                    schema.inputs = as_other.inputs + as_ego.inputs
                    schema.input_types = as_other.input_types + as_ego.input_types
                    schema.preconditions = as_other.preconditions + [Atom("is_ego",["?rob1"]), Not(Atom("in_obj",["?obj1",REGIONS[0]])), Atom("stable",["?obj1","?reg1"]),
                                                                     Not(Exists(Atom("holding",["?rob1","?obj"]),["?obj"],["physical"])), Atom("in_rob",["?rob1","?reg1"])] # special Pre
                    schema.verify_effects = as_other.verify_effects + as_ego.verify_effects 
                    
                # case 6: close, place
                elif as_other_name == "close_other" and as_ego_name == "place_ego":
                    
                    schema.name = as_other_name+"*"+as_ego_name
                    schema.inputs = as_other.inputs + as_ego.inputs
                    schema.input_types = as_other.input_types + as_ego.input_types
                    schema.preconditions = as_other.preconditions + [Atom("is_ego",["?rob1"]), Atom("holding",["?rob1","?obj1"]), Atom("in_rob",["?rob1","?reg1"]), 
                                                                     Not(Atom("in_rob",["?rob1",REGIONS[0]])), Atom("stable",["?obj1","?reg1"])] # special Pre
                    schema.verify_effects = as_other.verify_effects + as_ego.verify_effects
                    
                # case 7: close, open
                elif as_other_name == "close_other" and as_ego_name == "open_ego":
                    
                    schema.name = as_other_name+"*"+as_ego_name
                    schema.inputs = as_other.inputs + as_ego.inputs
                    schema.input_types = as_other.input_types + as_ego.input_types
                    schema.preconditions = as_other.preconditions + [Atom("is_ego",["?rob1"]),Atom("in_rob",["?rob1",REGIONS[1]]), Not(Exists(Atom("holding",["?rob1","?obj"]),["?obj"],["physical"]))] # Pre for open + special Pre
                    schema.verify_effects = [OneOf(po) for po in possible_outcomes] + as_ego.verify_effects
                
                # regular cases
                else: 
                    
                    schema.name = as_other_name+"*"+as_ego_name
                    schema.inputs = as_other.inputs + as_ego.inputs
                    schema.input_types = as_other.input_types + as_ego.input_types
                    schema.preconditions = as_other.preconditions + as_ego.preconditions
                    schema.verify_effects = as_other.verify_effects + as_ego.verify_effects  
                    
                schema.execute_fn = joint_execute_fn
                schema.effects_fn = joint_effects_fn

                action_schemas.append(schema)
        
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
    cfg["envelope_threshold"] = 10e-100
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
    cfg["num_skeletons"] = 100
    cfg['envelope_threshold'] = 10e-100
    
    cfg["flat_sample"] = False # TODO: check; may cause progressive widening
    cfg['save_dir'] = os.getcwd()+"/runs/run{}".format(time.time())
    
    log_dir = "/home/am/Videos/mug/"  
    
    # parse configuration
    sim_env_cfg: FrankaMultiMugEnvCfgik = parse_env_cfg(
        "Isaac-Lift-Cube-Franka-IK-Abs-multi-mug",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    sim_env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-multi-mug", cfg=sim_env_cfg,render_mode="rgb_array" if args_cli.video else None)
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
            cfg["batch_size"] = 10  
            cfg["num_samples"] = 50
        elif TRAIN == 1: # random
            cfg['batch_size'] = 1000
            cfg['num_samples'] = 1000
            if DOOR_BIAS:
                cfg['batch_size'] = 100
                cfg['num_samples'] = 6000 # 6000-10000 should work
                cfg['num_skeletons'] = 100
                
        elif TRAIN == 2:
            cfg["batch_size"] = 100  
            cfg["num_samples"] = 5000 
        
        

        # state
        s = EnvState(holding={ROBOTS[0]:[],ROBOTS[1]:[]},open_door=False,
                rob_regions={ROBOTS[0]:REGIONS[-1],ROBOTS[1]:REGIONS[-1]}, # short horizon
                obj_regions={MUG:REGIONS[0]},
                next_actions=["nothing_action-"+ROBOTS[0],"nothing_action-"+ROBOTS[1]])
        s.sim_env = sim_env
        s.sim_env_cfg = sim_env_cfg
        s.robot_1_offset = torch.tensor([[0.0, -0.20, 0.0]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)
        s.robot_2_offset = torch.tensor([[0.0, 0.20, 0.0]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)
        s.mug_sm = MugSm(sim_env_cfg.sim.dt * sim_env_cfg.decimation, sim_env.unwrapped.num_envs, sim_env.unwrapped.device, position_threshold=0.01)

        
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
                observation = env.step(action, b, store) # should call execute function
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

            next_actions = get_next_actions_execute(action,b,store)
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
        s = EnvState(holding={ROBOTS[0]:[],ROBOTS[1]:[]},open_door=False,
                rob_regions={ROBOTS[0]:REGIONS[-1],ROBOTS[1]:REGIONS[-1]}, # short horizon
                obj_regions={MUG:REGIONS[0]},
                next_actions=["nothing_action-"+ROBOTS[0],"nothing_action-"+ROBOTS[1]])
        s.sim_env = sim_env
        s.sim_env_cfg = sim_env_cfg
        s.robot_1_offset = torch.tensor([[0.0, -0.20, 0.0]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)
        s.robot_2_offset = torch.tensor([[0.0, 0.20, 0.0]]*sim_env.unwrapped.num_envs,device=sim_env.unwrapped.device)
        s.mug_sm = MugSm(sim_env_cfg.sim.dt * sim_env_cfg.decimation, sim_env.unwrapped.num_envs, sim_env.unwrapped.device, position_threshold=0.01)

        
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
            print(b1.abstract(store1))
            print(b2.abstract(store2))
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
