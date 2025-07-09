# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# joint pos
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# ik-abs
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


from isaaclab_tasks.manager_based.manipulation.lift import mdp # modified


# other
import copy
##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the pick-clean-place scene with two robots and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robots and end-effectors frames
    """

    # robots: will be populated by agent env cfg
    robot_1: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame_1: FrameTransformerCfg = MISSING
    
    # robots: will be populated by agent env cfg
    robot_2: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame_2: FrameTransformerCfg = MISSING
    
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING
    
    # debris
    debris: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
                         scale=(2, 2, 2),),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    pass
    # object_pose_1 = mdp.UniformPoseCommandCfg(
    #     asset_name="robot_1",
    #     body_name=MISSING,  # will be set by agent env cfg
    #     resampling_time_range=(5.0, 5.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
    #     ),
    # )
    
    # object_pose_2 = mdp.UniformPoseCommandCfg(
    #     asset_name="robot_2",
    #     body_name=MISSING,  # will be set by agent env cfg
    #     resampling_time_range=(5.0, 5.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
    #     ),
    # )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action_1: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action_1: mdp.BinaryJointPositionActionCfg = MISSING
    
    # will be set by agent env cfg
    arm_action_2: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action_2: mdp.BinaryJointPositionActionCfg = MISSING



@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # robot_1
        joint_pos_1 = ObsTerm(func=mdp.joint_pos_rel, params = {"asset_cfg": SceneEntityCfg("robot_1")})
        joint_vel_1 = ObsTerm(func=mdp.joint_vel_rel, params = {"asset_cfg": SceneEntityCfg("robot_1")})
        object_position_1 = ObsTerm(func=mdp.object_position_in_robot_root_frame, params = {"robot_cfg": SceneEntityCfg("robot_1")}) 
        # target_object_position_1 = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose_1"})
        
        # robot_2
        joint_pos_2 = ObsTerm(func=mdp.joint_pos_rel, params = {"asset_cfg": SceneEntityCfg("robot_2")})
        joint_vel_2 = ObsTerm(func=mdp.joint_vel_rel, params = {"asset_cfg": SceneEntityCfg("robot_2")})
        object_position_2 = ObsTerm(func=mdp.object_position_in_robot_root_frame, params = {"robot_cfg": SceneEntityCfg("robot_2")}) 
        # target_object_position_2 = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose_2"})
        
        # the full action tensor
        actions = ObsTerm(func=mdp.last_action)
        
        

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.01, 0.01), "y": (-0.025, 0.025), "z": (0.0, 0.0)}, # modified!!
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    pass
    


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass
    


##
# Environment configuration
##


@configclass
class MultiCleanEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 120.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


# joint pos 

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause






@configclass
class FrankaMultiCleanEnvCfgjp(MultiCleanEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # robot_1
        # Set Franka as robot
        
        self.scene.robot_1 = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action_1 = mdp.JointPositionActionCfg(
            asset_name="robot_1", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action_1 = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_1",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        # self.commands.object_pose_1.body_name = "panda_hand"
        
        # robot_2
        # Set Franka as robot
        
        self.scene.robot_2 = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action_2 = mdp.JointPositionActionCfg(
            asset_name="robot_2", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action_2 = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_2",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        # self.commands.object_pose_2.body_name = "panda_hand"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        
        # Set Cube as object
        self.scene.debris = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Debris",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(1.5, 1.5, 0.2),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        # robot_1
        marker_cfg_1 = FRAME_MARKER_CFG.copy()
        marker_cfg_1.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg_1.prim_path = "/Visuals/FrameTransformer_1"
        self.scene.ee_frame_1 = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_1/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg_1,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_1/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )
        
        # robot_2
        marker_cfg_2 = FRAME_MARKER_CFG.copy()
        marker_cfg_2.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg_2.prim_path = "/Visuals/FrameTransformer_2"
        self.scene.ee_frame_2 = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_2/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg_2,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_2/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )


@configclass
class FrankaMultiCleanEnvCfgjp_PLAY(FrankaMultiCleanEnvCfgjp):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


# ik-abs

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause



##
# Rigid object lift environment.
##


@configclass
class FrankaMultiCleanEnvCfgik(FrankaMultiCleanEnvCfgjp):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        # robot_1
        init_state_1 = copy.deepcopy(FRANKA_PANDA_CFG.init_state)
        init_state_1.pos = [0.0, -0.20, 0.0]
        init_state_1.rot = [1.0, 0.0, 0.0, 0.0]
        self.scene.robot_1 = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1",
                                                              init_state=init_state_1)
        

        # Set actions for the specific robot type (franka)
        self.actions.arm_action_1 = DifferentialInverseKinematicsActionCfg(
            asset_name="robot_1",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
        
        # robot_2
        init_state_2 = copy.deepcopy(FRANKA_PANDA_CFG.init_state)
        init_state_2.pos = [0.0, 0.20, 0.0]
        init_state_2.rot = [1.0, 0.0, 0.0, 0.0]
        self.scene.robot_2 = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2",
                                                              init_state=init_state_2)

        # Set actions for the specific robot type (franka)
        self.actions.arm_action_2 = DifferentialInverseKinematicsActionCfg(
            asset_name="robot_2",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )


@configclass
class FrankaMultiCleanEnvCfgik_PLAY(FrankaMultiCleanEnvCfgik):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

