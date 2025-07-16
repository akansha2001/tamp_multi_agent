# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
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

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.00001, 0.00001, 0.00001)

# ik-abs
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

from isaaclab_tasks.manager_based.manipulation.cabinet import mdp # modified


# other
import copy
##
# Scene definition
##


@configclass
class CabinetMugSceneCfg(InteractiveSceneCfg):
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
    
    cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
            scale = (1.0,0.4,1.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.8, 0, 0.2),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )

    # Frame definitions for the cabinet.
    cabinet_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet/sektion",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/CabinetFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_top",
                name="drawer_handle_top",
                offset=OffsetCfg(
                    pos=(0.305, 0.0, 0.01),
                    rot=(0.5, 0.5, -0.5, -0.5),  # align with end-effector frame
                ),
            ),
        ],
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
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
        
        rel_ee_drawer_distance_1 = ObsTerm(func=mdp.rel_ee_drawer_distance, params={"ext":"_1"})

        # robot_2
        joint_pos_2 = ObsTerm(func=mdp.joint_pos_rel, params = {"asset_cfg": SceneEntityCfg("robot_2")})
        joint_vel_2 = ObsTerm(func=mdp.joint_vel_rel, params = {"asset_cfg": SceneEntityCfg("robot_2")})
        
        rel_ee_drawer_distance_2 = ObsTerm(func=mdp.rel_ee_drawer_distance, params={"ext":"_2"})

        cabinet_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
        )
        cabinet_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
        )
        
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
    robot_physics_material_1 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_1", body_names=".*"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.8, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )
    
    robot_physics_material_2 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_2", body_names=".*"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.8, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    cabinet_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cabinet", body_names="drawer_handle_top"),
            "static_friction_range": (1.0, 1.25),
            "dynamic_friction_range": (1.25, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints_1 = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_1"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    
    reset_robot_joints_2 = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_2"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
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



    


##
# Environment configuration
##


@configclass
class MultiMugEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: CabinetMugSceneCfg = CabinetMugSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 120.0
        
        self.viewer.eye = (-2.0, -2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        
        

# joint pos 

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause






@configclass
class FrankaMultiMugEnvCfgjp(MultiMugEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # robot_1
        # Set Franka as robot
        
        self.scene.robot_1 = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action_1 = mdp.JointPositionActionCfg(
            asset_name="robot_1",
            joint_names=["panda_joint.*"],
            scale=1.0,
            use_default_offset=True,
        )
        self.actions.gripper_action_1 = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_1",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        
        self.scene.ee_frame_1 = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_1/panda_link0",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer_1"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_1/panda_hand",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_1/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_1/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )

        
        # robot_2
        # Set Franka as robot
        
        self.scene.robot_2 = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action_2 = mdp.JointPositionActionCfg(
            asset_name="robot_2",
            joint_names=["panda_joint.*"],
            scale=1.0,
            use_default_offset=True,
        )
        self.actions.gripper_action_2 = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_2",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        
        self.scene.ee_frame_2 = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_2/panda_link0",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer_2"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_2/panda_hand",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_2/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_2/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
        

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.0, 0.5], rot=[1, 0, 0, 0]),
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
        
        

        
@configclass
class FrankaMultiMugEnvCfgjp_PLAY(FrankaMultiMugEnvCfgjp):
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
class FrankaMultiMugEnvCfgik(FrankaMultiMugEnvCfgjp):
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
class FrankaMultiMugEnvCfgik_PLAY(FrankaMultiMugEnvCfgik):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

