"""G1 continuation task: human-like gait with staged energy optimization."""

from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.core.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1Rewards, G1RoughEnvCfg
from isaaclab_tasks.core.velocity.velocity_env_cfg import CurriculumCfg

from . import rewards as energy_mdp
from .commands import EdgeReturnVelocityCommandCfg


@configclass
class EnergyHumanRewards(G1Rewards):
    # Mechanical power is the direct efficiency signal requested by the user.
    mechanical_power = RewTerm(
        func=energy_mdp.mechanical_power_l1,
        weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    lower_body_power = RewTerm(
        func=energy_mdp.lower_body_power_l1,
        weight=-0.04,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
            )
        },
    )


@configclass
class AdaptiveEnergyCurriculum(CurriculumCfg):
    # The supervisor gradually increases the energy objective. This avoids
    # destroying the useful walking behavior inherited from model_1999.pt.
    energy_stage_1 = CurrTerm(
        func=base_mdp.modify_reward_weight,
        params={"term_name": "mechanical_power", "weight": -0.10, "num_steps": 100_000},
    )
    energy_stage_2 = CurrTerm(
        func=base_mdp.modify_reward_weight,
        params={"term_name": "mechanical_power", "weight": -0.20, "num_steps": 400_000},
    )
    leg_energy_stage_1 = CurrTerm(
        func=base_mdp.modify_reward_weight,
        params={"term_name": "lower_body_power", "weight": -0.10, "num_steps": 200_000},
    )
    smooth_stage = CurrTerm(
        func=base_mdp.modify_reward_weight,
        params={"term_name": "action_rate_l2", "weight": -0.012, "num_steps": 300_000},
    )


@configclass
class EnergyHumanG1EnvCfg(G1FlatEnvCfg):
    rewards: EnergyHumanRewards = EnergyHumanRewards()
    curriculum: AdaptiveEnergyCurriculum = AdaptiveEnergyCurriculum()

    def __post_init__(self):
        super().__post_init__()

        # Human-like gait proxies: longer alternating single-foot stance,
        # upright torso, low slip, smooth motion, and restrained upper body.
        self.rewards.feet_air_time.weight = 1.1
        self.rewards.feet_air_time.params["threshold"] = 0.45
        self.rewards.feet_slide.weight = -0.18
        self.rewards.flat_orientation_l2.weight = -1.5
        self.rewards.action_rate_l2.weight = -0.007
        self.rewards.joint_deviation_arms.weight = -0.14
        self.rewards.joint_deviation_torso.weight = -0.14
        self.rewards.termination_penalty.weight = -250.0

        # Keep command tracking dominant so standing still cannot win by using
        # zero energy. Continue with realistic walking, turning, and side steps.
        self.rewards.track_lin_vel_xy_exp.weight = 2.5
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.25, 0.25)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)


@configclass
class EnergyHumanG1RunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_energy_human"
        self.max_iterations = 2000
        self.save_interval = 25


@configclass
class EnergySmartRoughG1EnvCfg(G1RoughEnvCfg):
    """Rough-terrain policy with height sensing and diverse motion commands."""

    rewards: EnergyHumanRewards = EnergyHumanRewards()
    curriculum: AdaptiveEnergyCurriculum = AdaptiveEnergyCurriculum()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_air_time.weight = 0.8
        self.rewards.feet_air_time.params["threshold"] = 0.45
        self.rewards.feet_slide.weight = -0.18
        self.rewards.flat_orientation_l2.weight = -1.5
        self.rewards.action_rate_l2.weight = -0.007
        self.rewards.joint_deviation_arms.weight = -0.14
        self.rewards.joint_deviation_torso.weight = -0.14
        self.rewards.termination_penalty.weight = -250.0
        self.rewards.track_lin_vel_xy_exp.weight = 2.5
        self.rewards.track_ang_vel_z_exp.weight = 1.2

        # Every environment receives independently sampled commands, preventing
        # the visualized robots from all walking in the same direction.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # Preserve the standard command settings but replace the generator with
        # one that sends robots home before they reach the edge of an 8 m tile.
        old_command = self.commands.base_velocity
        self.commands.base_velocity = EdgeReturnVelocityCommandCfg(
            asset_name=old_command.asset_name,
            resampling_time_range=old_command.resampling_time_range,
            rel_standing_envs=old_command.rel_standing_envs,
            rel_heading_envs=old_command.rel_heading_envs,
            heading_command=old_command.heading_command,
            heading_control_stiffness=old_command.heading_control_stiffness,
            debug_vis=old_command.debug_vis,
            ranges=EdgeReturnVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.4, 1.2),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-1.0, 1.0),
                heading=(-3.141592653589793, 3.141592653589793),
            ),
            return_trigger_distance=3.0,
            return_release_distance=1.0,
            return_speed=0.7,
            return_heading_gain=1.5,
        )


@configclass
class EnergySmartRoughG1RunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_energy_smart_rough"
        self.max_iterations = 2500
        self.save_interval = 25
