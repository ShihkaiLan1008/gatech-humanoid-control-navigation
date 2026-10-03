import copy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from rl.g1_assets.g1_constants import get_g1_robot_home_cfg

from .env_configs.action_cfgs import JOINT_POS_ACTIONS
from .env_configs.command_cfgs import COMMANDS, PLANNER_COMMANDS
from .env_configs.curriculum_cfgs import CURRICULUM
from .env_configs.event_cfgs import EVENTS, NON_CONTACT_GEOMS
from .env_configs.metrics_cfgs import METRICS
from .env_configs.observation_cfgs import OBSERVATIONS
from .env_configs.reward_cfgs import REWARDS
from .env_configs.termination_cfgs import TERMINATIONS


def make_sensors() -> tuple:
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_hip_pitch_link|right_hip_pitch_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    nonfoot_ground_cfg = ContactSensorCfg(
        name="nonfoot_ground_touch",
        primary=ContactMatch(mode="geom", entity="robot", pattern=NON_CONTACT_GEOMS),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    return (feet_ground_cfg, self_collision_cfg, nonfoot_ground_cfg)


def lower_body_env_cfg(play: bool = False,
                       planner: bool = False) -> ManagerBasedRlEnvCfg:
    d = copy.deepcopy(
        {
            "observations": OBSERVATIONS,
            "actions": JOINT_POS_ACTIONS,
            "commands": COMMANDS,
            "events": EVENTS,
            "rewards": REWARDS,
            "terminations": TERMINATIONS,
            "curriculum": {},   # disabled (mjlab needs a dict, not None)
            "metrics": METRICS,
        }
    )

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1,
            extent=2.0,
            entities={"robot": get_g1_robot_home_cfg()},
            sensors=make_sensors(),
        ),
        observations=d["observations"],
        actions=d["actions"],
        commands=d["commands"],
        events=d["events"],
        rewards=d["rewards"],
        terminations=d["terminations"],
        curriculum=d["curriculum"],
        metrics=d["metrics"],
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=None,
            njmax=300,
            contact_sensor_maxmatch=64,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                ccd_iterations=50,
            ),
        ),
        decimation=4,
        episode_length_s=10.0,
    )

    if play:
        cfg.commands["step_cmd"].debug_vis = True     # ghost + ref foot frames

    if planner:
        cfg.commands = copy.deepcopy(PLANNER_COMMANDS)
        cfg.commands["step_cmd"].debug_vis = True

    return cfg
