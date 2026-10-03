"""Commands: the foot_mpc EE + foothold terms verbatim, the MPC replaced by the
H-LIP reference term (declared AFTER ``foothold`` — update order)."""
import copy

from mjlab.managers.command_manager import CommandTermCfg

from rl.decoupled.mdp import FootHoldHLIPCommandCfg
from rl.decoupled.tasks.joint_foot_mpc.env_configs.command_cfgs import COMMANDS as _BASE
from rl.g1_assets.g1_constants import COM_KNEES_BENT_Z

HLIP_CLF = {
    # (q_pos, q_vel) per channel; CLF-RL's LIP task values for com_x / com_y
    "Q_weights": {"com:x": (25.0, 200.0), "com:y": (300.0, 50.0)},
    "R_weights": {"com:x": 0.1, "com:y": 0.1},
}
"""CARE weights of the CLF over the CoM proxy (x, y) position + velocity."""

COMMANDS: dict[str, CommandTermCfg] = {
    "left_ee_cmd": copy.deepcopy(_BASE["left_ee_cmd"]),
    "right_ee_cmd": copy.deepcopy(_BASE["right_ee_cmd"]),
    "foothold": copy.deepcopy(_BASE["foothold"]),
    "hlip": FootHoldHLIPCommandCfg(
        foothold_name="foothold",
        entity_name="robot",
        com_height=COM_KNEES_BENT_Z,
        proxy_offset_x=0.04,
        apa_time=0.30,
        clf=HLIP_CLF,
        debug_vis=False,  # play assembly turns the reference sphere on
    ),
}
