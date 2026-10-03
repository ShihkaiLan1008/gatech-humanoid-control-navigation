"""Observation declarations — Opt2Skill-style asymmetric actor-critic
(arXiv:2409.20514 Sec. III-B.2), adapted for the G1 wall task.

Actor  o_a = [s_ap ; s_ar]: noisy DEPLOYABLE proprioception + PARTIAL reference.
Critic o_c = [s_cp ; s_cr]: privileged sim state + FULL reference (superset of
the actor's command, noise-free).

Deliberate actor exclusions (the deployability rule):
- NO world/base pose and NO reference-anchor drift error (the old BeyondMimic
  motion_anchor_pos_b/ori_b): localization belongs to the upstream planner;
  the policy must never key on a world frame.
- NO measured base_lin_vel (needs an onboard estimator; the policy infers it
  from proprio history). Deviation from the paper, which feeds a noisy one.
- NO measured EE force (no tactile/wrist sensor on the hardware yet) — it
  stays critic-only; append it to s_ap (with noise + delay DR) when the
  tactile hardware lands.
- NO joint_vel_ref (per the paper's partial reference).

The task command is ``motion_ee_pose_ref``: the clip's WORLD EE pose
re-expressed in the robot's CURRENT anchor frame every step
(mdp.motion_ee_pose_b) — what a perception planner would command at
deployment; base drift moves the target in the policy's frame and the wall
stays world-fixed within the episode. Its Unoise is planner-error DR:
cm-level target uncertainty against a rigid wall forces compliant pressing
instead of a stiff memorized penetration depth.
"""

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from .. import mdp
from .command_cfgs import EE_BODY_NAMES

# 10 samples @ 50 Hz = 0.2 s window — Opt2Skill's best history ablation
# (10 samples at an effective 50 Hz; 5 measurably worse, 15 degrades).
PROPRIO_HISTORY_LENGTH = 10

ACTOR_TERMS: dict[str, ObservationTermCfg] = {
    # --- s_ap: noisy deployable proprioception -------------------------------
    "base_ang_vel": ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_ang_vel"},
        noise=Unoise(n_min=-0.2, n_max=0.2),
        history_length=PROPRIO_HISTORY_LENGTH,
    ),
    "projected_gravity_b": ObservationTermCfg(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
        history_length=PROPRIO_HISTORY_LENGTH,
    ),
    "joint_pos": ObservationTermCfg(
        func=mdp.joint_pos_rel,
        noise=Unoise(n_min=-0.01, n_max=0.01),
        params={"biased": True},
        history_length=PROPRIO_HISTORY_LENGTH,
    ),
    "joint_vel": ObservationTermCfg(
        func=mdp.joint_vel_rel,
        noise=Unoise(n_min=-0.5, n_max=0.5),
        history_length=PROPRIO_HISTORY_LENGTH,
    ),
    "actions": ObservationTermCfg(
        func=mdp.last_action, history_length=PROPRIO_HISTORY_LENGTH
    ),
    # --- s_ar: partial reference (== the deployment command interface) -------
    # Feedforward base velocities, rotated into the robot's own frame
    # (heading-invariant; the rotation uses the measured attitude, hence noise).
    "motion_base_lin_vel_ref": ObservationTermCfg(
        func=mdp.motion_anchor_lin_vel_b,
        params={"command_name": "motion"},
        noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "motion_base_ang_vel_ref": ObservationTermCfg(
        func=mdp.motion_anchor_ang_vel_b,
        params={"command_name": "motion"},
        noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    # "motion_ee_pose_ref": ObservationTermCfg(
    #     func=mdp.motion_ee_pose_b,
    #     params={"command_name": "motion", "body_names": EE_BODY_NAMES},
    #     noise=Unoise(n_min=-0.02, n_max=0.02),
    # ),
    "motion_joint_pos_ref": ObservationTermCfg(
        func=mdp.motion_joint_pos,
        params={"command_name": "motion"},
    ),
    "motion_joint_torque_ref": ObservationTermCfg(
        func=mdp.motion_joint_torque,
        params={"command_name": "motion"},
    ),
    # Force command + contact schedule (mask is authoritative: force 0 during
    # scheduled swing means deliberately airborne).
    "motion_hand_force_ref": ObservationTermCfg(
        func=mdp.motion_hand_force,
        params={"command_name": "motion"},
    ),
    "motion_hand_contact_ref": ObservationTermCfg(
        func=mdp.motion_hand_contact,
        params={"command_name": "motion"},
    ),
    "motion_feet_grf_ref": ObservationTermCfg(
        func=mdp.motion_feet_force,
        params={"command_name": "motion"},
    ),
    "motion_feet_contact_ref": ObservationTermCfg(
        func=mdp.motion_feet_contact,
        params={"command_name": "motion"},
    ),
}

CRITIC_TERMS: dict[str, ObservationTermCfg] = {
    # --- s_cp: privileged proprioception -------------------------------------
    "base_pos_w": ObservationTermCfg(
        func=mdp.robot_base_pos_w,
        history_length=PROPRIO_HISTORY_LENGTH,
    ),
    "base_ori_w": ObservationTermCfg(
        func=mdp.robot_base_ori_w,
        history_length=PROPRIO_HISTORY_LENGTH,
    ),
    "base_lin_vel": ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_lin_vel"},
        history_length=PROPRIO_HISTORY_LENGTH,
    ),
    "base_ang_vel": ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_ang_vel"},
        history_length=PROPRIO_HISTORY_LENGTH,
    ),
    "projected_gravity_b": ObservationTermCfg(
        func=mdp.projected_gravity,
        history_length=PROPRIO_HISTORY_LENGTH,
    ),
    "joint_pos": ObservationTermCfg(
        func=mdp.joint_pos_rel,
        params={"biased": True},
        history_length=PROPRIO_HISTORY_LENGTH,
    ),
    "joint_vel": ObservationTermCfg(
        func=mdp.joint_vel_rel,
        history_length=PROPRIO_HISTORY_LENGTH,
    ),
    "joint_torque": ObservationTermCfg(
        func=mdp.robot_joint_torque,
        history_length=PROPRIO_HISTORY_LENGTH,
    ),
    "actions": ObservationTermCfg(
        func=mdp.last_action, history_length=PROPRIO_HISTORY_LENGTH
    ),
    # Measured EE pose (FK) in the robot frame + measured contact forces —
    # the privileged signals the actor is denied.
    "ee_pose_b": ObservationTermCfg(
        func=mdp.robot_ee_pose_b,
        params={"command_name": "motion", "body_names": EE_BODY_NAMES},
        history_length=PROPRIO_HISTORY_LENGTH,
    ),
    "hand_force_meas": ObservationTermCfg(
        func=mdp.robot_hand_force,
        params={"command_name": "motion", "sensor_name": "hand_wall"},
        history_length=PROPRIO_HISTORY_LENGTH,
    ),
    "feet_grf_meas": ObservationTermCfg(
        func=mdp.robot_feet_grf,
        params={"command_name": "motion", "sensor_name": "feet_ground"},
        history_length=PROPRIO_HISTORY_LENGTH,
    ),
    # --- s_cr: FULL reference -------------------------------------------------
    # Reference base pose, expressed relative to the robot (= base_pos_ref /
    # base_ori_ref; better conditioned than raw world coords, critic-only).
    "motion_anchor_pos_b": ObservationTermCfg(
        func=mdp.motion_anchor_pos_b,
        params={"command_name": "motion"},
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
        func=mdp.motion_anchor_ori_b,
        params={"command_name": "motion"},
    ),
    "motion_base_lin_vel_ref": ObservationTermCfg(
        func=mdp.motion_anchor_lin_vel_b,
        params={"command_name": "motion"},
    ),
    "motion_base_ang_vel_ref": ObservationTermCfg(
        func=mdp.motion_anchor_ang_vel_b,
        params={"command_name": "motion"},
    ),
    # The actor's command, noise-free (critic obs ⊇ actor obs).
    "motion_ee_pose_ref": ObservationTermCfg(
        func=mdp.motion_ee_pose_b,
        params={"command_name": "motion", "body_names": EE_BODY_NAMES},
    ),
    "motion_joint_pos_ref": ObservationTermCfg(
        func=mdp.motion_joint_pos,
        params={"command_name": "motion"},
    ),
    "motion_joint_vel_ref": ObservationTermCfg(
        func=mdp.motion_joint_vel,
        params={"command_name": "motion"},
    ),
    "motion_joint_torque_ref": ObservationTermCfg(
        func=mdp.motion_joint_torque,
        params={"command_name": "motion"},
    ),
    "motion_hand_force_ref": ObservationTermCfg(
        func=mdp.motion_hand_force,
        params={"command_name": "motion"},
    ),
    "motion_hand_contact_ref": ObservationTermCfg(
        func=mdp.motion_hand_contact,
        params={"command_name": "motion"},
    ),
    "motion_feet_grf_ref": ObservationTermCfg(
        func=mdp.motion_feet_force,
        params={"command_name": "motion"},
    ),
    "motion_feet_contact_ref": ObservationTermCfg(
        func=mdp.motion_feet_contact,
        params={"command_name": "motion"},
    ),
}

OBSERVATIONS: dict[str, ObservationGroupCfg] = {
    "actor": ObservationGroupCfg(
        terms=ACTOR_TERMS,
        concatenate_terms=True,
        enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
        terms=CRITIC_TERMS,
        concatenate_terms=True,
        enable_corruption=False,
    ),
}
