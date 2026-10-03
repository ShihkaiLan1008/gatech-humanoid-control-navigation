"""Visual runner for the foot_lip single-step pipeline (meshcat).

Builds one WalkPrimitive from CLI args, runs the full pipeline (ZMP staircase
-> CoM -> swing spline -> 4.70 multibody-ZMP refinement -> legs-only IK) and
plays it: translucent robot + CoM (red) / ZMP ref (blue) / swing foot (green).

    uv run python -m data_gen.foot_lip.vis_foothold --dx 0.15 --T 1.0
    uv run python -m data_gen.foot_lip.vis_foothold --stance right --dx -0.1 --dyaw 0.2
    uv run python -m data_gen.foot_lip.vis_foothold --lift -0.1 0.05 0.1
    uv run python -m data_gen.foot_lip.vis_foothold --orbit --dx 0.2 --lift -0.2 0 0   # H-LIP step, 0.5 s clock
"""

import argparse

import numpy as np

import assets
from data_gen.foot_lip.lipm import LIPM, WalkPrimitive

DT = 0.02  # 50 Hz


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stance", default="left", choices=["left", "right"])
    ap.add_argument("--dx", type=float, default=0.15, help="landing x [m]")
    ap.add_argument("--dy", type=float, default=0.0,
                    help="landing width offset from nominal [m], + = outward")
    ap.add_argument("--dyaw", type=float, default=0.0, help="landing yaw [rad]")
    ap.add_argument("--lift", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    metavar=("DX", "DY", "DYAW"),
                    help="initial swing-foot offset from its nominal spot")
    ap.add_argument("--T", type=float, default=None,
                    help="total duration with v3 fractions 0.25/0.45/0.30; "
                         "default: legacy 0.6/0.4/0.8 split")
    ap.add_argument("--apex", type=float, default=0.08, help="swing apex [m]")
    ap.add_argument("--no-refine", action="store_true",
                    help="skip the 4.70 multibody-ZMP refinement")
    ap.add_argument("--once", action="store_true", help="play once, no loop")
    ap.add_argument("--orbit", action="store_true",
                    help="steady-state H-LIP step on the FootHoldCommand clock "
                         "(step 0.5 s, lift_frac 0.25, z_orbit 0.757) instead of rest-to-rest")
    args = ap.parse_args()

    stance = f"{args.stance}_foot"
    swing = "right_foot" if args.stance == "left" else "left_foot"
    if args.T is not None:
        timing = dict(step_Tsupport=0.45 * args.T,
                      Tdsp_start=0.25 * args.T, Tdsp_end=0.30 * args.T)
    else:
        timing = dict(step_Tsupport=0.4)
    primitive = WalkPrimitive(
        stance_foot=stance,
        swing_foot=swing,
        swing_start_pose=np.eye(4),
        step_dx=args.dx,
        step_dy=args.dy,
        step_dyaw=args.dyaw,
        swing_height=args.apex,
        lift_dx=args.lift[0],
        lift_dy=args.lift[1],
        lift_dyaw=args.lift[2],
        **timing,
    )

    lipm = LIPM(assets.ASSET_DIR / assets.UNITREE_TACTILE_XML)
    lipm.setup_robot()

    if args.orbit:
        T_STEP, LIFT_FRAC, Z_ORBIT = 0.5, 0.25, 0.757
        primitive.step_Tsupport = (1.0 - LIFT_FRAC) * T_STEP
        primitive.Tdsp_start, primitive.Tdsp_end = LIFT_FRAC * T_STEP, 0.0
        com_result = lipm.generate_orbit_com_trajectory(
            primitive, Z_ORBIT, T_STEP, LIFT_FRAC, dt=DT)
        foot_traj = lipm.generate_swing_trajectory_clock(
            com_result, primitive, T_STEP, LIFT_FRAC)
        print("orbit:", {k: np.round(v, 4) for k, v in com_result["orbit"].items()})
    else:
        com_result = lipm.generate_com_trajectory(primitive, dt=DT)
        foot_traj = lipm.generate_swing_trajectory(com_result, primitive)
    if args.no_refine:
        qs = lipm.solve_ik(com_result, foot_traj)
    else:
        qs = lipm.refine_with_multibody_zmp(com_result, foot_traj, dt=DT,
                                            max_iters=12)
        print("zmp refine:", com_result["zmp_refine"])
    print("boundary shift [m]:", com_result["zmp_boundary_shift"].round(4))

    lipm.draw_trajectories(com_result, foot_traj, primitive)
    lipm.play_motion(qs, dt=DT, loop=not args.once)


if __name__ == "__main__":
    main()
