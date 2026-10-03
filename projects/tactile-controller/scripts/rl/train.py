"""Script to train RL agent with RSL-RL."""

import getpass
import inspect
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import tyro
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder



@dataclass(frozen=True)
class TrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: RslRlBaseRunnerCfg
    registry_name: str | None = None
    video: bool = False
    video_length: int = 200
    video_interval: int = 2000
    enable_nan_guard: bool = False
    log_root: str = "logs/rsl_rl"
    """Root directory under which experiment logs are written."""
    torchrunx_log_dir: str | None = None
    wandb_run_path: str | None = None
    wandb_checkpoint_name: str | None = None
    """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

    @staticmethod
    def from_task(task_id: str) -> "TrainConfig":
        """Task defaults, with W&B checkpoint upload OFF.

        Inverts mjlab's `upload_model=True`, under which every `save_interval`
        pushes a ~20 MB .pt into W&B storage. Opt back in per run with
        `--agent.upload-model True` (what `submit.sh -e wandb` does).
        """
        env_cfg = load_env_cfg(task_id)
        agent_cfg = replace(load_rl_cfg(task_id), upload_model=False)
        return TrainConfig(env=env_cfg, agent=agent_cfg)


def snapshot_cfg_sources(cfg: TrainConfig, params_dir: Path,
                         task_id: str | None = None) -> None:
    """Snapshot the .py sources behind THIS run's cfg under params/cfg_src/.

    Task-agnostic: collects the modules of every term function / command /
    action class actually wired into the built cfg, then pulls in the
    declaration files (env_cfgs.py, rl_cfg.py, env_configs/*.py) of every
    involved task package. Two ways to find a task package: walking UP from a
    collected mdp file to a dir holding env_cfgs.py (works for per-task-mdp
    layouts), and — since the decoupled framework moved to a SHARED mdp
    (2026-08-07), which made every collected file live OUTSIDE the task dirs
    and silently dropped the declaration files from the snapshot — by finding
    the package whose __init__.py registers `task_id`. NB params/env.yaml and
    agent.yaml already record the built cfg VALUES — this preserves the code
    they came from.
    """
    rl_root = Path(__file__).resolve().parents[2] / "src" / "rl"
    files: set[Path] = set()

    def add(target: object) -> None:
        try:
            p = Path(inspect.getfile(target)).resolve()  # type: ignore[arg-type]
        except TypeError:  # builtins / non-introspectable
            return
        if rl_root in p.parents:
            files.add(p)

    for group_name in ("rewards", "terminations", "events", "curriculum", "metrics"):
        group = getattr(cfg.env, group_name, None) or {}
        for term in group.values():
            add(getattr(term, "func", None))
    for obs_group in cfg.env.observations.values():
        for obs_term in obs_group.terms.values():
            add(obs_term.func)
    for cmd in (cfg.env.commands or {}).values():
        add(type(cmd))
        wall_cfg = getattr(cmd, "wall_cfg", None)
        if wall_cfg is not None:
            add(type(wall_cfg))
    for act in (cfg.env.actions or {}).values():
        add(type(act))

    def add_task_pkg(pkg: Path) -> None:
        files.add(pkg / "env_cfgs.py")
        for extra in ("rl_cfg.py", "rl_cfgs.py", "__init__.py"):
            if (pkg / extra).exists():
                files.add(pkg / extra)
        files.update((pkg / "env_configs").glob("*.py"))

    # Declaration files of every task package involved: walk up from each
    # collected file to a dir holding env_cfgs.py (a task root, or a
    # single-task framework root like mpc_offline_tracking).
    for f in list(files):
        for parent in f.parents:
            if parent == rl_root:
                break
            if (parent / "env_cfgs.py").exists():
                add_task_pkg(parent)
                break

    # Shared-mdp layouts: locate the task package by its registration.
    if task_id is not None:
        for init in rl_root.rglob("__init__.py"):
            try:
                text = init.read_text()
            except OSError:
                continue
            if f'"{task_id}"' in text and (init.parent / "env_cfgs.py").exists():
                add_task_pkg(init.parent)

    for f in sorted(files):
        dest = params_dir / "cfg_src" / f.relative_to(rl_root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)


def run_identity_tags(task_id: str, experiment_name: str) -> list[str]:
    """W&B tags identifying who ran it and where, which task it trained, and
    which experiment it belongs to. Returns tag strings, does not apply them.

    The branch/commit/dirty tags are emitted only when the repo has a `.git`:
    cluster jobs import from a plain rsync mirror that has none, and
    scripts/skynet/submit.sh injects the submit-time git state into WANDB_TAGS
    for them instead. Never raises — a failure here drops tags, not the run.
    """
    def slug(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", value)

    tags = [f"task-{slug(task_id)}", f"exp-{slug(experiment_name)}"]
    if not os.environ.get("SLURM_JOB_ID"):
        # `<who>-local`. Omitted on a compute node, where the OS user is the
        # shared cluster account rather than whoever submitted:
        # scripts/skynet/submit.sh contributes `<who>-cluster` there instead.
        try:
            tags.append(f"{slug(os.environ.get('TACNAV_WHO') or getpass.getuser())}-local")
        except Exception:
            pass

    repo = Path(__file__).resolve().parents[2]
    if not (repo / ".git").exists():
        return tags
    try:
        def git(*args: str) -> str:
            out = subprocess.run(("git", "-C", str(repo), *args),
                                 capture_output=True, text=True, timeout=10)
            return out.stdout.strip()

        sha = git("rev-parse", "--short", "HEAD")
        if sha:
            tags += [
                f"branch-{slug(git('rev-parse', '--abbrev-ref', 'HEAD'))}",
                f"commit-{sha}",
                "dirty" if git("status", "--porcelain") else "clean",
            ]
    except Exception:
        pass
    return tags


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
        device = "cpu"
        seed = cfg.agent.seed
        rank = 0
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        # Set EGL device to match the CUDA device.
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
        device = f"cuda:{local_rank}"
        # Set seed to have diversity in different processes.
        seed = cfg.agent.seed + rank

    configure_torch_backends()

    cfg.agent.seed = seed
    cfg.env.seed = seed

    print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

    registry_name: str | None = None

    # Check if this is a tracking task by checking for motion command.
    is_tracking_task = "motion" in cfg.env.commands and isinstance(
        cfg.env.commands["motion"], MotionCommandCfg
    )

    if is_tracking_task:
        motion_cmd = cfg.env.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)

        # Check if motion_file is already set (e.g., via CLI --env.commands.motion.motion-file).
        if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
            print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")
        elif cfg.registry_name:
            # Download from WandB registry.
            registry_name = cast(str, cfg.registry_name)
            if ":" not in registry_name:
                registry_name = registry_name + ":latest"
            import wandb

            api = wandb.Api()
            artifact = api.artifact(registry_name)
            motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
        else:
            raise ValueError(
                "For tracking tasks, provide either:\n"
                "  --registry-name your-org/motions/motion-name (download from WandB)\n"
                "  --env.commands.motion.motion-file /path/to/motion.npz (local file)"
            )

    # Enable NaN guard if requested.
    if cfg.enable_nan_guard:
        cfg.env.sim.nan_guard.enabled = True
        print(
            f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}"
        )

    if rank == 0:
        print(f"[INFO] Logging experiment in directory: {log_dir}")

    # A custom runner may need a custom env / vec-env wrapper (the joint policy
    # needs the grouped-reward pair). Resolve it before the env is built.
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    env_cls = getattr(runner_cls, "env_cls", ManagerBasedRlEnv)
    wrapper_cls = getattr(runner_cls, "vec_wrapper_cls", RslRlVecEnvWrapper)

    env = env_cls(
        cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
    )

    log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

    resume_path: Path | None = None
    if cfg.agent.resume:
        if cfg.wandb_run_path is not None:
            # Load checkpoint from W&B.
            resume_path, was_cached = get_wandb_checkpoint_path(
                log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
            )
            if rank == 0:
                run_id = resume_path.parent.name
                checkpoint_name = resume_path.name
                cached_str = "cached" if was_cached else "downloaded"
                print(
                    f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
                    f"(run: {run_id}, {cached_str})"
                )
        else:
            # Load checkpoint from local filesystem.
            resume_path = get_checkpoint_path(
                log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
            )

    # Only record videos on rank 0 to avoid multiple workers writing to the same files.
    if cfg.video and rank == 0:
        env = VideoRecorder(
            env,
            video_folder=Path(log_dir) / "videos" / "train",
            step_trigger=lambda step: step % cfg.video_interval == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )
        print("[INFO] Recording videos during training.")

    env = wrapper_cls(env, clip_actions=cfg.agent.clip_actions)

    agent_cfg = asdict(cfg.agent)
    env_cfg = asdict(cfg.env)

    runner_kwargs = {}
    if is_tracking_task:
        runner_kwargs["registry_name"] = registry_name

    # Write config files before runner creation, since the runner mutates agent_cfg
    # in-place (e.g., injecting non-serializable objects).
    if rank == 0:
        dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
        dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)
        snapshot_cfg_sources(cfg, log_dir / "params", task_id=task_id)

    runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

    add_wandb_tags([*cfg.agent.wandb_tags,
                    *run_identity_tags(task_id, cfg.agent.experiment_name)])
    runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(str(resume_path))

    runner.learn(
        num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
    )

    env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
    args = args or TrainConfig.from_task(task_id)

    # Create log directory once before launching workers.
    log_root_path = (Path(args.log_root) / args.agent.experiment_name).resolve()
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.agent.run_name:
        log_dir_name += f"_{args.agent.run_name}"
    log_dir = log_root_path / log_dir_name

    # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
    selected_gpus, num_gpus = select_gpus(args.gpu_ids)

    # Set environment variables for all modes.
    if selected_gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))

    if num_gpus <= 1:
        # CPU or single GPU: run directly without torchrunx.
        run_train(task_id, args, log_dir)
    else:
        # Multi-GPU: use torchrunx.
        import torchrunx

        # torchrunx redirects stdout to logging.
        logging.basicConfig(level=logging.INFO)

        # Configure torchrunx logging directory.
        # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
        if "TORCHRUNX_LOG_DIR" not in os.environ:
            if args.torchrunx_log_dir is not None:
                # User specified a value via flag (could be "" to disable).
                os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
            else:
                # Default: put logs in training directory.
                os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

        print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
        torchrunx.Launcher(
            hostnames=["localhost"],
            workers_per_host=num_gpus,
            backend=None,  # Let rsl_rl handle process group initialization.
            copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
        ).run(run_train, task_id, args, log_dir)


def main():
    maybe_print_top_level_help("train")

    # Parse first argument to choose the task.
    # Import tasks to populate the registry.
    import mjlab.tasks

    import rl.mpc_offline_tracking as mpc_offline_tracking  # noqa: F401  (registers Unitree-G1-MPC-Tracking)

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    args = tyro.cli(
        TrainConfig,
        args=remaining_args,
        default=TrainConfig.from_task(chosen_task),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    del remaining_args

    launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
    main()
