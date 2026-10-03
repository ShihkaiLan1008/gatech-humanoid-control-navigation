"""Task packages of the "mpc online tracking" framework.

One folder per task; each task package registers its own mjlab task id(s) in
its ``__init__``. Layout per task (repo-wide rl convention):

    tasks/<task>/
    ├── env_configs/   # ALL manager-term declarations — one file per manager
    ├── mdp/           # task-specific implementation (extra commands/rewards)
    ├── env_cfgs.py    # ASSEMBLY ONLY (scene/sim/viewer + play overrides)
    ├── rl_cfg.py      # PPO/runner config
    └── __init__.py    # task registration
"""
