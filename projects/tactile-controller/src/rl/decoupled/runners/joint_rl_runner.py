"""On-policy runner for the decoupled ("joint") policy.

Architecture (ported from the user's steady_tray stack, which proved it out on
IsaacLab + rsl-rl 2.x): the joint policy is **N independent PPO stacks**, one
per body — N actors, N critics, N optimizers, N rollout storages, N adaptive-KL
learning-rate controllers. There is no shared loss, no shared entropy and no
gradient path between bodies; they couple only through the physics, through
`last_action` (which carries the full concatenated action), and through the
`shared` reward group that `RewardGroupManager` folds into every stream.

Nothing in rsl-rl needs forking to express this. rsl-rl 5.x already takes an
arbitrary observation-SET name per model, gives every model its own normalizer,
and lets `PPO` be constructed from injected models + storage. Each body's
storage is allocated over `obs.select(...)` of just that body's groups, so no
storage carries another body's observations.

Per body, per step:
    actor(obs_set) -> a_i   ->  cat in ACTION-TERM order -> env.step
    reward TensorDict[group_i] -> PPO_i.process_env_step -> its own GAE/update
"""

from __future__ import annotations

import copy
import os
import statistics
import time
from collections import deque

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import check_nan, resolve_callable, resolve_obs_groups
from rsl_rl.utils.logger import Logger
from tensordict import TensorDict

from rl.decoupled.runners.joint_vecenv_wrapper import JointRslRlVecEnvWrapper
from rl.envs.joint_manager_based_rl_env import JointManagerBasedRlEnv


class JointOnPolicyRunner(OnPolicyRunner):
    """Drives one stock `PPO` per body over a shared environment."""

    # Picked up by scripts/rl/{train,play}.py so the joint task builds the
    # grouped-reward env and the matching wrapper.
    env_cls = JointManagerBasedRlEnv
    vec_wrapper_cls = JointRslRlVecEnvWrapper

    env: JointRslRlVecEnvWrapper
    algs: dict[str, PPO]

    def __init__(
        self,
        env: JointRslRlVecEnvWrapper,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.env = env
        self.cfg = train_cfg
        self.device = device

        self._configure_multi_gpu()  # sets self.cfg["multi_gpu"]

        obs = self.env.get_observations()
        self.body_names: list[str] = list(self.cfg["body_names"])
        if not self.body_names:
            raise ValueError("`body_names` is empty — nothing to train.")
        body_cfgs = {name: self.cfg[name] for name in self.body_names}

        # Observation sets: validate every set against the env's groups.
        obs_sets: list[str] = []
        for body in body_cfgs.values():
            obs_sets += [body["actor_obs_set"], body["critic_obs_set"]]
        self.obs_groups = resolve_obs_groups(obs, self.cfg["obs_groups"], obs_sets)

        self._resolve_action_layout(body_cfgs)
        self._reward_groups = {n: b["reward_group"] for n, b in body_cfgs.items()}

        self.algs = {}
        self._actor_groups: dict[str, list[str]] = {}
        self._body_groups: dict[str, list[str]] = {}
        for name in self.body_names:
            self.algs[name] = self._build_body(name, body_cfgs[name], obs)

        # rsl-rl's Logger reads cfg["algorithm"]["rnd_cfg"] unconditionally; the
        # joint config has per-body algorithm blocks instead, so shim it.
        logger_cfg = {**self.cfg, "algorithm": {"rnd_cfg": None}}
        self.logger = Logger(
            log_dir=log_dir,
            cfg=logger_cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )
        self.current_learning_iteration = 0

    # Construction helpers.

    def _resolve_action_layout(self, body_cfgs: dict[str, dict]) -> None:
        """Order bodies by their action term's position in the ActionManager.

        The concatenated action must line up with the manager's slices, so the
        declaration order of the action terms — not the order of `body_names` —
        decides the concatenation order.
        """
        term_dims = self.env.action_term_dims
        claimed: dict[str, str] = {}
        for name, body in body_cfgs.items():
            term = body["action_term"]
            if term not in term_dims:
                raise KeyError(
                    f"Body '{name}' drives action term '{term}', which the env's "
                    f"ActionManager does not have. Available: {list(term_dims)}"
                )
            if term in claimed:
                raise ValueError(
                    f"Action term '{term}' is claimed by both '{claimed[term]}' "
                    f"and '{name}'."
                )
            claimed[term] = name
        unclaimed = [t for t in term_dims if t not in claimed]
        if unclaimed:
            raise ValueError(
                f"Action terms {unclaimed} are not driven by any body. Every term "
                "must belong to exactly one policy, or its slice of the action "
                "vector would stay at zero."
            )
        self._action_order = [claimed[term] for term in term_dims]
        self._action_dims = {claimed[term]: dim for term, dim in term_dims.items()}

    def _build_body(self, name: str, body_cfg: dict, obs: TensorDict) -> PPO:
        actor_set = body_cfg["actor_obs_set"]
        critic_set = body_cfg["critic_obs_set"]
        actor_groups = list(self.obs_groups[actor_set])
        critic_groups = list(self.obs_groups[critic_set])
        # Deduplicated union, order preserved — the storage allocates exactly
        # these groups and nothing else.
        groups = list(dict.fromkeys(actor_groups + critic_groups))
        self._actor_groups[name] = actor_groups
        self._body_groups[name] = groups
        sub_obs = obs.select(*groups)

        actor_cfg = self._prepare_model_cfg(body_cfg["actor"])
        critic_cfg = self._prepare_model_cfg(body_cfg["critic"])
        actor_class: type[MLPModel] = resolve_callable(actor_cfg.pop("class_name"))  # type: ignore[assignment]
        critic_class: type[MLPModel] = resolve_callable(critic_cfg.pop("class_name"))  # type: ignore[assignment]

        num_actions = self._action_dims[name]
        actor = actor_class(
            sub_obs, self.obs_groups, actor_set, num_actions, **actor_cfg
        ).to(self.device)
        critic = critic_class(sub_obs, self.obs_groups, critic_set, 1, **critic_cfg).to(
            self.device
        )
        print(f"[{name}] Actor Model: {actor}")
        print(f"[{name}] Critic Model: {critic}")

        storage = RolloutStorage(
            "rl",
            self.env.num_envs,
            self.cfg["num_steps_per_env"],
            sub_obs,
            [num_actions],
            self.device,
        )

        alg_cfg = copy.deepcopy(body_cfg["algorithm"])
        alg_class: type[PPO] = resolve_callable(alg_cfg.pop("class_name"))  # type: ignore[assignment]
        alg_cfg.pop("share_cnn_encoders", None)  # actor/critic are built apart
        alg = alg_class(
            actor,
            critic,
            storage,
            device=self.device,
            **alg_cfg,
            multi_gpu_cfg=self.cfg["multi_gpu"],
        )
        alg.compile(self.cfg.get("torch_compile_mode"))
        return alg

    @staticmethod
    def _prepare_model_cfg(model_cfg: dict) -> dict:
        """Drop the optional-but-None keys that `MLPModel` would choke on.

        Same normalization `MjlabOnPolicyRunner` applies to its single
        actor/critic pair. Deep-copied because `MLPModel` pops `class_name` out
        of `distribution_cfg` in place.
        """
        cfg = copy.deepcopy(model_cfg)
        for key in ("cnn_cfg", "distribution_cfg"):
            if cfg.get(key) is None:
                cfg.pop(key, None)
        if cfg.get("rnn_type") is None:
            for key in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                cfg.pop(key, None)
        return cfg

    # Training.

    def learn(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False
    ) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            for alg in self.algs.values():
                alg.broadcast_parameters()

        self.logger.init_logging_writer()

        # Per-body episodic returns (the Logger only tracks the total).
        body_reward_sum = {
            name: torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            for name in self.body_names
        }
        body_reward_buffer = {name: deque(maxlen=100) for name in self.body_names}

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    sub = self._split_obs(obs)
                    actions = {
                        name: self.algs[name].act(sub[name])
                        for name in self.body_names
                    }
                    action = torch.cat(
                        [actions[name] for name in self._action_order], dim=1
                    )

                    obs, rewards, dones, extras = self.env.step(
                        action.to(self.env.device)
                    )
                    total_reward = extras["reward_total"]
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, total_reward, dones)
                    obs = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    total_reward = total_reward.to(self.device)

                    sub = self._split_obs(obs)
                    for name in self.body_names:
                        reward = rewards[self._reward_groups[name]]
                        self.algs[name].process_env_step(
                            sub[name], reward, dones, extras
                        )
                        body_reward_sum[name] += reward

                    self.logger.process_env_step(total_reward, dones, extras)
                    done_ids = (dones > 0).nonzero(as_tuple=False)
                    for name in self.body_names:
                        body_reward_buffer[name].extend(
                            body_reward_sum[name][done_ids][:, 0].cpu().numpy().tolist()
                        )
                        body_reward_sum[name][done_ids] = 0

                stop = time.time()
                collect_time = stop - start
                start = stop

                sub = self._split_obs(obs)
                for name in self.body_names:
                    self.algs[name].compute_returns(sub[name])

            loss_dict: dict[str, float] = {}
            for name in self.body_names:
                for key, value in self.algs[name].update().items():
                    loss_dict[f"{name}/{key}"] = value

            learn_time = time.time() - start
            self.current_learning_iteration = it

            self._log_bodies(it, body_reward_buffer)
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self._mean_learning_rate(),
                action_std=self._concatenated_action_std(),
                rnd_weight=None,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

        if self.logger.writer is not None:
            self.save(
                os.path.join(
                    self.logger.log_dir,  # type: ignore[arg-type]
                    f"model_{self.current_learning_iteration}.pt",
                )
            )
            self.logger.stop_logging_writer()

    def _split_obs(self, obs: TensorDict) -> dict[str, TensorDict]:
        """Per-body views of the observation TensorDict (no copy)."""
        return {
            name: obs.select(*self._body_groups[name]) for name in self.body_names
        }

    def _mean_learning_rate(self) -> float:
        return float(
            sum(alg.learning_rate for alg in self.algs.values()) / len(self.algs)
        )

    def _concatenated_action_std(self) -> torch.Tensor:
        return torch.cat(
            [
                self.algs[name].get_policy().output_std.flatten()
                for name in self.body_names
            ]
        )

    def _log_bodies(self, it: int, body_reward_buffer: dict[str, deque]) -> None:
        """Per-body scalars. `Logger.log` only knows about the aggregate."""
        writer = self.logger.writer
        if writer is None:
            return
        for name in self.body_names:
            alg = self.algs[name]
            writer.add_scalar(f"Loss/learning_rate_{name}", alg.learning_rate, it)
            writer.add_scalar(
                f"Policy/mean_std_{name}",
                alg.get_policy().output_std.mean().item(),
                it,
            )
            if len(body_reward_buffer[name]) > 0:
                writer.add_scalar(
                    f"Train/mean_reward_{name}",
                    statistics.mean(body_reward_buffer[name]),
                    it,
                )

    def train_mode(self) -> None:
        for alg in self.algs.values():
            alg.train_mode()

    def eval_mode(self) -> None:
        for alg in self.algs.values():
            alg.eval_mode()

    # Checkpointing.

    def save(self, path: str, infos: dict | None = None) -> None:
        """Per-body keyed checkpoint: `<body>.actor_state_dict`, etc."""
        saved_dict: dict = {}
        for name, alg in self.algs.items():
            for key, value in alg.save().items():
                saved_dict[f"{name}.{key}"] = value
        saved_dict["body_names"] = self.body_names
        saved_dict["iter"] = self.current_learning_iteration
        env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
        saved_dict["infos"] = {**(infos or {}), "env_state": env_state}
        torch.save(saved_dict, path)
        if self.cfg["upload_model"]:
            self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
        saved_bodies = loaded_dict.get("body_names")
        if saved_bodies is not None and list(saved_bodies) != self.body_names:
            raise ValueError(
                f"Checkpoint was trained with bodies {list(saved_bodies)} but this "
                f"runner has {self.body_names}."
            )

        load_iteration = False
        for name, alg in self.algs.items():
            prefix = f"{name}."
            body_dict = {
                key[len(prefix) :]: value
                for key, value in loaded_dict.items()
                if key.startswith(prefix)
            }
            if not body_dict:
                raise KeyError(
                    f"Checkpoint {path} has no state for body '{name}'. Keys: "
                    f"{sorted(k for k in loaded_dict if isinstance(k, str))}"
                )
            load_iteration = alg.load(body_dict, load_cfg, strict) or load_iteration

        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        infos = loaded_dict["infos"]
        if infos and "env_state" in infos:
            self.env.unwrapped.common_step_counter = infos["env_state"][
                "common_step_counter"
            ]
        return infos

    # Inference / export.

    def get_inference_policy(self, device: str | None = None):  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return `f(obs_tensordict) -> action[num_envs, total_action_dim]`.

        Takes the whole observation TensorDict (what the viewers already hand a
        policy) and slices each body's actor groups out of it, so play.py's call
        site is unchanged.
        """
        self.eval_mode()
        actors = {}
        for name in self.body_names:
            policy = self.algs[name].get_policy()
            actors[name] = policy if device is None else policy.to(device)
        actor_groups = dict(self._actor_groups)
        order = list(self._action_order)

        def joint_policy(obs: TensorDict) -> torch.Tensor:
            return torch.cat(
                [actors[name](obs.select(*actor_groups[name])) for name in order],
                dim=1,
            )

        return joint_policy

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """One TorchScript file per body — the actors have different inputs."""
        os.makedirs(path, exist_ok=True)
        stem = filename.removesuffix(".pt")
        for name in self.body_names:
            jit_model = self.algs[name].get_policy().as_jit()
            jit_model.to("cpu")
            torch.jit.script(jit_model).save(os.path.join(path, f"{stem}_{name}.pt"))

    def export_policy_to_onnx(
        self, path: str, filename: str = "policy.onnx", verbose: bool = False
    ) -> None:
        """One ONNX file per body. `dynamo=False` for the same reason mjlab does."""
        os.makedirs(path, exist_ok=True)
        stem = filename.removesuffix(".onnx")
        for name in self.body_names:
            onnx_model = self.algs[name].get_policy().as_onnx(verbose=verbose)
            onnx_model.to("cpu")
            onnx_model.eval()
            torch.onnx.export(
                onnx_model,
                onnx_model.get_dummy_inputs(),  # type: ignore[operator]
                os.path.join(path, f"{stem}_{name}.onnx"),
                export_params=True,
                opset_version=18,
                verbose=verbose,
                input_names=onnx_model.input_names,  # type: ignore[arg-type]
                output_names=onnx_model.output_names,  # type: ignore[arg-type]
                dynamic_axes={},
                dynamo=False,
            )
