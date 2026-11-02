"""Transfer a human-only actor into a new assisted PPO policy and keep it fixed."""

from __future__ import annotations

import json
from pathlib import Path

import torch as th
from stable_baselines3.common.save_util import load_from_zip_file

from rl_train.train.policies.rl_agent_exo import HumanExoActorCriticPolicy
from rl_train.train.policies.rl_agent_human import HumanActorCriticPolicy


# These fields determine what each human actor input means. A same-sized vector
# is not sufficient evidence of compatibility.
_OBSERVATION_FIELDS = (
    "msk_key",
    "observation_joint_pos_keys",
    "observation_joint_vel_keys",
    "observation_joint_sensor_keys",
    "reference_observation_keys",
    "reference_observation_steps",
    "reference_motion_mode",
    "reference_data_path",
    "control_framerate",
    "lumbar_fixed_angle_convention",
    "lumbar_joint_fixed_angle",
    "joint_range_overrides",
)


def _validate_source_session(checkpoint: Path, target_env_params) -> None:
    session = checkpoint.parent.parent / "session_config.json"
    if not session.is_file():
        return
    source = json.loads(session.read_text(encoding="utf-8"))["env_params"]
    if source.get("device_key") != "NoDevice":
        raise ValueError(f"Human actor checkpoint must come from NoDevice training: {session}")
    for field in _OBSERVATION_FIELDS:
        if source.get(field) != getattr(target_env_params, field):
            raise ValueError(f"Human actor checkpoint has incompatible {field}: {session}")


def _check_human_input_and_output(policy: HumanExoActorCriticPolicy, n_human: int) -> None:
    handler = policy.policy_network.network_index_handler
    n_obs = policy.observation_space.shape[0]
    with th.no_grad():
        sample = th.arange(n_obs, device=policy.device, dtype=th.float32)[None, :]
        actual = handler.map_observation_to_network(sample, "human_actor")
        if actual.shape[1] != n_obs or not th.equal(actual, sample):
            raise ValueError("Assisted human actor must read the same observation vector in the same order")
        human_output = th.arange(n_human, device=policy.device, dtype=th.float32)[None, :] + 1
        actions = handler.map_network_to_action({"human_actor": human_output})
        if not th.equal(actions[:, :n_human], human_output) or th.count_nonzero(actions[:, n_human:]):
            raise ValueError("Assisted human actor must map exactly to the first human muscle actions")


def freeze_human_actor(policy: HumanExoActorCriticPolicy, n_human: int = 22) -> None:
    """Lock human mean and Gaussian noise; leave device actor and critic trainable.

    The diagonal Gaussian owns one log_std parameter for all actions. A gradient
    mask locks its human slice while permitting the device slice to learn.
    """
    if not isinstance(policy, HumanExoActorCriticPolicy):
        raise TypeError("Frozen human training requires HumanExoActorCriticPolicy")
    if policy.action_space.shape[0] <= n_human or policy.log_std.numel() != policy.action_space.shape[0]:
        raise ValueError("Frozen human training requires human and device action slots")
    for parameter in policy.policy_network.human_policy_net.parameters():
        parameter.requires_grad_(False)
    hook = getattr(policy, "_frozen_human_log_std_hook", None)
    if hook is not None:
        hook.remove()
    mask = th.ones_like(policy.log_std)
    mask[:n_human] = 0
    policy._frozen_human_log_std_hook = policy.log_std.register_hook(lambda grad: grad * mask)


def initialize_frozen_human_actor(model, checkpoint_path: str, target_env_params) -> None:
    """Copy only the human actor and human action noise from a human-only zip."""
    path = Path(checkpoint_path)
    _validate_source_session(path, target_env_params)
    source, params, _ = load_from_zip_file(path, device=model.device)
    if source is None or source.get("policy_class") is not HumanActorCriticPolicy:
        raise ValueError("Human actor checkpoint must contain HumanActorCriticPolicy")
    if not isinstance(model.policy, HumanExoActorCriticPolicy):
        raise TypeError("Target model must use HumanExoActorCriticPolicy")
    n_human = source["action_space"].shape[0]
    if n_human != 22 or source["observation_space"].shape != (76,) or model.observation_space.shape != (76,):
        raise ValueError("Human actor checkpoint must have 76 matching observations and 22 muscle actions")
    _check_human_input_and_output(model.policy, n_human)
    state = params["policy"]
    prefix = "policy_network.policy_net."
    actor_state = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    if not actor_state:
        raise ValueError("Human actor checkpoint has no human policy network")
    model.policy.policy_network.human_policy_net.load_state_dict(actor_state, strict=True)
    source_log_std = state["log_std"]
    if source_log_std.shape != (n_human,):
        raise ValueError("Human actor checkpoint has incompatible action noise")
    with th.no_grad():
        model.policy.log_std[:n_human].copy_(source_log_std)
        model.policy.log_std[n_human:].fill_(-2.0)
        # Starting from approximately zero assistance limits disturbance when
        # the pretrained human first encounters the attached device.
        exo_nets = [net for name, net in model.policy.policy_network.actor_nets.items() if name.startswith("exo_actor")]
        for net in {id(net): net for net in exo_nets}.values():
            last = next(layer for layer in reversed(net) if isinstance(layer, th.nn.Linear))
            last.weight.zero_()
            last.bias.zero_()
    freeze_human_actor(model.policy, n_human)
