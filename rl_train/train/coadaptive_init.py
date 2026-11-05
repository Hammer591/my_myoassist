"""Explicit stage transfer and resume for cooperative PPO."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch as th
from stable_baselines3.common.save_util import load_from_zip_file

from rl_train.train.coadaptive_ppo import CoAdaptivePPO
from rl_train.train.frozen_human import _OBSERVATION_FIELDS
from rl_train.train.policies.rl_agent_coadaptive import CoAdaptiveActorCriticPolicy
from rl_train.train.policies.rl_agent_exo import HumanExoActorCriticPolicy
from rl_train.utils.data_types import DictionableDataclass


def build_coadaptive_model(config, env, trained_model_path=None):
    cfg = config.env_params.coadaptation
    cfg.validate()
    if config.env_params.freeze_human_actor or config.env_params.human_actor_init_path:
        raise ValueError("Co-adaptation requires freeze_human_actor=False and human_actor_init_path=null")
    if config.ppo_params.mirror_coef or config.ppo_params.use_sde:
        raise ValueError("Co-adaptation uses structural mirror actors and does not support mirror_coef or SDE")
    previous = config.env_params.prev_trained_policy_path
    if trained_model_path is None and previous and cfg.init_path:
        raise ValueError("Choose coadaptation.init_path for a new stage OR prev_trained_policy_path for resume")
    source = trained_model_path or previous or cfg.init_path
    if not source:
        raise ValueError("Co-adaptation requires an assisted checkpoint; cannot silently train from scratch")
    path = Path(source)
    if path.suffix != ".zip":
        path = Path(str(path) + ".zip")
    session = path.parent.parent / "session_config.json"
    if not session.is_file():
        raise ValueError("Co-adaptation requires the source session_config.json for semantic compatibility checks")
    source_config = json.loads(session.read_text())
    for field in (*_OBSERVATION_FIELDS, "device_key", "physics_sim_framerate"):
        if source_config["env_params"].get(field) != getattr(config.env_params, field):
            raise ValueError(f"Co-adaptation checkpoint has incompatible {field}")
    data, params, _ = load_from_zip_file(path, device=config.ppo_params.device)
    if data["policy_class"] not in (HumanExoActorCriticPolicy, CoAdaptiveActorCriticPolicy):
        raise ValueError("Co-adaptation source must be an assisted policy")
    if data["observation_space"] != env.observation_space or data["action_space"] != env.action_space:
        raise ValueError("Co-adaptation checkpoint spaces differ from environment")
    source_kwargs = deepcopy(data["policy_kwargs"])
    target_custom = DictionableDataclass.to_dict(config.policy_params.custom_policy_params)
    for key in (
        "net_arch",
        "net_indexing_info",
        "frozen_human_structural_symmetry",
        "human_symmetry_observation_permutation",
        "human_symmetry_action_permutation",
    ):
        if source_kwargs["custom_policy_params"].get(key, target_custom[key]) != target_custom[key]:
            raise ValueError(f"Co-adaptation checkpoint has incompatible policy {key}")
    kwargs = DictionableDataclass.to_dict(config.ppo_params)
    kwargs.pop("mirror_coef")
    if trained_model_path is not None or previous:
        if data["policy_class"] is not CoAdaptiveActorCriticPolicy:
            raise ValueError("Resume/evaluation expects a co-adaptive checkpoint; use init_path for frozen-stage transfer")
        model = CoAdaptivePPO.load(path, env=env, **({} if trained_model_path else kwargs))
    else:
        model = CoAdaptivePPO(CoAdaptiveActorCriticPolicy, env, policy_kwargs=source_kwargs, verbose=1, **kwargs)
        result = model.policy.load_state_dict(params["policy"], strict=False)
        allowed = {"human_log_std", "exo_log_std"}
        if result.unexpected_keys or any(k not in allowed and not k.startswith("exo_value_net.") for k in result.missing_keys):
            raise ValueError(f"Unexpected co-adaptation transfer state: {result}")
        with th.no_grad():
            model.policy.human_log_std.copy_(params["policy"].get("human_log_std", params["policy"]["log_std"][:22]))
            model.policy.exo_log_std.copy_(params["policy"].get("exo_log_std", params["policy"]["log_std"][22:]))
        model.coadaptation_provenance = dict(
            source=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_timesteps=data["num_timesteps"],
            transfer="all actors, action noise and human critic; fresh optimizer/stage counter; independent exo critic",
            effort_model=cfg.effort_model,
        )
    config.coadaptation_provenance = deepcopy(getattr(model, "coadaptation_provenance", {}))
    return model
