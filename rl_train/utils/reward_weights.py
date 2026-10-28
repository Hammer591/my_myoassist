"""Resolve outer reward multipliers identically for training, logs and plots."""

from rl_train.utils.data_types import DictionableDataclass


def effective_reward_weights(reward_weights, imitation_reward=None):
    weights = reward_weights if isinstance(reward_weights, dict) else DictionableDataclass.to_dict(reward_weights)
    result = {key: sum(value.values()) if isinstance(value, dict) else value for key, value in weights.items()}
    if imitation_reward is not None and imitation_reward.mode == "normalized":
        result["qpos_imitation_rewards"] = imitation_reward.qpos_weight
        result["qvel_imitation_rewards"] = imitation_reward.qvel_weight
    return result
