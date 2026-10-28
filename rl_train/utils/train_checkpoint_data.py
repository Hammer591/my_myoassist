from rl_train.utils.data_types import DictionableDataclass
from dataclasses import dataclass, field


@dataclass
class TrainCheckpointData(DictionableDataclass):
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    clip_range: float = 0.0
    clip_range_vf: float = 0.0
    entropy_loss: float = 0.0
    explained_variance: float = 0.0
    learning_rate: float = 0.0
    loss: float = 0.0
    n_updates: int = 0
    policy_gradient_loss: float = 0.0
    std: float = 0.0
    value_loss: float = 0.0
    mirror_loss: float | None = None
    mirror_loss_weighted: float | None = None
    mirror_loss_muscle: float | None = None
    mirror_loss_device: float | None = None
    mirror_device_grad_share: float | None = None
    num_timesteps: int = 0
    average_num_timestep: float = 0.0
    average_reward_per_episode: float = 0.0
    average_reward_dict_per_episode: dict = field(default_factory=dict)
    completed_episodes: int = 0
    reward_log_schema_version: int = 0  # Historical records have no verified component accounting.
    target_velocity_stats: dict = field(default_factory=dict)
    termination_counts: dict = field(default_factory=dict)
    time: str = ""
