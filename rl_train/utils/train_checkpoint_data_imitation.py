from dataclasses import dataclass
from rl_train.utils.train_checkpoint_data import TrainCheckpointData


@dataclass
class ImitationTrainCheckpointData(TrainCheckpointData):
    reward_accumulate: dict = None
    reward_weights: dict = None
    training_trajectory_threshold: float | None = None
    effective_reward_weights: dict = None
    imitation_reward_mode: str = "legacy"
    reference_motion_mode: str = "legacy_scaled"
