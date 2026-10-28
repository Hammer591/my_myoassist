from rl_train.train.train_configs.config import TrainSessionConfigBase
from dataclasses import dataclass, field
import math


@dataclass
class ImitationTrainSessionConfig(TrainSessionConfigBase):
    @dataclass
    class AutoRewardAdjustParams:
        learning_rate: float = 0.001

    auto_reward_adjust_params: AutoRewardAdjustParams = field(default_factory=AutoRewardAdjustParams)

    @dataclass
    class EnvParams(TrainSessionConfigBase.EnvParams):
        @dataclass
        class TrajectoryCurriculum:
            enabled: bool = False
            initial_threshold: float = 0.3
            anneal_timesteps: int = 5_000_000

            def threshold_at(self, elapsed_timesteps: int, final_threshold: float) -> float:
                if not self.enabled:
                    return final_threshold
                if (
                    not math.isfinite(self.initial_threshold)
                    or not math.isfinite(final_threshold)
                    or not 0 < final_threshold <= self.initial_threshold
                    or self.anneal_timesteps <= 0
                ):
                    raise ValueError("Trajectory curriculum requires initial >= final > 0 and positive anneal_timesteps")
                progress = min(1.0, max(0.0, elapsed_timesteps / self.anneal_timesteps))
                return self.initial_threshold + progress * (final_threshold - self.initial_threshold)

        trajectory_curriculum: TrajectoryCurriculum = field(default_factory=TrajectoryCurriculum)

        @dataclass
        class RewardWeights(TrainSessionConfigBase.EnvParams.RewardWeights):
            qpos_imitation_rewards: dict = field(default_factory=dict)
            qvel_imitation_rewards: dict = field(default_factory=dict)

            end_effector_imitation_reward: float = 0.3

        reward_keys_and_weights: RewardWeights = field(default_factory=RewardWeights)

        flag_random_ref_index: bool = False
        out_of_trajectory_threshold: float = 1
        reference_data_path: str = ""

        reference_data_keys: list[str] = field(default_factory=list[str])

    env_params: EnvParams = field(default_factory=EnvParams)
