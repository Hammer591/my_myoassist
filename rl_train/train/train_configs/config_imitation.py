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
        class ImitationReward:
            # Legacy mode keeps historical checkpoints/configurations reproducible.
            mode: str = "legacy"
            qpos_weight: float = 4.0
            qvel_weight: float = 1.0
            qpos_error_scales: dict = field(default_factory=dict)
            qvel_error_scales: dict = field(default_factory=dict)

            def validate(self, reward_weights):
                if self.mode not in {"legacy", "normalized"}:
                    raise ValueError("imitation_reward.mode must be 'legacy' or 'normalized'")
                if self.mode == "legacy":
                    return
                for kind in ("qpos", "qvel"):
                    weight = getattr(self, f"{kind}_weight")
                    if not math.isfinite(weight) or weight < 0:
                        raise ValueError(f"{kind}_weight must be finite and nonnegative")
                    joint_weights = getattr(reward_weights, f"{kind}_imitation_rewards")
                    scales = getattr(self, f"{kind}_error_scales")
                    for joint, joint_weight in joint_weights.items():
                        if not math.isfinite(joint_weight) or joint_weight < 0:
                            raise ValueError(f"{kind} joint weight for {joint} must be finite and nonnegative")
                        if joint_weight > 0 and (joint not in scales or not math.isfinite(scales[joint]) or scales[joint] <= 0):
                            raise ValueError(f"{kind}_error_scales requires a finite positive scale for {joint}")
                    if weight > 0 and not sum(joint_weights.values()) > 0:
                        raise ValueError(f"{kind} requires at least one positive joint weight")

        imitation_reward: ImitationReward = field(default_factory=ImitationReward)

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
        # Historical configs keep their original grid and velocity scaling.
        # fixed_rate uses exactly one control-period per reference frame, with
        # unscaled reference velocities for both rewards and initialization.
        reference_motion_mode: str = "legacy_scaled"

        reference_data_keys: list[str] = field(default_factory=list[str])

    env_params: EnvParams = field(default_factory=EnvParams)
