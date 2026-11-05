from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from dataclasses import dataclass, field
import math


@dataclass
class ExoImitationTrainSessionConfig(ImitationTrainSessionConfig):
    coadaptation_provenance: dict = field(default_factory=dict)

    @dataclass
    class EnvParams(ImitationTrainSessionConfig.EnvParams):
        @dataclass
        class CoAdaptation:
            enabled: bool = False
            # New stage: transfer actors, start fresh optimizer and stage counters.
            init_path: str = ""
            # Explicit surrogate: the current model has no calibrated Uchida energetics.
            effort_model: str = "hip_activation_squared"
            hip_muscle_names: list[str] = field(
                default_factory=lambda: [
                    f"{stem}_{side}" for side in ("r", "l") for stem in ("hamstrings", "glutmax", "iliopsoas", "rectfem")
                ]
            )
            effort_reference: float = 0.25
            effort_weight: float = 1.0
            action_diff_weight: float = 0.2
            action_second_diff_weight: float = 0.2
            action_magnitude_weight: float = 0.01
            power_weight: float = 0.005
            negative_power_multiplier: float = 5.0
            human_reward_coupling: float = 0.02
            # Preserves the existing physical soft-limit penalty in the exo objective.
            torque_excess_weight: float = 2.0

            def validate(self):
                if not self.enabled:
                    return
                if self.effort_model != "hip_activation_squared":
                    raise ValueError("Only explicit hip_activation_squared surrogate is available; not Uchida metabolic power")
                if not math.isfinite(self.effort_reference) or self.effort_reference <= 0:
                    raise ValueError("coadaptation.effort_reference must be finite and positive")
                if not self.hip_muscle_names or len(set(self.hip_muscle_names)) != len(self.hip_muscle_names):
                    raise ValueError("coadaptation requires unique hip_muscle_names")
                for name in (
                    "effort_weight",
                    "action_diff_weight",
                    "action_second_diff_weight",
                    "action_magnitude_weight",
                    "power_weight",
                    "negative_power_multiplier",
                    "human_reward_coupling",
                    "torque_excess_weight",
                ):
                    value = getattr(self, name)
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f"coadaptation.{name} must be finite and nonnegative")

        coadaptation: CoAdaptation = field(default_factory=CoAdaptation)

    env_params: EnvParams = field(default_factory=EnvParams)

    @dataclass
    class PolicyParams(ImitationTrainSessionConfig.PolicyParams):
        """
        ActorCriticPolicy parameters:
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
            activation_fn: type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[dict[str, Any]] = None,
            share_features_extractor: bool = True,
            normalize_images: bool = True,
            optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[dict[str, Any]] = None,
        """

        # @dataclass
        # class CustomPolicyParams:
        #     reset_shared_net: bool = False
        #     reset_policy_net: bool = False
        #     reset_value_net: bool = False
        # custom_policy_params: CustomPolicyParams = field(default_factory=CustomPolicyParams)

        # This actually does nothing
        @dataclass
        class CustomPolicyParams(ImitationTrainSessionConfig.PolicyParams.CustomPolicyParams):
            frozen_human_structural_symmetry: bool = False
            human_symmetry_observation_permutation: list[int] = field(default_factory=list)
            human_symmetry_action_permutation: list[int] = field(default_factory=list)
            human_observation_indices: list[int] = field(default_factory=list[int])
            exo_observation_indices: list[int] = field(default_factory=list[int])
            human_action_size: int = 0
            exo_action_size: int = 0

        custom_policy_params: CustomPolicyParams = field(default_factory=CustomPolicyParams)

    policy_params: PolicyParams = field(default_factory=PolicyParams)
    # @dataclass
    # class EnvParams(ImitationTrainSessionConfig.EnvParams):
    #     pass
    # env_params: EnvParams = field(default_factory=EnvParams)


##############################################################################
