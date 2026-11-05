from rl_train.envs.myoassist_leg_imitation import MyoAssistLegImitation
import numpy as np


class MyoAssistLegImitationExo(MyoAssistLegImitation):
    def _setup(self, *, env_params, **kwargs):
        super()._setup(env_params=env_params, **kwargs)
        cfg = getattr(env_params, "coadaptation", None)
        self._coadapt = cfg if cfg is not None and cfg.enabled else None
        if self._coadapt is not None:
            cfg.validate()
            if env_params.joint_limit_penalty_mode != "substep_squared_torque":
                raise ValueError("Co-adaptation requires physics-substep cost integration")
            if env_params.freeze_human_actor or env_params.human_actor_init_path:
                raise ValueError("Co-adaptation requires an unfrozen assisted actor, not human-only initialization")
            ids = [self.sim.model.actuator(name).id for name in cfg.hip_muscle_names]
            if not set(ids).issubset(set(self._muscle_actuator_ids)):
                raise ValueError("Co-adaptation effort must select only muscle actuators")
            self._coadapt_act_adrs = np.array([self.sim.model.actuator_actadr[i] for i in ids])
            if np.any(self._coadapt_act_adrs < 0):
                raise ValueError("Hip muscles require activation states")
        self._coadapt_prev = self._coadapt_prev2 = None

    def reset(self, **kwargs):
        self._coadapt_prev = self._coadapt_prev2 = None
        return super().reset(**kwargs)

    def _accumulate_physics_costs(self, physics_dt, model_weight):
        super()._accumulate_physics_costs(physics_dt, model_weight)
        if getattr(self, "_coadapt", None) is None:
            return
        data = self.sim.data
        self._coadapt_effort_integral += physics_dt * float(np.mean(data.act[self._coadapt_act_adrs] ** 2))
        power = data.actuator_force[self._device_actuator_ids] * data.actuator_velocity[self._device_actuator_ids]
        self._coadapt_positive_work += physics_dt * float(np.maximum(power, 0).sum())
        self._coadapt_negative_work += physics_dt * float(np.maximum(-power, 0).sum())

    # override
    def step(self, a, **kwargs):
        self._coadapt_effort_integral = 0.0
        self._coadapt_positive_work = self._coadapt_negative_work = 0.0
        next_obs, reward, terminated, truncated, info = super().step(a, **kwargs)
        cfg = getattr(self, "_coadapt", None)
        if cfg is not None:
            action = np.clip(a, self.action_space.low, self.action_space.high)[self._device_actuator_ids]
            first = action - self._coadapt_prev if self._coadapt_prev is not None else np.zeros_like(action)
            second = (
                action - 2 * self._coadapt_prev + self._coadapt_prev2
                if self._coadapt_prev2 is not None
                else np.zeros_like(action)
            )
            self._coadapt_prev2 = self._coadapt_prev
            self._coadapt_prev = action.copy()
            # Retain gait/effort/limit rewards for the human, and move device-specific
            # costs to the device objective. A reference-boundary zero stays zero.
            human = float(reward)
            components = dict(info["rwd_dict"])
            if "reference_exhausted" not in info.get("termination_reasons", []):
                for key in ("exo_activation_penalty", "exo_action_diff_penalty", "exo_torque_excess_penalty"):
                    human -= float(self.rwd_keys_wt.get(key, 0) * components.get(key, 0))
            effort = self._coadapt_effort_integral / self.dt
            terms = {
                "effort_proxy": self.dt * cfg.effort_weight * max(1 - effort / cfg.effort_reference, -5),
                "action_smoothness": -self.dt
                * (cfg.action_diff_weight * float(first @ first) + cfg.action_second_diff_weight * float(second @ second)),
                "action_magnitude": -self.dt * cfg.action_magnitude_weight * float(action @ action),
                "power": cfg.power_weight
                * (self._coadapt_positive_work - cfg.negative_power_multiplier * self._coadapt_negative_work),
                "human_coupling": cfg.human_reward_coupling * human,
                "torque_excess": cfg.torque_excess_weight * float(components.get("exo_torque_excess_penalty", 0)),
            }
            exo = sum(terms.values())
            if not np.isfinite([human, exo, effort]).all():
                raise FloatingPointError("Nonfinite co-adaptation reward")
            info = dict(
                info,
                agent_rewards=np.array([human, exo], dtype=np.float32),
                coadaptation=dict(
                    terms,
                    hip_activation_squared=effort,
                    positive_work_J=self._coadapt_positive_work,
                    negative_work_J=self._coadapt_negative_work,
                ),
            )
            components["dense"] = human
            info["rwd_dict"] = components
            reward = human
        return (next_obs, reward, terminated, truncated, info)
