"""Episode parameter sampling and causal sensor/motor models, independent of reward truth."""

from collections import deque

import numpy as np


class Robustness:
    def __init__(self, env, config):
        self.env, self.config = env, config
        self.active = config.enabled and (not env.is_evaluate_mode or config.enable_in_evaluation)
        self.sample = {}
        if not self.active:
            return
        ranges = (
            "human_mass_scale", "human_inertia_scale", "muscle_strength_scale", "motor_gain_scale",
            "observation_delay_steps", "motor_delay_seconds", "motor_time_constant_seconds",
        )
        for name in ranges:
            bounds = np.asarray(getattr(config, name + "_range"), dtype=float)
            positive = name.endswith("scale")
            if (bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]
                    or bounds[0] < 0 or (positive and bounds[0] == 0)):
                raise ValueError(f"Invalid robustness.{name}_range")
            if name == "observation_delay_steps" and np.any(bounds != np.floor(bounds)):
                raise ValueError("Observation delay bounds must be integer control steps")
        self.randomize_inertial = any(
            list(getattr(config, name + "_range")) != [1.0, 1.0]
            for name in ("human_mass_scale", "human_inertia_scale")
        )
        if not config.human_body_names and self.randomize_inertial:
            raise ValueError("Human mass/inertia randomization requires explicit human_body_names")

        model = env.sim.model
        self.body_ids = np.asarray([model.body(name).id for name in config.human_body_names], dtype=int)
        if len(set(self.body_ids)) != len(self.body_ids) or np.any(self.body_ids == 0):
            raise ValueError("Human body names must be unique and exclude world")
        import mujoco

        self.muscle_ids = np.flatnonzero(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)
        self.nominal = {key: getattr(model, key).copy() for key in (
            "body_mass", "body_inertia", "actuator_gainprm", "actuator_biasprm",
        )}
        self.indices, noise, bias = [], [], []
        offset = 0
        groups = env.get_obs_dict(env.sim)
        allowed = {"qpos", "qvel", "sensor"}
        for values in (config.observation_noise_std, config.observation_bias_std):
            if set(values) - allowed:
                raise ValueError("Observation noise/bias supports qpos, qvel and sensor only")
        for key in env.obs_keys:
            size = np.asarray(groups[key]).size
            if key in allowed:
                self.indices.extend(range(offset, offset + size))
                for dest, values in ((noise, config.observation_noise_std), (bias, config.observation_bias_std)):
                    std = np.broadcast_to(np.asarray(values.get(key, 0.0), dtype=float), (size,))
                    if np.any(~np.isfinite(std)) or np.any(std < 0):
                        raise ValueError(f"Invalid observation standard deviation for {key}")
                    dest.extend(std)
            offset += size
        self.indices = np.asarray(self.indices, dtype=int)
        self.noise_std, self.bias_std = np.asarray(noise), np.asarray(bias)

    def reset(self):
        if not self.active:
            return
        env, cfg = self.env, self.config
        rng = env.np_random

        def draw(name):
            return float(rng.uniform(*getattr(cfg, name + "_range")))

        self.sample = {name: draw(name) for name in (
            "human_mass_scale", "human_inertia_scale", "muscle_strength_scale", "motor_delay_seconds",
            "motor_time_constant_seconds", "motor_gain_scale",
        )}
        lo, hi = map(int, cfg.observation_delay_steps_range)
        self.delay = int(rng.integers(lo, hi + 1))
        self.sample["observation_delay_steps"] = self.delay
        self.bias = rng.normal(size=len(self.indices)) * self.bias_std
        self.sample["observation_bias"] = self.bias.tolist()
        self.observations = deque(maxlen=self.delay + 1)
        physics_dt = float(env.sim.model.opt.timestep)
        self.motor_delay = int(np.floor(self.sample["motor_delay_seconds"] / physics_dt + 0.5))
        self.sample["effective_motor_delay_seconds"] = self.motor_delay * physics_dt
        self.motor_active = (self.motor_delay > 0 or self.sample["motor_time_constant_seconds"] > 0
                             or self.sample["motor_gain_scale"] != 1.0)
        n_motor = len(env._device_actuator_ids)
        self.commands = deque([np.zeros(n_motor) for _ in range(self.motor_delay)], maxlen=self.motor_delay + 1)
        self.output = np.zeros(n_motor)
        ids = self.body_ids
        mass = self.sample["human_mass_scale"]
        inertia = self.sample["human_inertia_scale"]
        strength = self.sample["muscle_strength_scale"]
        # Use one scalar per episode: preserve body proportions and inertia triangle inequalities.
        # Inertia scales with mass, plus an independent multiplicative inertia uncertainty.
        for sim in (env.sim,) if env.sim_obsd is env.sim else (env.sim, env.sim_obsd):
            model = sim.model
            for key, value in self.nominal.items():
                getattr(model, key)[:] = value
            model.body_mass[ids] *= mass
            model.body_inertia[ids] *= mass * inertia
            # MuJoCo muscle force parameter is gainprm[2] AND biasprm[2]. For force=-1,
            # MuJoCo derives force from acc0 and the scale parameter [3] instead.
            for key in ("actuator_gainprm", "actuator_biasprm"):
                params = getattr(model, key)
                fixed = self.muscle_ids[params[self.muscle_ids, 2] >= 0]
                auto = self.muscle_ids[params[self.muscle_ids, 2] < 0]
                params[fixed, 2] *= strength
                params[auto, 3] *= strength
            # Refresh subtree masses, inverse inertias and actuator acceleration constants.
            if self.randomize_inertial:
                sim.lib.mj_setConst(sim.get_handle(model), sim.get_handle(sim.data))
                sim.forward()

    def observe(self, obs):
        if not self.active:
            return obs
        result = np.asarray(obs).copy()
        measured = result[self.indices] + self.bias + self.env.np_random.normal(size=len(self.indices)) * self.noise_std
        if not self.observations:
            self.observations.extend([measured.copy() for _ in range(self.delay)])
        self.observations.append(measured)
        result[self.indices] = self.observations[0]
        return result

    def motor_control(self, desired, physics_dt, limits):
        """Advance at physics rate: transport delay, first-order response, gain, saturation."""
        self.commands.append(np.asarray(desired).copy())
        delayed = self.commands[0]
        tau = self.sample["motor_time_constant_seconds"]
        alpha = 1.0 if tau == 0 else -np.expm1(-physics_dt / tau)
        target = np.clip(delayed * self.sample["motor_gain_scale"], limits[:, 0], limits[:, 1])
        self.output += alpha * (target - self.output)
        return self.output.copy()
