"""Resample reference trajectories without changing their physical time base."""

import numpy as np


REFERENCE_MOTION_MODES = {"legacy_scaled", "fixed_rate"}


def resample_reference_data(data, target_rate, mode="legacy_scaled"):
    if mode not in REFERENCE_MOTION_MODES:
        raise ValueError(f"reference_motion_mode must be one of {sorted(REFERENCE_MOTION_MODES)}")
    if mode == "legacy_scaled" and "resampled_series_data" in data:
        return data
    metadata = dict(data["metadata"])
    source_rate = metadata["sample_rate"]
    if mode == "fixed_rate" and "resampled_series_data" in data:
        source_rate = metadata.get("resampled_sample_rate", source_rate)
    if not np.isfinite(source_rate) or source_rate <= 0 or not np.isfinite(target_rate) or target_rate <= 0:
        raise ValueError("Reference source and target sample rates must be finite and positive")
    series = data["series_data"]
    lengths = {len(values) for values in series.values()}
    if not series or len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError("Reference series must have the same length and at least two samples")
    length = next(iter(lengths))
    original_x = np.arange(length, dtype=float)
    if mode == "fixed_rate":
        # A frame is exactly 1 / target_rate seconds. Endpoint-aligned linspace
        # would stretch each frame to fit the final source sample instead.
        new_length = int(np.floor((length - 1) * target_rate / source_rate + 1e-9)) + 1
        new_x = np.arange(new_length, dtype=float) * source_rate / target_rate
    else:
        # Preserve historical grids and results when the config does not opt in.
        new_length = int(length * target_rate / source_rate)
        new_x = np.linspace(0, length - 1, new_length)
    result = dict(data)
    result["series_data"] = {}
    for key, values in series.items():
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(f"Reference series {key} must be a finite one-dimensional array")
        result["series_data"][key] = np.interp(new_x, original_x, values)
    metadata["resampled_data_length"] = new_length
    metadata["resampled_sample_rate"] = target_rate
    metadata["reference_motion_mode"] = mode
    result["metadata"] = metadata
    result["resampled_series_data"] = {}
    return result
