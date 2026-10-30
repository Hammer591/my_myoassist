"""Project device registry, alongside assist_sim's bundled devices.

Keep local assets in the project rather than modifying site-packages. All RL
and CO callers enter through the shared compose pipeline.
"""

from pathlib import Path

from assist_sim import registry
from assist_sim.loading import load_combined as _load_combined


def local_device_configs() -> dict[str, Path]:
    import MyHipExo

    return {"PhotoHipExo_L1": Path(MyHipExo.__file__).parent / "L1config.yaml"}


def available_combinations() -> dict[str, list[str]]:
    from assist_sim.config import DeviceConfig

    combos = registry.get_available_combinations()
    for devices in combos.values():
        devices.append("NoDevice")
    for key, path in local_device_configs().items():
        config = DeviceConfig.from_yaml(str(path))
        for msk, devices in combos.items():
            if msk in config.compatible_msk:
                devices.append(key)
    return combos


def device_source_token(device_key: str) -> str:
    """Invalidate composed caches when local config, geometry or inertia changes."""
    if device_key == "NoDevice":
        from assist_sim.loading import _myosim_token

        return f"bare@{_myosim_token()}@{Path(__file__).stat().st_mtime_ns}"
    path = local_device_configs().get(device_key)
    if path is None:
        return "bundled"
    import hashlib

    digest = hashlib.sha256()
    for source in sorted(path.parent.rglob("*")):
        if source.suffix in {".yaml", ".xml", ".stl"}:
            digest.update(str(source.relative_to(path.parent)).encode())
            digest.update(source.read_bytes())
    return digest.hexdigest()


def load_combined(msk_key: str, device_key: str, **kwargs):
    if device_key == "NoDevice":
        from assist_sim.loading import load_msk

        if kwargs.pop("planar_root", False) and msk_key != "myolegs22":
            raise ValueError("NoDevice planar_root currently requires myolegs22")
        return load_msk(msk_key, **kwargs)
    path = local_device_configs().get(device_key)
    if path is None:
        return _load_combined(msk_key, device_key, **kwargs)

    from assist_sim.combine import ModelCombiner
    from assist_sim.config import DeviceConfig

    if device_key not in available_combinations().get(msk_key, []):
        raise ValueError(f"Device {device_key!r} is not compatible with MSK {msk_key!r}")
    config = DeviceConfig.from_yaml(str(path))
    # assist_sim exposes fresh human specs through its registry; the combiner
    # mutates them, so never reuse a previously built spec.
    human = registry._resolve_msk(msk_key)
    return ModelCombiner().combine(human, config, msk_key=msk_key, **kwargs)
