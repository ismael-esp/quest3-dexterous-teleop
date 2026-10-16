# Subset of the thesis task registry: only the dual-arm handover env the VR teleop drives.
import importlib
from typing import Dict, Tuple


# Keep this module free of Isaac/Omniverse imports at import-time.
# Task modules are imported lazily inside make_env() after SimulationApp starts.
TASK_MAP: Dict[str, Tuple[str, str, str, str, str]] = {
    # task_name: (task_module, cfg_class, env_class, vec_module, vec_class)
    "handover_tshape": (
        "tasks.task_handover_tshape",
        "TaskEnvCfg",
        "TaskEnv",
        "trl_utils.common_cfg",
        "CommonVecEnv",
    ),
}


def _import_attr(module_path: str, attr_name: str):
    mod = importlib.import_module(module_path)
    try:
        return getattr(mod, attr_name)
    except AttributeError as exc:
        raise AttributeError(f"'{module_path}' has no attribute '{attr_name}'") from exc


def _resolve_task_classes(task_name: str):
    if task_name not in TASK_MAP:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(TASK_MAP.keys())}")

    task_module, cfg_name, env_name, vec_module, vec_name = TASK_MAP[task_name]
    cfg_cls = _import_attr(task_module, cfg_name)
    env_cls = _import_attr(task_module, env_name)
    vec_cls = _import_attr(vec_module, vec_name)
    return cfg_cls, env_cls, vec_cls


def make_env(task_name, num_envs, device="cuda:0", env_cfg_overrides=None):
    """태스크 이름으로 환경을 생성해서 반환하는 헬퍼 함수"""
    CfgClass, EnvClass, VecEnvClass = _resolve_task_classes(task_name)

    env_cfg = CfgClass()
    env_cfg.scene.num_envs = num_envs
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "device"):
        try:
            env_cfg.sim.device = str(device)
        except Exception:
            pass
    if env_cfg_overrides:
        for key, value in dict(env_cfg_overrides).items():
            try:
                setattr(env_cfg, key, value)
            except Exception:
                pass
    if hasattr(env_cfg, "apply_runtime_mode"):
        try:
            env_cfg.apply_runtime_mode()
        except Exception:
            pass
    raw_env = EnvClass(cfg=env_cfg)
    raw_env.reset()

    # Some legacy wrappers may not accept device kwarg.
    try:
        env = VecEnvClass(raw_env, device=device)
    except TypeError:
        env = VecEnvClass(raw_env)

    return env, raw_env, env_cfg
