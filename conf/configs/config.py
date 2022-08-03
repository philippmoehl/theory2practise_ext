from dataclasses import dataclass
import pathlib
from typing import Dict, List, Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

cs = ConfigStore.instance()


@dataclass
class ExperimentConfig:
    _target_: str = "theory2practice.experiments.Experiment"
    checkpoint_freq: int = 1
    config: Any = MISSING
    local_dir: pathlib.Path = MISSING
    max_failures: int = 10
    name: str = "exp"
    preprocessors: List = MISSING
    resources_per_trial: Dict = MISSING
    run: Any = MISSING
    stop: Dict = MISSING


@dataclass
class RunnerConfig:
    _target_: str = "theory2practice.experiments.Runner"
    tune_run_kwargs: Dict = MISSING


@dataclass
class Config:
    runner: RunnerConfig = MISSING
    specs: ExperimentConfig = MISSING


@dataclass
class TrainableConfig:
    # _recursive_ and _target_ are needed for grid_search be given to ray.tune
    # before hydra.utils.instantiate is called
    _recursive_: bool = False
    _target_: str = "builtins.dict"
    algorithm: Any = MISSING
    dtype: Any = MISSING
    log_path: pathlib.Path = MISSING
    n_samples: Any = MISSING
    plot_save_dir: pathlib.Path = pathlib.Path("plots")
    seed: Any = MISSING
    test_losses: List = MISSING
    wandb: Dict = MISSING


cs.store(name="base_config", node=Config)
cs.store(group="runner", name="base_runner", node=RunnerConfig)
