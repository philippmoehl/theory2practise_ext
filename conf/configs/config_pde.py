import dataclasses
from dataclasses import dataclass
import pathlib
from typing import Dict, List, Any

from omegaconf import MISSING

from .config import ExperimentConfig, TrainableConfig


@dataclass
class TrainablePdeConfig(TrainableConfig):
    dtype: Any = "${get_dtype: torch.float}"
    n_samples: Any = 0
    pde: Any = MISSING
    seed: Any = None
    test_losses: List = dataclasses.field(
        default_factory=lambda: [
            {
                "_target_": "theory2practice.utils.LpLoss",
                "p": 1,
            },
            {
                "_target_": "theory2practice.utils.LpLoss",
                "p": 2,
                "relative": True,
            },
            {
                "_target_": "theory2practice.utils.LpLoss",
                "p": "infinity",
            },
        ]
    )
    plot_grid: bool = False


@dataclass
class PdeExperimentConfig(ExperimentConfig):
    config: TrainablePdeConfig = MISSING
    resources_per_trial: Dict = dataclasses.field(
        default_factory=lambda: {"cpu": 1, "gpu": 0})
    run: Any = "${get_cls: theory2practice.experiments.Trainable2}"
    stop: Dict = dataclasses.field(
        default_factory=lambda: {"training_iteration": 1})


@dataclass
class PdeTestConfig(PdeExperimentConfig):
    local_dir: pathlib.Path = pathlib.Path("results/pde/test")
    preprocessors: List = dataclasses.field(
        default_factory=lambda: [
            {
                "_target_": "theory2practice.experiments.SetupEnviron",
                "environ_update": {
                    "TUNE_GLOBAL_CHECKPOINT_S": "600",
                    "TUNE_MAX_PENDING_TRIALS_PG": "1",
                    "TUNE_RESULT_BUFFER_LENGTH": "0",
                    "TUNE_WARN_THRESHOLD_S": "2",
                },
            },
            {
                "_target_": "theory2practice.experiments.LogSpecs",
                "log_path": pathlib.Path("results/pde/test"),
                "spec_dir": pathlib.Path("conf/specs/pde/test"),
            }
        ]
    )
    config: TrainablePdeConfig = TrainablePdeConfig(
        log_path=pathlib.Path("results/pde/test"),
        pde={
            "_target_": "theory2practice.pde_utils.ConvectionDiffusion",
            "params": {"beta": 1},
            "u0": "${get_lambda: 'torch.sin(x)'}",
            "system": "convection",
            "source": 0,
        },
        wandb={"group": "exp", "project": "t2p_pde_test"}
    )


@dataclass
class ConvConfig(PdeExperimentConfig):
    local_dir: pathlib.Path = pathlib.Path("results/pde/conv")
    preprocessors: List = dataclasses.field(
        default_factory=lambda: [
            {
                "_target_": "theory2practice.experiments.SetupEnviron",
                "environ_update": {
                    "TUNE_GLOBAL_CHECKPOINT_S": "600",
                    "TUNE_MAX_PENDING_TRIALS_PG": "1",
                    "TUNE_RESULT_BUFFER_LENGTH": "0",
                    "TUNE_WARN_THRESHOLD_S": "2",
                },
            },
            {
                "_target_": "theory2practice.experiments.LogSpecs",
                "log_path": pathlib.Path("results/pde/conv"),
                "spec_dir": pathlib.Path("conf/specs/pde/conv"),
            }
        ]
    )
    config: TrainablePdeConfig = TrainablePdeConfig(
        log_path=pathlib.Path("results/pde/conv"),
        pde={
            "_target_": "theory2practice.pde_utils.ConvectionDiffusion",
            "params": {"beta": 1},
            "u0": "${get_lambda: 'torch.sin(x)'}",
            "system": "convection",
            "source": 0,
        },
        wandb={"group": "exp", "project": "t2p_pde_conv"}
    )


@dataclass
class DiffConfig(PdeExperimentConfig):
    local_dir: pathlib.Path = pathlib.Path("results/pde/diff")
    preprocessors: List = dataclasses.field(
        default_factory=lambda: [
            {
                "_target_": "theory2practice.experiments.SetupEnviron",
                "environ_update": {
                    "TUNE_GLOBAL_CHECKPOINT_S": "600",
                    "TUNE_MAX_PENDING_TRIALS_PG": "1",
                    "TUNE_RESULT_BUFFER_LENGTH": "0",
                    "TUNE_WARN_THRESHOLD_S": "2",
                },
            },
            {
                "_target_": "theory2practice.experiments.LogSpecs",
                "log_path": pathlib.Path("results/pde/diff"),
                "spec_dir": pathlib.Path("conf/specs/pde/diff"),
            }
        ]
    )
    config: TrainablePdeConfig = TrainablePdeConfig(
        log_path=pathlib.Path("results/pde/diff"),
        pde={
            "_target_": "theory2practice.pde_utils.ConvectionDiffusion",
            "params": {"nu": 1},
            "u0": "${get_lambda: 'torch.sin(x)'}",
            "system": "diffusion",
            "source": 0,
        },
        wandb={"group": "exp", "project": "t2p_pde_diff"}
    )


@dataclass
class ReactConfig(PdeExperimentConfig):
    local_dir: pathlib.Path = pathlib.Path("results/pde/react")
    preprocessors: List = dataclasses.field(
        default_factory=lambda: [
            {
                "_target_": "theory2practice.experiments.SetupEnviron",
                "environ_update": {
                    "TUNE_GLOBAL_CHECKPOINT_S": "600",
                    "TUNE_MAX_PENDING_TRIALS_PG": "1",
                    "TUNE_RESULT_BUFFER_LENGTH": "0",
                    "TUNE_WARN_THRESHOLD_S": "2",
                },
            },
            {
                "_target_": "theory2practice.experiments.LogSpecs",
                "log_path": pathlib.Path("results/pde/react"),
                "spec_dir": pathlib.Path("conf/specs/pde/react"),
            }
        ]
    )
    config: TrainablePdeConfig = TrainablePdeConfig(
        log_path=pathlib.Path("results/pde/react"),
        pde={
            "_target_": "theory2practice.pde_utils.Reaction",
            "params": {"rho": 0.1},
            "u0": "${get_lambda: 'torch.sin(x)'}",
        },
        wandb={"group": "exp", "project": "t2p_pde_react"}
    )


@dataclass
class ReactDiffConfig(PdeExperimentConfig):
    local_dir: pathlib.Path = pathlib.Path("results/pde/react_diff")
    preprocessors: List = dataclasses.field(
        default_factory=lambda: [
            {
                "_target_": "theory2practice.experiments.SetupEnviron",
                "environ_update": {
                    "TUNE_GLOBAL_CHECKPOINT_S": "600",
                    "TUNE_MAX_PENDING_TRIALS_PG": "1",
                    "TUNE_RESULT_BUFFER_LENGTH": "0",
                    "TUNE_WARN_THRESHOLD_S": "2",
                },
            },
            {
                "_target_": "theory2practice.experiments.LogSpecs",
                "log_path": pathlib.Path("results/pde/react_diff"),
                "spec_dir": pathlib.Path("conf/specs/pde/react_diff"),
            }
        ]
    )
    config: TrainablePdeConfig = TrainablePdeConfig(
        log_path=pathlib.Path("results/pde/react_diff"),
        pde={
            "_target_": "theory2practice.pde_utils.ReactionDiffusion",
            "params": {"rho": 0.1, "nu": 0.5},
            "u0": "${get_lambda: 'torch.sin(x)'}",
        },
        wandb={"group": "exp", "project": "t2p_pde_react_diff"}
    )


@dataclass
class BurgerConfig(PdeExperimentConfig):
    local_dir: pathlib.Path = pathlib.Path("results/pde/burger")
    preprocessors: List = dataclasses.field(
        default_factory=lambda: [
            {
                "_target_": "theory2practice.experiments.SetupEnviron",
                "environ_update": {
                    "TUNE_GLOBAL_CHECKPOINT_S": "600",
                    "TUNE_MAX_PENDING_TRIALS_PG": "1",
                    "TUNE_RESULT_BUFFER_LENGTH": "0",
                    "TUNE_WARN_THRESHOLD_S": "2",
                },
            },
            {
                "_target_": "theory2practice.experiments.LogSpecs",
                "log_path": pathlib.Path("results/pde/burger"),
                "spec_dir": pathlib.Path("conf/specs/pde/burger"),
            }
        ]
    )
    config: TrainablePdeConfig = TrainablePdeConfig(
        log_path=pathlib.Path("results/pde/burger"),
        pde={
            "_target_": "theory2practice.pde_utils.Burger",
            "params": {"nu": 0.01},
            "u0": "${get_lambda: 'torch.sin(x)'}",
        },
        wandb={"group": "exp", "project": "t2p_pde_burger"}
    )


def store():
    from .config import cs

    cs.store(group="specs", name="base_pde_test", node=PdeTestConfig)
    cs.store(group="specs", name="base_conv", node=ConvConfig)
    cs.store(group="specs", name="base_diff", node=DiffConfig)
    cs.store(group="specs", name="base_react", node=ReactConfig)
    cs.store(group="specs", name="base_react_diff",
             node=ReactDiffConfig)
    cs.store(group="specs", name="base_burger", node=BurgerConfig)
