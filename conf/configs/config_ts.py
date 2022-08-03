import dataclasses
from dataclasses import dataclass
import pathlib
from typing import Dict, List, Any, Optional

from omegaconf import MISSING

from .config import ExperimentConfig, TrainableConfig


@dataclass
class TrainableTsConfig(TrainableConfig):
    dim: int = MISSING
    dtype: Any = "${get_dtype: torch.double}"
    plot_x: Optional[Dict] = MISSING
    plot_x_axis: Optional[Dict] = MISSING
    target_fn: Dict = MISSING
    test_batch_size: int = MISSING
    test_batches: int = MISSING
    test_distribution_wrapper: Dict = MISSING
    test_losses: List = dataclasses.field(
        default_factory=lambda: [
            {
                "_target_": "theory2practice.utils.LpLoss",
                "p": 1,
            },
            {
                "_target_": "theory2practice.utils.LpLoss",
                "p": 2,
            },
            {
                "_target_": "theory2practice.utils.LpLoss",
                "p": "infinity",
            },
        ]
    )


@dataclass
class TsExperimentConfig(ExperimentConfig):
    config: TrainableTsConfig = MISSING
    resources_per_trial: Dict = dataclasses.field(
        default_factory=lambda: {"cpu": 1, "gpu": 0})
    run: Any = "${get_cls: theory2practice.experiments.Trainable1}"
    stop: Dict = dataclasses.field(
        default_factory=lambda: {"training_iteration": 5})


@dataclass
class TsTestConfig(TsExperimentConfig):
    config: TrainableTsConfig = TrainableTsConfig(
        dim=1,
        log_path=pathlib.Path("results/ts/test"),
        n_samples=100,
        plot_x={
            "_target_": "theory2practice.utils.create_tensor",
            "creation_op": "${get_method: torch.linspace}",
            "end": 0.5,
            "methods": [{"args": [-1], "name": "unsqueeze"}],
            "start": -0.5,
            "steps": 1000,
        },
        plot_x_axis={
            "_target_": "theory2practice.utils.create_tensor",
            "creation_op": "${get_method: torch.linspace}",
            "end": 0.5,
            "start": -0.5,
            "steps": 1000,
        },
        seed=10000,
        target_fn={
            "_target_": "theory2practice.experiments.restore_model",
            "model": "results/ts/test/target_fn/spec.yaml",
            "params_file": f"results/ts/test/target_fn/0.pt",
        },
        test_batch_size=131072,
        test_batches=128,
        test_distribution_wrapper={
            "_target_": "theory2practice.utils.DistributionWrapper",
            "distribution_factory": "${get_method: torch.distributions.uniform.Uniform}",
            "tensor_kwargs": {
                "high": 0.5,
                "low": {
                    "_target_": "theory2practice.utils.create_tensor",
                    "data": [-0.5],
                }
            }
        },
        wandb={"group": "exp", "project": "t2p_ts_test"}
    )
    local_dir: pathlib.Path = pathlib.Path("results/ts/test")
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
                "log_path": pathlib.Path("results/ts/test"),
                "spec_dir": pathlib.Path("conf/specs/ts/test"),
            },
            {
                "_target_": "theory2practice.experiments.ModelSaver",
                "_recursive_": False,
                "model_spec": {
                    "_target_": "theory2practice.algorithms.FeedForward",
                    "activation": {
                        "_target_": "torch.nn.ReLU",
                    },
                    "bias_init_factory": {
                        "_target_": "theory2practice.utils.method_wrapper",
                        "func": "torch.nn.init.uniform_",
                    },
                    "bias_init_kwargs": {
                        "a": -0.5,
                        "b": 0.5,
                    },
                    "depth": 5,
                    "input_dim": 1,
                    "weight_init_factory": {
                        "_target_": "theory2practice.utils.method_wrapper",
                        "func": "torch.nn.init.uniform_",
                    },
                    "weight_init_kwargs": {
                        "a": -0.5,
                        "b": 0.5,
                    },
                    "width": 32,
                },
                "path": pathlib.Path("results/ts/test/target_fn"),
                "seeds": [0],
            }
        ]
    )


@dataclass
class OneD5x32Config(TsExperimentConfig):
    config: TrainableTsConfig = TrainableTsConfig(
        dim=1,
        log_path=pathlib.Path("results/ts/1d_5x32"),
        n_samples={
            "grid_search": [
                10**i
                for i in range(2, 5)
            ]
        },
        plot_x={
            "_target_": "theory2practice.utils.create_tensor",
            "creation_op": "${get_method: torch.linspace}",
            "end": 0.5,
            "methods": [{"args": [-1], "name": "unsqueeze"}],
            "start": -0.5,
            "steps": 1000,
        },
        plot_x_axis={
            "_target_": "theory2practice.utils.create_tensor",
            "creation_op": "${get_method: torch.linspace}",
            "end": 0.5,
            "start": -0.5,
            "steps": 1000,
        },
        seed={
            "grid_search": [
                10000,
                10001,
                10002,
            ]
        },
        target_fn={
            "_target_": "theory2practice.experiments.restore_model",
            "model": "results/ts/1d_5x32/target_fn/spec.yaml",
            "params_file": {
                "grid_search": [
                    pathlib.Path(f"results/ts/1d_5x32/target_fn/{i}.pt")
                    for i in range(40)
                ]
            },
        },
        test_batch_size=131072,
        test_batches=128,
        test_distribution_wrapper={
            "_target_": "theory2practice.utils.DistributionWrapper",
            "distribution_factory": "${get_method: torch.distributions.uniform.Uniform}",
            "tensor_kwargs": {
                "high": 0.5,
                "low": {
                    "_target_": "theory2practice.utils.create_tensor",
                    "data": [-0.5],
                }
            }
        },
        wandb={"group": "exp", "project": "t2p_ts_1d_5x32"}
    )
    local_dir: pathlib.Path = pathlib.Path("results/ts/1d_5x32")
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
                "log_path": pathlib.Path("results/ts/1d_5x32"),
                "spec_dir": pathlib.Path("conf/specs/ts/1d_5x32"),
            },
            {
                "_target_": "theory2practice.experiments.ModelSaver",
                "_recursive_": False,
                "model_spec": {
                    "_target_": "theory2practice.algorithms.FeedForward",
                    "activation": {
                        "_target_": "torch.nn.ReLU",
                    },
                    "bias_init_factory": {
                        "_target_": "theory2practice.utils.method_wrapper",
                        "func": "torch.nn.init.uniform_",
                    },
                    "bias_init_kwargs": {
                        "a": -0.5,
                        "b": 0.5,
                    },
                    "depth": 5,
                    "input_dim": 1,
                    "weight_init_factory": {
                        "_target_": "theory2practice.utils.method_wrapper",
                        "func": "torch.nn.init.uniform_",
                    },
                    "weight_init_kwargs": {
                        "a": -0.5,
                        "b": 0.5,
                    },
                    "width": 32,
                },
                "path": pathlib.Path("results/ts/1d_5x32/target_fn"),
                "seeds": [
                    i
                    for i in range(40)
                ],
            }
        ]
    )


@dataclass
class ThreeD5x32Config(TsExperimentConfig):
    config: TrainableTsConfig = TrainableTsConfig(
        dim=3,
        log_path=pathlib.Path("results/ts/3d_5x32"),
        n_samples={
            "grid_search": [
                10**i
                for i in range(2, 6)
            ]
        },
        plot_x={
            "_target_": "theory2practice.utils.create_tensor",
            "creation_op": "${get_method: torch.linspace}",
            "end": 0.5,
            "methods": [
                {"args": [-1], "name": "unsqueeze"},
                {"args": [-1, 3], "name": "expand"}
                        ],
            "start": -0.5,
            "steps": 1000,
        },
        plot_x_axis={
            "_target_": "theory2practice.utils.create_tensor",
            "creation_op": "${get_method: torch.linspace}",
            "end": 0.5,
            "start": -0.5,
            "steps": 1000,
        },
        seed={
            "grid_search": [
                10000,
                10001,
                10002,
            ]
        },
        target_fn={
            "_target_": "theory2practice.experiments.restore_model",
            "model": "results/ts/3d_5x32/target_fn/spec.yaml",
            "params_file": {
                "grid_search": [
                    pathlib.Path(f"results/ts/3d_5x32/target_fn/{i}.pt")
                    for i in range(40)
                ]
            },
        },
        test_batch_size=131072,
        test_batches=128,
        test_distribution_wrapper={
            "_target_": "theory2practice.utils.DistributionWrapper",
            "distribution_factory": "${get_method: torch.distributions.uniform.Uniform}",
            "tensor_kwargs": {
                "high": 0.5,
                "low": {
                    "_target_": "theory2practice.utils.create_tensor",
                    "data": [-0.5],
                    "methods": [
                        {"args": [3], "name": "expand"},
                    ],
                },
            },
        },
        wandb={"group": "exp", "project": "t2p_ts_3d_5x32"}
    )
    local_dir: pathlib.Path = pathlib.Path("results/ts/3d_5x32")
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
                "log_path": pathlib.Path("results/ts/3d_5x32"),
                "spec_dir": pathlib.Path("conf/specs/ts/3d_5x32"),
            },
            {
                "_target_": "theory2practice.experiments.ModelSaver",
                "_recursive_": False,
                "model_spec": {
                    "_target_": "theory2practice.algorithms.FeedForward",
                    "activation": {
                        "_target_": "torch.nn.ReLU",
                    },
                    "bias_init_factory": {
                        "_target_": "theory2practice.utils.method_wrapper",
                        "func": "torch.nn.init.uniform_",
                    },
                    "bias_init_kwargs": {
                        "a": -0.5,
                        "b": 0.5,
                    },
                    "depth": 5,
                    "input_dim": 3,
                    "weight_init_factory": {
                        "_target_": "theory2practice.utils.method_wrapper",
                        "func": "torch.nn.init.uniform_",
                    },
                    "weight_init_kwargs": {
                        "a": -0.5,
                        "b": 0.5,
                    },
                    "width": 32,
                },
                "path": pathlib.Path("results/ts/3d_5x32/target_fn"),
                "seeds": [
                    i
                    for i in range(40)
                ],
            }
        ]
    )


def store():
    from .config import cs

    cs.store(group="specs", name="base_ts_test", node=TsTestConfig)
    cs.store(group="specs", name="base_1d_5x32", node=OneD5x32Config)
    cs.store(group="specs", name="base_3d_5x32",
             node=ThreeD5x32Config)
