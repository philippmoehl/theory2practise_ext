from abc import ABC, abstractmethod
from collections import namedtuple
import os
from pathlib import Path
import shutil

import ray
from ray.tune.integration.wandb import WandbTrainableMixin
import torch
import wandb

from . import utils


class Preprocessor(ABC):
    """
    Base class for different preprocessors.
    """

    @abstractmethod
    def run(self):
        pass


class SetupEnviron(Preprocessor):
    """
    Preprocessor to setup environment variables.
    """

    def __init__(self, environ_update):
        self.environ_update = environ_update

    def run(self):
        os.environ.update(self.environ_update)


class LogSpecs(Preprocessor):
    """
    Preprocessor to log specification files.
    """

    def __init__(self, spec_dir, log_path, overwrite=False):
        self.spec_dir = utils.absolute_path(spec_dir)
        self.log_path = utils.absolute_path(log_path)
        self.overwrite = overwrite

    def _log_dir(self, source_dir, target_dir):
        target_dir.mkdir(exist_ok=True, parents=True)
        for f in source_dir.iterdir():
            if f.is_dir():
                self._log_dir(f, target_dir / f.name)
            else:
                target_f = target_dir / f.name
                if not target_f.is_file() or self.overwrite:
                    shutil.copy(f, target_f)
                elif not utils.load_spec(target_f) == utils.load_spec(f):
                    print(f"Different spec at {target_f} but overwrite is false.")
                    # logger.warning(
                    #     f"Different spec at {target_f} but overwrite is false."
                    # )

    def run(self):
        self._log_dir(self.spec_dir, self.log_path / "specs")


class ModelSaver(Preprocessor):
    """
    Preprocessor to save the parameters of target functions.
    """

    def __init__(self, model_spec, path, seeds=0, overwrite=False):
        self.model_spec = model_spec
        self.path = utils.absolute_path(path)
        if not (isinstance(seeds, list) or isinstance(seeds, tuple)):
            seeds = [seeds]
        self.seeds = seeds
        self.overwrite = overwrite

    @staticmethod
    def state_dict_close(*args):
        state_dicts = [utils.flatten(s) for s in args]
        state_dict = state_dicts.pop()
        return all(
            all(torch.allclose(tensor, s[k]) for s in state_dicts)
            for k, tensor in state_dict.items()
        )

    def run(self):
        self.path.mkdir(exist_ok=True, parents=True)
        spec_file = self.path / "spec.yaml"
        if not spec_file.is_file() or self.overwrite:
            utils.store_spec(self.model_spec, spec_file)
        elif not utils.load_spec(spec_file) == self.model_spec:
            logger.warning(f"Different spec at {spec_file} but overwrite is false.")

        for s in self.seeds:
            file = self.path / f"{s}.pt"
            utils.determinism(s)
            model = utils.deserialize_spec(self.model_spec)
            if not file.is_file() or self.overwrite:
                torch.save(model.state_dict(), file)
            elif not ModelSaver.state_dict_close(model.state_dict(), torch.load(file)):
                logger.warning(
                    f"Different model checkpoint at {file} but overwrite is false."
                )


def restore_model(model, params_file, freeze=True):
    """
    Load model parameters from a file.
    """
    params = torch.load(utils.absolute_path(params_file))
    model.load_state_dict(params)
    model.eval()
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    return model


TrainableConfig = namedtuple(
    "TrainableConfig",
    [
        "target_fn",
        "n_samples",
        "algorithm",
        "test_distribution_wrapper",
        "test_losses",
        "test_batches",
        "test_batch_size",
        "dim",
        "log_path",
        "wandb",
        "plot_x",
        "plot_x_axis",
        "plot_save_dir",
        "device",
        "dtype",
        "seed",
    ],
    defaults=(None, None, None, "plots", "cuda", torch.float, None),
)


TrainablePdeConfig = namedtuple(
    "TrainablePdeConfig",
    [
        "pde",
        "n_f",
        "algorithm",
        "test_grid",
        "test_losses",
        "dim",
        "log_path",
        "wandb",
        "plot_x_axis",
        "plot_t_axis",
        "plot_save_dir",
        "device",
        "dtype",
        "seed",
    ],
    defaults=(None, None, None, "plots", "cuda", torch.float, None),
)


class Trainable(WandbTrainableMixin, ray.tune.Trainable):
    """
    Tune Trainable for executing experiments.
    """

    def setup(self, config):
        # settings
        self.seed = config.get("seed")
        if self.seed is not None:
            utils.determinism(self.seed)

        config = utils.deserialize_spec(config)
        self._device = config.device if torch.cuda.is_available() else "cpu"
        self._dtype = config.dtype

        # algorithm
        self.algorithm = config.algorithm

        if hasattr(config, 'target_fn'):
            self.target_fn = config.target_fn
            self.target_fn = utils.distribute(self.target_fn, self._device, self._dtype)
            self.algorithm.initialize(
                self.target_fn, config.n_samples, device=self._device, dtype=self._dtype
            )
        if hasattr(config, 'pde'):
            config.pde.to(device=self._device, dtype=self._dtype)
            self.pde = config.pde
            self.algorithm.initialize(
                self.pde, config.n_f, device=self._device, dtype=self._dtype
            )

        # test config
        if hasattr(config, 'test_distribution_wrapper'):
            config.test_distribution_wrapper.to(device=self._device, dtype=self._dtype)
            self.test_distribution = config.test_distribution_wrapper.get_distribution()
            self._test_batches = config.test_batches
            self._test_batch_size = config.test_batch_size
        if hasattr(config, 'test_grid'):
            config.test_grid.to(device=self._device, dtype=self._dtype)
            self.test_grid = config.test_grid

        self.metrics = utils.Metrics(losses=config.test_losses)

        # plot
        if hasattr(config, 'plot_x'):
            self._plot_x = config.plot_x
            self._plot_x_axis = config.plot_x_axis
            if self._plot_x is not None:
                self._plot_x = self._plot_x.to(device=self._device, dtype=self._dtype)
        else:
            self._plot_x_axis = config.plot_x_axis
            self._plot_t_axis = config.plot_t_axis

        if config.plot_save_dir is None:
            self._plot_save_path = None
        else:
            self._plot_save_path = (
                utils.absolute_path(self.logdir) / config.plot_save_dir
            )
            self._plot_save_path.mkdir(exist_ok=True)

        self.save_attrs = ["metrics"] + [
            f"algorithm/{attr}" for attr in self.algorithm.save_attrs
        ]

    def test(self):

        if hasattr(self.algorithm.model, "eval"):
            self.algorithm.model.eval()
        self.metrics.zero()
        with torch.no_grad():
            if hasattr(self, 'test_distribution'):
                for _ in range(self._test_batches):
                    x = self.test_distribution.sample((self._test_batch_size,))
                    self.metrics(prediction=self.algorithm.model(x), y=self.target_fn(x))
            if hasattr(self, 'test_grid'):
                _x_star = self.test_grid.x_star()
                self.metrics(prediction=self.algorithm.model(_x_star), y=self.pde.solver(_x_star))
        return self.metrics.finalize()

    def _step(self, train=True):
        results = {"iteration": self.metrics.epoch}
        if train:
            results["train"] = self.algorithm.run()

        if hasattr(self, '_plot_x') and self._plot_x is not None:
            file = self._plot_save_path
            if file is not None:
                file = self._plot_save_path / f"iteration={self.metrics.epoch}.pdf"
            results["visualization"] = utils.visualize(
                plot_x=self._plot_x,
                plot_x_axis=self._plot_x_axis,
                named_fns={
                    "target": self.target_fn,
                    "model": self.algorithm.model,
                },
                samples=self.algorithm.samples,
                file=file,
                update_layout={
                    "title": f"{self.algorithm.samples[0].shape[0]} samples | seed {self.seed}"
                },
            )

        elif (self._plot_x_axis is not None) and (self._plot_t_axis is not None):
            file = self._plot_save_path
            if file is not None:
                file = self._plot_save_path / f"iteration={self.metrics.epoch}.pdf"
            results["visualization"] = utils.visualize_pde(
                plot_x_axis=self._plot_x_axis,
                plot_t_axis=self._plot_t_axis,
                pde=self.pde.solver,
                model=self.algorithm.model,
                file=file,
                update_layout={
                    "title": "Heatmap", "xaxis_title": "t", "yaxis_title": "x"
                },
            )

        results["test"] = self.test()
        wandb.log(results)
        results.pop("visualization", None)
        results.pop("iteration")
        return results

    def step(self):
        if self.metrics.epoch == 0:
            self._step(train=False)
            wandb.run.summary["device"] = str(self._device)
            wandb.run.summary["n_gpus"] = torch.cuda.device_count()
            wandb.run.summary["dtype"] = str(self._dtype)

        results = self._step()
        return utils.flatten(results)

    def save_checkpoint(self, tmp_checkpoint_dir):
        state_dict = {
            k: utils.nested_get(self, k).state_dict() for k in self.save_attrs
        }
        checkpoint_path = Path(tmp_checkpoint_dir) / "state.pt"
        torch.save(state_dict, checkpoint_path)
        return tmp_checkpoint_dir
        # for older versions of ray: https://github.com/ray-project/ray/issues/15367
        # return str(checkpoint_path)

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = Path(tmp_checkpoint_dir) / "state.pt"
        state_dict = torch.load(checkpoint_path, map_location=self._device)
        for k in self.save_attrs:
            utils.nested_get(self, k).load_state_dict(state_dict[k])


class Experiment(ray.tune.Experiment):
    """
    Tune Experiment with preprocessors.
    """

    def __init__(self, *args, preprocessors=None, **kwargs):
        self.preprocessors = preprocessors or []
        super().__init__(*args, **kwargs)


class Runner:
    """
    Tune trainer to run a Tune Trainable.
    """

    def __init__(
        self,
        tune_run_kwargs=None,
        ray_init_kwargs=None,
    ):
        self.tune_run_kwargs = tune_run_kwargs or {}
        self.ray_init_kwargs = ray_init_kwargs or {}

    def run(self, exp_files):

        experiments = [utils.load_spec(f, deserialize=True) for f in exp_files]

        for exp in experiments:
            for preprocessor in exp.preprocessors:
                preprocessor.run()

        ray.init(**self.ray_init_kwargs)

        analysis = ray.tune.run(
            experiments,
            **self.tune_run_kwargs,
        )

        ray.shutdown()
        return analysis
