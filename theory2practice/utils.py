from abc import ABC, abstractmethod
import collections
from functools import partial
from importlib import import_module
import json
import netrc
import os
from pathlib import Path
import random
import time
import yaml

import numpy as np
import plotly.graph_objects as go
import torch

_NO_DEFAULT = object()
WANDB_HOST = "api.wandb.ai"
WANDB_API_KEY = "WANDB_API_KEY"
NO_WANDB_API_KEY = "__placeholder__"

LOADER = {".json": json.load, ".yaml": yaml.safe_load, ".yml": yaml.safe_load}
DUMPER = {
    ".json": partial(json.dumps, indent=4),
    ".yaml": partial(yaml.dump, indent=4),
    ".yml": partial(yaml.dump, indent=4),
}

CALL_KEY = "__call__"
IMPORT_KEY = "__import__"
SPEC_KEY = "__spec__"
FILE_KEY = "__file__"
ENVIRON_KEY = "__environ__"
ENVIRON_DEFAULT_KEY = "default"
SERIALIZATION_KEYS = (CALL_KEY, IMPORT_KEY, SPEC_KEY, FILE_KEY, ENVIRON_KEY)


def project_root():
    """
    Returns the project root folder.
    """
    return Path(__file__).parent.parent


def absolute_path(path):
    """
    Returns the absolute version of a path relative to the project root.
    """
    path = Path(path)
    return path if path.is_absolute() else project_root() / path


def flatten(spec, parent_key="", sep="/"):
    """
    Flattens a nested spec by using a separator string.
    """
    items = []
    new_key = parent_key + sep if parent_key else ""
    if issequenceform(spec):
        for i, v in enumerate(spec):
            items.extend(flatten(v, new_key + str(i), sep=sep).items())
    elif isinstance(spec, collections.abc.Mapping):
        for k, v in spec.items():
            items.extend(flatten(v, new_key + k, sep=sep).items())
    else:
        items.append((parent_key, spec))
    return dict(items)


def nested_get(obj, string, default=_NO_DEFAULT, sep="/"):
    """
    Returns the nested attribute/item/value given by a string.
    """
    for field in string.split(sep):
        try:
            if issequenceform(obj):
                obj = obj[int(field)]
            elif isinstance(obj, collections.abc.Mapping):
                obj = obj[field]
            else:
                obj = getattr(obj, field)
        except (AttributeError, IndexError, KeyError) as err:
            if default is not _NO_DEFAULT:
                return default
            raise ValueError(f"Could not get `{field}` for `{obj}`.") from err
    return obj


def nested_pop(obj, string, default=_NO_DEFAULT, sep="/"):
    """
    Deletes the attribute/item/value at the nested position given by a string.
    """
    substrings = string.split(sep)
    field = substrings[-1]
    try:
        if len(substrings) > 1:
            obj = nested_get(obj, sep.join(substrings[:-1]), default=default, sep=sep)
        if issequenceform(obj):
            return obj.pop(int(field))
        elif isinstance(obj, collections.abc.Mapping):
            return obj.pop(field)
        else:
            return obj.__dict__.pop(field)
    except (AttributeError, IndexError, KeyError, ValueError) as err:
        if default is not _NO_DEFAULT:
            return default
        raise ValueError(f"Could not pop `{field}` for `{obj}`.") from err


def nested_set(obj, string, value, sep="/"):
    """
    Sets the attribute/item/value at the nested position given by a string.
    """
    substrings = string.split(sep)
    field = substrings[-1]
    if len(substrings) > 1:
        obj = nested_get(obj, sep.join(substrings[:-1]), sep=sep)
    if issequenceform(obj):
        obj[int(field)] = value
    elif isinstance(obj, collections.abc.Mapping):
        obj[field] = value
    elif hasattr(obj, field):
        setattr(obj, field, value)
    else:
        raise ValueError(f"Could not set `{field}` for `{obj}`.")


def nested_update(obj, update):
    """
    Update an object by mapping a string to the new value.
    """
    for path, value in update.items():
        nested_set(obj, path, value)


def import_string(string):
    """
    Import a module path and return the attribute/class designated
    by the last name in the dotted path.
    Raise ImportError if the import failed.
    """
    try:
        module_path, class_name = string.rsplit(".", 1)
    except ValueError as err:
        raise ImportError(
            f"`{string}` does not look like a dotted module path."
        ) from err

    module = import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError as err:
        raise ImportError(
            f"Module `{module_path}` does not define a `{class_name}` attribute/class."
        ) from err


def issequenceform(obj):
    """
    Whether the object is a sequence but not a string.
    """
    if isinstance(obj, str):
        return False
    return isinstance(obj, collections.abc.Sequence)


def store_spec(spec, file):
    """
    Stores the spec with the given name.
    """
    file = absolute_path(file)
    file.parent.mkdir(exist_ok=True, parents=True)
    try:
        dump_fn = DUMPER[file.suffix]
    except KeyError as err:
        raise ValueError(
            f"No suitable dumper found for suffix `{file.suffix}`."
        ) from err
    with file.open(mode="w") as f:
        f.write(dump_fn(spec))


def store_specs(specs_mapping, path=None):
    """
    Stores each value as file named according to its key.
    """
    path = path or Path(".")
    for k, v in specs_mapping.items():
        store_spec(v, file=path / k)


def deserialize_spec(spec):
    """
    Deserialize a specification by replacing mappings with
    a class or instance key by the corresponding (imported) class
    or the object instantiated with the provided arguments.
    """
    if issequenceform(spec):
        return [deserialize_spec(v) for v in spec]
    elif isinstance(spec, collections.abc.Mapping):
        keys = [k for k in spec if k in SERIALIZATION_KEYS]
        if len(keys) == 0:
            return {k: deserialize_spec(v) for k, v in spec.items()}
        elif len(keys) == 1:
            key = keys[0]
            if key == CALL_KEY:
                cls_ptr = import_string(spec[key])
                cls_kwargs = {
                    k: deserialize_spec(v) for k, v in spec.items() if not k == CALL_KEY
                }
                return cls_ptr(**cls_kwargs)
            elif key == ENVIRON_KEY:
                default = spec.get(ENVIRON_DEFAULT_KEY, "")
                return os.environ.get(spec[key], default)
            elif key == FILE_KEY:
                update = {k: v for k, v in spec.items() if not k == FILE_KEY}
                return load_spec(spec[key], update=update, deserialize=True)
            if not len(spec) == 1:
                raise ValueError(f"Cannot deserialize spec `{spec}` with arguments!")
            if key == SPEC_KEY:
                return spec[key]
            return import_string(spec[key])
        else:
            raise ValueError(
                f"Invalid spec, multiple deserialization methods `{keys}` are given."
            )
    else:
        return spec


def load_spec(file, update=None, deserialize=False):
    """
    Load a spec from a file and optionally deserialize or update it.
    """
    file = absolute_path(file)
    suffix = file.suffix.lower()
    if suffix not in LOADER:
        raise ValueError(
            f"No suitable loader found for file with suffix `{file.suffix}`."
        )
    with file.open(mode="r") as f:
        spec = LOADER[suffix](f)
    if update is not None:
        nested_update(spec, update)
    if not deserialize:
        return spec
    return deserialize_spec(spec)


def determinism(seed):
    """
    Set seeds and flags for deterministic experiments.
    """
    # see https://github.com/ray-project/ray/issues/8569
    import torch

    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def distribute(module, device="cpu", dtype=torch.float):
    """
    Distribute torch module on GPUs.
    """
    if (
        isinstance(module, torch.nn.Module)
        and device == "cuda"
        and torch.cuda.device_count() > 1
    ):
        module = torch.nn.DataParallel(module)
    if hasattr(module, "to"):
        module.to(device=device, dtype=dtype)
    return module


def create_tensor(data=None, creation_op=None, methods=None, **kwargs):
    """
    Utility function to create torch tensors.
    """
    if (data is None) is (creation_op is None):
        raise ValueError("Either `data` or `creation_op` needs to be specified!")
    if data is not None:
        tensor = torch.tensor(data, **kwargs)
    else:
        tensor = creation_op(**kwargs)
    if methods is not None:
        for method in methods:
            tensor = getattr(tensor, method["name"])(
                *method.get("args", []), **method.get("kwargs", {})
            )
    return tensor


class DistributionWrapper(torch.nn.Module):
    """
    Distribution module.
    """

    # see https://github.com/pytorch/pytorch/issues/7795
    def __init__(
        self,
        distribution_factory,
        *args,
        tensor_kwargs=None,
        **kwargs,
    ):
        super().__init__()
        self.distribution_factory = distribution_factory
        self.args = args
        self.kwargs = kwargs
        if tensor_kwargs is not None:
            for k, v in tensor_kwargs.items():
                self.register_buffer(k, torch.as_tensor(v))

    def get_distribution(self):
        return self.distribution_factory(
            *self.args, **dict(self.named_buffers()), **self.kwargs
        )


class Loss(ABC):
    """
    Base class for different losses.
    """

    @abstractmethod
    def loss(self, prediction, y):
        pass

    @abstractmethod
    def store(self, loss, batch_size):
        pass

    @abstractmethod
    def zero(self):
        pass

    @abstractmethod
    def __call__(self, prediction, y, store=True):
        pass

    @abstractmethod
    def finalize(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass


class LpLoss(Loss):
    """
    (Relative) Lp losses.
    """

    def __init__(self, p=1.0, relative=False, eps=1.0, final_inverse_power=True):

        if not (p == "infinity" or (isinstance(p, (float, int)) and p > 0)):
            raise ValueError("p can either be a positive number or `infinity`.")

        self.p = p
        self.relative = relative
        self.eps = eps
        self.final_inverse_power = final_inverse_power

        self.best = None
        self.last_improve = None
        self._running = []
        self._batch_sizes = []

    def _error(self, prediction, y):
        assert prediction.shape == y.shape
        error = (prediction - y).abs()
        if not self.relative:
            return error
        magnitude = y.abs() + self.eps
        return error / magnitude

    def loss(self, prediction, y):
        error = self._error(prediction, y)
        if self.p == "infinity":
            return error.max()
        return (error**self.p).mean()

    def store(self, loss, batch_size):
        self._running.append(loss.detach())
        self._batch_sizes.append(batch_size)

    def zero(self):
        self._running = []
        self._batch_sizes = []

    def __call__(self, prediction, y, store=True):
        loss = self.loss(prediction, y)
        if store:
            self.store(loss, y.shape[0])
        return loss

    def finalize(self):
        with torch.no_grad():
            losses = torch.stack(self._running)
            if self.p == "infinity":
                final_loss = losses.max()
            else:
                batch_sizes = torch.tensor(
                    self._batch_sizes, device=losses.device, dtype=losses.dtype
                )
                final_loss = (losses * batch_sizes / batch_sizes.sum()).sum()
                if self.final_inverse_power:
                    final_loss = final_loss ** (1 / self.p)

            final_loss = final_loss.item()

        if self.best is None or (final_loss < self.best):
            self.best = final_loss
            self.last_improve = 0
        else:
            self.last_improve += 1

        return {
            "current": final_loss,
            "best": self.best,
            "last_improve": self.last_improve,
        }

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    @property
    def name(self):
        prefix = "rel_" if self.relative else ""
        return f"{prefix}L{self.p}"


class Metrics:
    """
    Metrics for the trainer.
    """

    def __init__(self, losses):
        self.losses = losses
        self.runtime = 0.0
        self.step = 0
        self.epoch = 0
        self._t = time.time()

    def __call__(self, prediction, y, store=True):
        self.step += 1
        output = {
            loss.name: loss(prediction=prediction, y=y, store=store)
            for loss in self.losses
        }
        return output

    def zero(self):
        self._t = time.time()
        for loss in self.losses:
            loss.zero()

    def finalize(self):
        self.runtime += time.time() - self._t
        self.epoch += 1
        result = {loss.name: loss.finalize() for loss in self.losses}
        result.update({"time": self.runtime, "step": self.step, "epoch": self.epoch})
        return result

    def state_dict(self):
        state_dict = self.__dict__.copy()
        state_dict["losses"] = [loss.state_dict() for loss in self.losses]
        return state_dict

    def load_state_dict(self, state_dict):
        state_dict = state_dict.copy()
        for loss, loss_state_dict in zip(self.losses, state_dict.pop("losses")):
            loss.load_state_dict(loss_state_dict)
        self.__dict__.update(state_dict)


def visualize(
    plot_x,
    plot_x_axis=None,
    named_fns=None,
    samples=None,
    file=None,
    update_layout=None,
    fig=None,
):
    """
    Plot functions along a one-dimensional line.
    """

    dim = 1 if plot_x.dim() == 1 else plot_x.shape[-1]
    if plot_x_axis is None:
        if dim == 1:
            plot_x_axis = plot_x
        else:
            raise ValueError(
                "For multivariate plots, x-axis values need to be specified."
            )

    if fig is None:
        fig = go.Figure()
    with torch.no_grad():

        if named_fns is not None:
            for name, fn in named_fns.items():
                if isinstance(fn, torch.nn.Module):
                    fn.eval()

                fig.add_trace(
                    go.Scatter(
                        x=plot_x_axis.to("cpu").squeeze(),
                        y=fn(plot_x).to("cpu").squeeze(),
                        mode="lines",
                        name=name,
                    )
                )

        if samples is not None and dim == 1:
            fig.add_trace(
                go.Scatter(
                    x=samples[0].to("cpu").squeeze(),
                    y=samples[1].to("cpu").squeeze(),
                    mode="markers",
                    name="samples",
                )
            )

    if update_layout is not None:
        fig.update_layout(update_layout)
    if file is not None:
        fig.write_image(file)
    return fig


def sinusoidal_target(factor=5):
    """
    Sinusoidal target function.
    Based on https://arxiv.org/abs/2001.07523.
    """

    def target_fn(x):
        return torch.log(torch.sin(10 * factor * x) + 2) + torch.sin(factor * x)

    return target_fn


def setup(netrc_file=None):
    """
    Setup Weights & Biases.
    """
    try:
        netrc_config = netrc.netrc(netrc_file)
        if WANDB_HOST in netrc_config.hosts:
            os.environ[WANDB_API_KEY] = netrc_config.authenticators(WANDB_HOST)[2]

    except FileNotFoundError:
        pass

    if not os.environ.get(WANDB_API_KEY, NO_WANDB_API_KEY) == NO_WANDB_API_KEY:
        os.environ["WANDB_MODE"] = "run"
