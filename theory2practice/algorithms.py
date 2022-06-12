from abc import ABC, abstractmethod

import torch
from torch import nn

from . import pde_utils
from . import utils


class LevelNet(nn.Module):
    """
    Network for a single level.
    """

    def __init__(
        self,
        dim_in,
        dim,
        level,
        activation,
        normalization_factory=None,
        normalization_kwargs=None,
    ):
        super().__init__()
        self.level = level
        bias = normalization_factory is None
        self.dense_layers = nn.ModuleList([nn.Linear(dim_in, dim, bias=bias)])
        self.dense_layers += [
            nn.Linear(dim, dim, bias=bias) for _ in range(2**level - 1)
        ]
        self.dense_layers.append(nn.Linear(dim, 1))
        if normalization_factory is None:
            self.norm_layers = None
        else:
            normalization_kwargs = normalization_kwargs or {}
            self.norm_layers = nn.ModuleList(
                [
                    normalization_factory(dim, **normalization_kwargs)
                    for _ in range(2**level)
                ]
            )
        self.activation = activation

    def forward(self, tensor, res_tensors=None):
        out_tensors = []
        tensor = self.dense_layers[0](tensor)
        for i, dense in enumerate(self.dense_layers[1:]):
            if self.norm_layers is not None:
                tensor = self.norm_layers[i](tensor)
            tensor = self.activation(tensor)
            tensor = dense(tensor)
            if res_tensors:
                tensor = tensor + res_tensors[i]
            if i % 2 or self.level == 0:
                out_tensors.append(tensor)
        return out_tensors


class MultilevelNet(nn.Module):
    """
    Multilevel network.
    """

    def __init__(
        self,
        dim_in,
        factor,
        level,
        activation,
        normalization_factory=None,
        normalization_kwargs=None,
    ):
        super().__init__()
        self.nets = nn.ModuleList(
            [
                LevelNet(
                    dim_in,
                    factor * dim_in,
                    lvl,
                    activation,
                    normalization_factory,
                    normalization_kwargs,
                )
                for lvl in range(level)
            ]
        )

    def forward(self, tensor):
        res_tensors = None
        for net in self.nets[::-1]:
            res_tensors = net(tensor, res_tensors)
        return res_tensors[-1]


class FeedForward(nn.Module):
    """
    Feedforward network.
    """

    def __init__(
        self,
        activation,
        depth=None,
        width=None,
        input_dim=None,
        output_dim=1,
        arch=None,
        weight_init_factory=None,
        weight_init_kwargs=None,
        bias_init_factory=None,
        bias_init_kwargs=None,
        normalization_factory=None,
        normalization_kwargs=None,
        dropout=None,
        residual_connections=None,
    ):
        super().__init__()

        self.arch = arch
        if depth and width and input_dim and output_dim:
            self.arch = [input_dim] + (depth - 1) * [width] + [output_dim]
        if self.arch is None:
            raise ValueError(
                "Either an architecture `arch` or `depth`, `width`, `input_dim`, and `output_dim` need to be chosen!"
            )

        # affine linear layer
        bias = normalization_factory is None
        bias_init_kwargs = bias_init_kwargs or {}
        weight_init_kwargs = weight_init_kwargs or {}

        self.linear_layer = torch.nn.ModuleList(
            [
                nn.Linear(self.arch[i], self.arch[i + 1], bias=bias)
                for i in range(len(self.arch) - 2)
            ]
        )
        self.linear_layer.append(nn.Linear(self.arch[-2], self.arch[-1]))

        for linear in self.linear_layer:
            for tensor, init_factory, init_kwargs in zip(
                [linear.weight, linear.bias],
                [weight_init_factory, bias_init_factory],
                [weight_init_kwargs, bias_init_kwargs],
            ):
                if init_factory is not None:
                    init_factory(tensor, **init_kwargs)

        # activation function
        self.activation = activation

        # normalization layer
        if normalization_factory:
            normalization_kwargs = normalization_kwargs or {}
            self.norm_layer = torch.nn.ModuleList(
                [
                    normalization_factory(n, **normalization_kwargs)
                    for n in self.arch[1:-1]
                ]
            )
        else:
            self.norm_layer = None

        # dropout layer
        self.dropout = dropout

        # residual connections
        self.residual_connections = residual_connections or []

        if residual_connections is True:
            self.residual_connections = list(range(1, len(self.arch) - 2))
        elif any(self.arch[i] != self.arch[i + 1] for i in self.residual_connections):
            raise ValueError("Residual connections require the same dimension.")

    def forward(self, x):
        for i, linear in enumerate(self.linear_layer[:-1]):
            x = self.activation(linear(x))

            if i - 1 in self.residual_connections:
                x = x + res

            if i in self.residual_connections:
                res = x

            if self.dropout is not None:
                x = self.dropout(x)

            if self.norm_layer is not None:
                x = self.norm_layer[i](x)

        return self.linear_layer[-1](x)


class StandardizedModel(nn.Module):
    """
    Standardize the input using a given mean and stddev.
    """

    def __init__(
        self,
        model,
        mean,
        stddev,
    ):
        super().__init__()

        self.model = model
        self.mean = mean
        self.stddev = stddev

    def standardize(self, x):
        return (x - self.mean) / self.stddev

    def forward(self, x):
        x = self.standardize(x)
        return self.model(x)


class Algorithm(ABC):
    """
    Base class for different algorithms.
    """

    @abstractmethod
    def initialize(self, device, dtype, *args):
        pass

    @property
    @abstractmethod
    def model(self):
        pass

    @property
    @abstractmethod
    def samples(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @property
    @abstractmethod
    def save_attrs(self):
        pass


class GdAlgorithm(Algorithm):
    """
    Gradient descent algorithm.
    """

    def __init__(
        self,
        distribution_wrapper,
        model,
        epochs_per_iteration,
        data_loader_kwargs,
        loss,
        optimizer_factory,
        optimizer_kwargs,
        scheduler_factory=None,
        scheduler_kwargs=None,
        scheduler_step_frequency=1,
        scheduler_step_unit="batch",
        standardize=False,
        eval_keys=None,
    ):
        super().__init__()

        self._distribution_wrapper = distribution_wrapper
        self.raw_model = model

        # opt
        self._optimizer_factory = optimizer_factory
        self._optimizer_kwargs = optimizer_kwargs or {}
        self._scheduler_factory = scheduler_factory
        self._scheduler_kwargs = scheduler_kwargs or {}
        self._scheduler_step_frequency = scheduler_step_frequency
        if scheduler_step_unit not in ("batch", "epoch"):
            raise ValueError("`scheduler_step_unit` must be either `batch` or `epoch`.")
        self._scheduler_step_unit = scheduler_step_unit

        self._epochs_per_iteration = epochs_per_iteration
        self._data_loader_kwargs = data_loader_kwargs
        self._loss = loss
        self.metrics = utils.Metrics(losses=[self._loss])
        self._standardize = standardize
        self._eval_keys = eval_keys
        self._save_attrs = ["raw_model", "metrics", "optimizer"]

        # will be set in initialize method
        self.target_fn = None
        self.n_samples = None
        self._device = None
        self._dtype = None
        self.distribution = None
        self._model = None
        self.optimizer = None
        self.scheduler = None
        self._data_loader = None
        self._x = None
        self._y = None

        self._initialized = False

    def initialize(self, target_fn, n_samples, device="cpu", dtype=torch.float):

        self.target_fn = target_fn
        self.n_samples = n_samples
        self._device = device
        self._dtype = dtype

        if self._eval_keys is not None:
            for k in self._eval_keys:
                utils.nested_set(self, k, eval(utils.nested_get(self, k)))

        self._distribution_wrapper = self._distribution_wrapper.to(
            device=self._device, dtype=self._dtype
        )
        self.distribution = self._distribution_wrapper.get_distribution()

        if self._standardize:
            self.raw_model = StandardizedModel(
                self.raw_model,
                mean=self.distribution.mean,
                stddev=self.distribution.stddev,
            )

        self._model = utils.distribute(
            self.raw_model, device=self._device, dtype=self._dtype
        )
        self.optimizer = self._optimizer_factory(
            self._model.parameters(), **self._optimizer_kwargs
        )

        if self._scheduler_factory is not None:
            self.scheduler = self._scheduler_factory(
                self.optimizer, **self._scheduler_kwargs
            )
            self._save_attrs.append("scheduler")

        self._x = self.distribution.sample((self.n_samples,))
        self._y = self.target_fn(self._x)
        dataset = torch.utils.data.TensorDataset(self._x, self._y)
        self._data_loader = torch.utils.data.DataLoader(
            dataset, **self._data_loader_kwargs
        )
        self._initialized = True

    @property
    def samples(self):
        if not self._initialized:
            raise RuntimeError("Not initialized!")

        return self._x, self._y

    @property
    def model(self):
        if not self._initialized:
            raise RuntimeError("Not initialized!")

        return self._model

    @property
    def save_attrs(self):
        if not self._initialized:
            raise RuntimeError("Not initialized!")

        return self._save_attrs

    def run(self):
        if not self._initialized:
            raise RuntimeError("Not initialized!")

        self._model.train()
        for _ in range(self._epochs_per_iteration):
            self.metrics.zero()

            for x, y in self._data_loader:

                def closure():
                    self.optimizer.zero_grad()
                    loss = self._loss(prediction=self._model(x), y=y, store=False)
                    loss.backward()
                    return loss

                orig_closure_loss = self.optimizer.step(closure=closure)

                self._loss.store(loss=orig_closure_loss, batch_size=x.shape[0])
                self.metrics.step += 1

                if (
                    self.scheduler is not None
                    and self._scheduler_step_unit == "batch"
                    and self.metrics.step % self._scheduler_step_frequency == 0
                ):
                    self.scheduler.step()

            summary = self.metrics.finalize()

            if (
                self.scheduler is not None
                and self._scheduler_step_unit == "epoch"
                and self.metrics.epoch % self._scheduler_step_frequency == 0
            ):
                self.scheduler.step()

        summary["lr"] = self.optimizer.param_groups[0]["lr"]
        return summary


class PinnAlgorithm(Algorithm):
    """
    Gradient descent algorithm for Pinns.
    """

    def __init__(
        self,
        grid,
        model,
        epochs_per_iteration,
        data_loader_kwargs,
        loss,
        n_f,
        optimizer_factory,
        optimizer_kwargs,
        scheduler_factory=None,
        scheduler_kwargs=None,
        scheduler_step_frequency=1,
        scheduler_step_unit="batch",
        eval_keys=None,
    ):
        super().__init__()

        self.grid = grid
        self.raw_model = model

        # opt
        self._optimizer_factory = optimizer_factory
        self._optimizer_kwargs = optimizer_kwargs or {}
        self._scheduler_factory = scheduler_factory
        self._scheduler_kwargs = scheduler_kwargs or {}
        self._scheduler_step_frequency = scheduler_step_frequency
        if scheduler_step_unit not in ("batch", "epoch"):
            raise ValueError("`scheduler_step_unit` must be either `batch` or `epoch`.")
        self._scheduler_step_unit = scheduler_step_unit

        self._epochs_per_iteration = epochs_per_iteration
        self._data_loader_kwargs = data_loader_kwargs
        self._loss = loss
        self.n_f = n_f
        self.metrics = utils.Metrics(losses=[self._loss])
        self._eval_keys = eval_keys
        self._save_attrs = ["raw_model", "metrics", "optimizer"]

        # will be set in initialize method
        self.pde = None
        self._device = None
        self._dtype = None
        self._model = None
        self.optimizer = None
        self.scheduler = None
        self._data_loader = None
        self._ic = None
        self._u_ic = None
        self._bc_lb = None
        self._bc_ub = None
        self._x_f = None
        self._t_f = None

        self._initialized = False

    def initialize(self, pde, device="cpu", dtype=torch.float):

        self.pde = pde
        self._device = device
        self._dtype = dtype

        if self._eval_keys is not None:
            for k in self._eval_keys:
                utils.nested_set(self, k, eval(utils.nested_get(self, k)))

        self.grid.to(
            device=self._device, dtype=self._dtype
        )

        self._model = utils.distribute(
            self.raw_model, device=self._device, dtype=self._dtype
        )
        self.optimizer = self._optimizer_factory(
            self._model.parameters(), **self._optimizer_kwargs
        )

        if self._scheduler_factory is not None:
            self.scheduler = self._scheduler_factory(
                self.optimizer, **self._scheduler_kwargs
            )
            self._save_attrs.append("scheduler")

        self._x_f, self._t_f = self.grid.sample_f_data(self.n_f)
        self._ic, self._bc_lb, self._bc_ub = self.grid.x()
        self._u_ic = self.pde.u0(self.grid.nx, self.grid.x_lb, self.grid.x_ub)

        dataset = pde_utils.PdeDataset()
        self._data_loader = torch.utils.data.DataLoader(
            dataset, **self._data_loader_kwargs
        )
        self._initialized = True

    @property
    def samples(self):
        if not self._initialized:
            raise RuntimeError("Not initialized!")

        return self._x_f, self._t_f

    @property
    def model(self):
        if not self._initialized:
            raise RuntimeError("Not initialized!")

        return self._model

    @property
    def save_attrs(self):
        if not self._initialized:
            raise RuntimeError("Not initialized!")

        return self._save_attrs

    def run(self):
        if not self._initialized:
            raise RuntimeError("Not initialized!")

        self._model.train()
        for _ in range(self._epochs_per_iteration):
            self.metrics.zero()

            for _ in self._data_loader:

                def closure():
                    self.optimizer.zero_grad()
                    # enforce
                    u_pred_f = self._model(torch.cat([self._x_f, self._t_f], dim=1))
                    f_pred = self.pde.enforcer(u_pred_f, self._x_f, self._t_f)

                    predictions = {
                        "u_pred_ic": self._model(self._ic),
                        "u_pred_lb": self._model(self._bc_lb),
                        "u_pred_ub": self._model(self._bc_ub),
                        "f_pred": f_pred,
                    }
                    loss = self._loss(prediction=predictions, y=self._u_ic, store=False)
                    loss.backward()
                    return loss

                orig_closure_loss = self.optimizer.step(closure=closure)

                self._loss.store(loss=orig_closure_loss, batch_size=1)
                self.metrics.step += 1

                if (
                    self.scheduler is not None
                    and self._scheduler_step_unit == "batch"
                    and self.metrics.step % self._scheduler_step_frequency == 0
                ):
                    self.scheduler.step()

            summary = self.metrics.finalize()

            if (
                self.scheduler is not None
                and self._scheduler_step_unit == "epoch"
                and self.metrics.epoch % self._scheduler_step_frequency == 0
            ):
                self.scheduler.step()

        summary["lr"] = self.optimizer.param_groups[0]["lr"]
        return summary
