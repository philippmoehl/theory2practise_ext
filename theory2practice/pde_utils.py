from abc import ABC, abstractmethod
from functools import partial

from omegaconf.listconfig import ListConfig

import torch
from torchdiffeq import odeint


def create_tensor_spaces(nx, nt, x_lb, x_ub, t_min, t_max):
    """Grid of space, time."""
    x_space = torch.linspace(x_lb, x_ub, steps=nx+1)
    t_space = torch.linspace(t_min, t_max, steps=nt+1)
    return x_space, t_space


class TensorGrid:
    """
    Grid data preprocessor.
    """

    def __init__(self, nx, nt, x_lb=0., x_ub=2 * torch.pi, t_min=0.,
                 t_max=1.):
        self.nx = nx
        self.nt = nt
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.t_min = t_min
        self.t_max = t_max

        self.x_space = None
        self.t_space = None
        self._device = None
        self._dtype = None

        self.initialize()

    def initialize(self):
        self.x_space = torch.linspace(self.x_lb, self.x_ub, steps=self.nx + 1)
        self.t_space = torch.linspace(self.t_min, self.t_max, steps=self.nt + 1)

    def sample(self, n_samples):
        """No_initial_no_boundary, requires_grad for enforcing pde."""

        x_inner_mesh = self.x_mesh[1:, 1:-1].flatten().view(-1, 1)
        t_inner_mesh = self.t_mesh[1:, 1:-1].flatten().view(-1, 1)

        idx = torch.randperm(x_inner_mesh.size()[0])[:n_samples]

        x_mesh_sample = x_inner_mesh[idx].to(device=self._device,
                                             dtype=self._dtype)
        t_mesh_sample = t_inner_mesh[idx].to(device=self._device,
                                             dtype=self._dtype)

        return x_mesh_sample, t_mesh_sample

    def conds(self):
        # initial condition, from x = [-end, +end] and t=0
        ic = torch.cat([self.x_mesh[:1, :-1].t(), self.t_mesh[:1, :-1].t()],
                       dim=1)
        # boundary condition at x = start, and t = [0, 1]
        bc_lb = torch.cat([self.x_mesh[:, :1], self.t_mesh[:, :1]], dim=1)
        # at x = end
        bc_ub = torch.cat([self.x_mesh[:, -1:], self.t_mesh[:, -1:]], dim=1)

        ic = ic.to(device=self._device, dtype=self._dtype)
        bc_lb = bc_lb.to(device=self._device, dtype=self._dtype)
        bc_ub = bc_ub.to(device=self._device, dtype=self._dtype)

        return ic, bc_lb, bc_ub

    def full_grid(self):
        full_grid = torch.cat([
            self.x_mesh[:, :-1].flatten().view(-1, 1),
            self.t_mesh[:, :-1].flatten().view(-1, 1)
        ], dim=1)

        full_grid = full_grid.to(device=self._device, dtype=self._dtype)

        return full_grid

    @property
    def x_mesh(self):
        return self.x_space.view(1, -1).expand(self.nt + 1, self.nx + 1)

    @property
    def t_mesh(self):
        return self.t_space.view(-1, 1).expand(self.nt + 1, self.nx + 1)

    @property
    def x_axis(self):
        return self.x_space[:-1]

    @property
    def t_axis(self):
        return self.t_space

    def to(self, device, dtype):
        """Sets the device and dtype."""
        self._device = device
        self._dtype = dtype


def grad(u, argument):
    """Autograd of u wrt argument."""
    return torch.autograd.grad(
            u, argument,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]


class PDE(ABC):
    """
    PDE base class.
    """

    @abstractmethod
    def initialize(self):
        pass

    @property
    @abstractmethod
    def params(self):
        pass

    @abstractmethod
    def u0(self, x_space):
        pass

    @abstractmethod
    def u(self, x, t, full_grid):
        pass

    @abstractmethod
    def solver(self, full_grid):
        pass

    @abstractmethod
    def enforcer(self, u, x, t):
        pass

    @abstractmethod
    def to(self, device, dtype):
        pass


class ConvectionDiffusion(PDE):
    """
    Convection or Diffusion PDEs.

    Parameters:
      - nu: viscosity coefficient
      - beta: wavespeed coefficient
    """

    def __init__(self, system, u0, params, source):
        self.system = system
        self._u0 = u0
        self._params = params
        self.source = source

        self._device = None
        self._dtype = None
        self.beta = 0.0
        self.nu = 0.0

        self.initialize()

    def initialize(self):
        # params
        if self.system == "convection":
            if "beta" in self._params:
                self.beta = self._params["beta"]
        elif self.system == "diffusion":
            if "nu" in self._params:
                self.nu = self._params["nu"]
        else:
            raise ValueError(
                "System needs to be chosen either 'convection' or 'diffusion'."
            )

    def params(self):
        return self._params

    def u0(self, x_space):
        return self._u0(x_space).view(-1, 1)

    def u(self, x, t, full_grid):
        u_full = self.solver(full_grid)

        x_t_ = torch.cat([x, t], dim=1)
        idx, idx_perm = torch.where((full_grid[:, None] == x_t_).all(dim=-1))
        # invert index permutation
        idx_inv = idx[torch.argsort(idx_perm)]
        return u_full[idx_inv]

    def solver(self, full_grid):
        x_mesh_stacked, t_mesh_stacked = torch.unbind(full_grid, dim=1)
        x_space = x_mesh_stacked.unique().sort()[0]

        t_mesh = t_mesh_stacked.view(-1, x_space.size(dim=0))
        u_hat_0 = torch.fft.fft(self._u0(x_space))
        g = torch.zeros_like(x_space) + self.source
        ikx = j_tensor_grid(x_space.size(dim=0))

        nu_factor = torch.exp(self.nu * ikx ** 2 * t_mesh
                              - self.beta * ikx * t_mesh)
        a = u_hat_0 - torch.fft.fft(g) * 0  # at t=0, second term goes away
        # for constant, fft(p) dt = fft(p)*T
        u_hat = a * nu_factor + torch.fft.fft(g) * t_mesh

        u_full = torch.real(torch.fft.ifft(u_hat))
        u_full.to(device=self._device, dtype=self._dtype)

        return u_full.flatten().view(-1, 1)

    def enforcer(self, u, x, t):
        """u_t - \nu * u_xx + \beta * u_x - G"""
        g = torch.full_like(u, self.source)
        u_t = grad(u, t)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)

        return u_t - self.nu * u_xx + self.beta * u_x - g

    def to(self, device=None, dtype=None):
        self._device = device
        self._dtype = dtype


class Burger(PDE):
    """
    Burger PDE.

    Parameters:
      - nu: viscosity coefficient
    """

    def __init__(self, u0, params):
        self._u0 = u0
        self._params = params

        self._device = None
        self._dtype = None
        self.nu = 0.0

        self.initialize()

    def initialize(self):
        # params
        if "nu" in self._params:
            self.nu = self._params["nu"]

    def params(self):
        return self._params

    def u0(self, x_space):
        return self._u0(x_space).view(-1, 1)

    def u(self, x, t, full_grid):
        u_full = self.solver(full_grid)

        x_t_ = torch.cat([x, t], dim=1)
        idx, idx_perm = torch.where((full_grid[:, None] == x_t_).all(dim=-1))
        # invert index permutation
        idx_inv = idx[torch.argsort(idx_perm)]
        return u_full[idx_inv]

    def solver(self, full_grid):
        x_mesh_stacked, t_mesh_stacked = torch.unbind(full_grid, dim=1)
        x_space = x_mesh_stacked.unique().sort()[0]
        t_space = t_mesh_stacked.unique().sort()[0]

        dx = x_space[1] - x_space[0]
        n = 2 * torch.pi * torch.fft.fftfreq(x_space.size(dim=0), d=dx)
        u0 = self._u0(x_space)

        burger_partial = partial(burger, n=n, nu=self.nu)
        u_full = odeint(burger_partial, u0, t_space)
        u_full.to(device=self._device, dtype=self._dtype)

        return u_full.flatten().view(-1, 1)

    def enforcer(self, u, x, t):
        """u_t - \nu * u_xx + \beta * u_x - G"""
        u_t = grad(u, t)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)

        return u_t + u * u_x - self.nu * u_xx

    def to(self, device=None, dtype=None):
        self._device = device
        self._dtype = dtype


class Reaction(PDE):
    """
    Reaction PDE.

    Parameters:
      - rho
    """

    def __init__(self, u0, params):
        self._u0 = u0
        self._params = params

        self._device = None
        self._dtype = None
        self.rho = 0.0

        self.initialize()

    def initialize(self):
        # params
        if "rho" in self._params:
            self.rho = self._params["rho"]

    def params(self):
        return self._params

    def u0(self, x_space):
        return self._u0(x_space).view(-1, 1)

    def u(self, x, t, full_grid):
        u_full = self.solver(full_grid)

        x_t_ = torch.cat([x, t], dim=1)
        idx, idx_perm = torch.where((full_grid[:, None] == x_t_).all(dim=-1))
        # invert index permutation
        idx_inv = idx[torch.argsort(idx_perm)]
        return u_full[idx_inv]

    def solver(self, full_grid):
        x_mesh_stacked, t_mesh_stacked = torch.unbind(full_grid, dim=1)
        x_space = x_mesh_stacked.unique().sort()[0]
        t_mesh = t_mesh_stacked.view(-1, x_space.size(dim=0))

        u = reaction(self._u0(x_space), self.rho, t_mesh)
        u.to(device=self._device, dtype=self._dtype)

        return u.flatten().view(-1, 1)

    def enforcer(self, u, x, t):
        """u_t - \rho * u + \rho * u**2"""
        u_t = grad(u, t)

        return u_t - self.rho * u + self.rho * u ** 2

    def to(self, device=None, dtype=None):
        self._device = device
        self._dtype = dtype


class ReactionDiffusion(PDE):
    """
    Reaction PDE.

    Parameters:
      - nu: diffusion coefficient
      - rho: reaction coefficient
    """

    def __init__(self, u0, params):
        self._u0 = u0
        self._params = params

        self._device = None
        self._dtype = None
        self.nu = 0.0
        self.rho = 0.0

        self.initialize()

    def initialize(self):
        # params
        if "nu" in self._params:
            self.nu = self._params["nu"]
        if "rho" in self._params:
            self.rho = self._params["rho"]

    def params(self):
        return self._params

    def u0(self, x_space):
        return self._u0(x_space).view(-1, 1)

    def u(self, x, t, full_grid):
        u_full = self.solver(full_grid)

        x_t_ = torch.cat([x, t], dim=1)
        idx, idx_perm = torch.where((full_grid[:, None] == x_t_).all(dim=-1))
        # invert index permutation
        idx_inv = idx[torch.argsort(idx_perm)]
        return u_full[idx_inv]

    def solver(self, full_grid):
        x_mesh_stacked, t_mesh_stacekd = torch.unbind(full_grid, dim=1)
        x_space = x_mesh_stacked.unique().sort()[0]

        t_mesh = t_mesh_stacekd.view(-1, x_space.size(dim=0))
        u_full = torch.zeros_like(t_mesh.t())
        dt = (1 / t_mesh.size(dim=0))

        ikx = j_tensor_grid(x_space.size(dim=0))

        u_full[:, 0] = self._u0(x_space)
        u_ = self._u0(x_space)

        for i in range(t_mesh.size(dim=0) - 1):
            u_ = reaction(u_, self.rho, dt)

            u_ = diffusion(u_, self.nu, dt, ikx ** 2)
            u_full[:, i + 1] = u_

        u_full.to(device=self._device, dtype=self._dtype)

        return u_full.t().flatten().view(-1, 1)

    def enforcer(self, u, x, t):
        """u_t - \nu * u_xx - \rho * u + \rho * u**2"""
        u_t = grad(u, t)
        u_xx = grad(grad(u, x), x)

        return u_t - self.nu * u_xx - self.rho * u + self.rho * u ** 2

    def to(self, device=None, dtype=None):
        self._device = device
        self._dtype = dtype


def reaction(u, rho, dt):
    """
    du/dt = rho*u*(1-u)
    """
    factor_1 = u * torch.exp(rho * dt)
    factor_2 = (1 - u)
    u = factor_1 / (factor_2 + factor_1)
    return u


def diffusion(u, nu, dt, ikx2):
    """
    du/dt = nu*d2u/dx2
    """
    factor = torch.exp(nu * ikx2 * dt)
    u_hat = torch.fft.fft(u)
    u_hat *= factor
    u = torch.real(torch.fft.ifft(u_hat))
    return u


def j_tensor_grid(nx):
    ikx_pos = 1j * torch.arange(0, nx/2+1, 1)
    ikx_neg = 1j * torch.arange(-nx/2+1, 0, 1)
    return torch.cat((ikx_pos, ikx_neg))


def burger(t, u0, n, nu):
    u_hat_0 = torch.fft.fft(u0)
    u_hat_x = 1j * n * u_hat_0
    u_hat_xx = -n ** 2 * u_hat_0

    u_x = torch.fft.ifft(u_hat_x)
    u_xx = torch.fft.ifft(u_hat_xx)

    u_t = -u0 * u_x + nu * u_xx
    return torch.real(u_t)


class Scheduler(ABC):
    """
    Scheduler base class.
    """

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def print_step(self, is_verbose):
        pass


class CurriculumScheduler(Scheduler):
    def __init__(
            self, pde, params, warmup_factor=0., start_factor=0., end_factor=1.,
            warmup_iters=0, total_iters=0, last_step=-1, verbose=False):

        if not isinstance(pde, PDE):
            raise TypeError(f"{type(pde).__name__} is not a PDE")
        self.pde = pde

        # TODO: param specific warmup and factors
        if not isinstance(params, (list, tuple, ListConfig)):
            self.params = [params]
        else:
            if not 0 < len(params) <= len(self.pde.params):
                raise ValueError(f"Expected at least 1 and at most "
                                 f"{len(self.pde.params)} parameters, "
                                 f"but got {len(params)}")
            self.params = list(params)

        # Initialize epoch and base parameters
        if last_step == -1:
            for param in self.params:
                if not hasattr(pde, param):
                    raise KeyError(f"param {param} is not specified "
                                   f"in PDE {type(pde).__name__}")
                setattr(pde, f"initial_{param}", getattr(pde, param))
        else:
            for param in self.params:
                if not hasattr(pde, f"initial_{param}"):
                    raise KeyError(f"param initial_{param} is not specified "
                                   f"in PDE {type(pde).__name__} "
                                   f"when resuming training")
        self.base_params = [getattr(pde, f"initial_{param}")
                            for param in params]
        self.last_step = last_step

        # rates and steps
        if start_factor > 1.0 or start_factor < 0:
            raise ValueError(f"Starting multiplicative factor expected to be "
                             f"between 0 and 1, but got {start_factor}")

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError(f"Ending multiplicative factor expected to be "
                             f"between 0 and 1, but got {end_factor}")

        # warmup rates and steps
        if warmup_iters > total_iters:
            raise ValueError(f"warmup iterations expected to be below "
                             f"total iterations {total_iters}, "
                             f"but got {warmup_iters}")

        if warmup_iters > 0:
            if (
                warmup_factor > max(start_factor, end_factor) or
                warmup_factor < min(start_factor, end_factor)
            ):
                raise ValueError(f"Warmup multiplicative factor expected to be "
                                 f"between starting factor {start_factor} and "
                                 f"ending factor {end_factor}, "
                                 f"but got {warmup_factor}")

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters

        if warmup_iters > 0:
            self.warmup_factor = warmup_factor
        else:
            self.warmup_factor = start_factor
        self.warmup_iters = warmup_iters

        self.verbose = verbose

        self.step()

    def get_params(self):

        if self.last_step == 0:
            return [base_param * self.start_factor
                    for base_param in self.base_params]

        if self.last_step > self.total_iters:
            return self.base_params

        if self.last_step <= self.warmup_iters:
            factor_diff = self.warmup_factor - self.start_factor
            rel_factor = factor_diff / self.warmup_iters
            return [
                base_param * (self.start_factor + self.last_step * rel_factor)
                for base_param in self.base_params]

        factor_diff = self.end_factor - self.warmup_factor
        rel_factor = factor_diff/(self.total_iters - self.warmup_iters)
        rem_steps = self.last_step - self.warmup_iters
        return [
            base_param * (self.warmup_factor + rem_steps * rel_factor)
            for base_param in self.base_params]

    def print_step(self, is_verbose):
        if is_verbose:
            values = self.get_params()
            for param, value in zip(self.params, values):
                print(f"Adjusting parameter"
                      f" {param} to {value:.2f}.")

    def step(self):
        self.last_step += 1
        values = self.get_params()

        for param, value in zip(self.params, values):
            setattr(self.pde, param, value)
        self.print_step(self.verbose)


class GridScheduler(Scheduler):
    def __init__(self, grid, splits, axis="t", last_step=-1,
                 verbose=False):

        if not isinstance(grid, TensorGrid):
            raise TypeError(f"{type(grid).__name__} is not a grid")
        self.grid = grid

        if isinstance(axis, str):
            self.axis = axis
        elif isinstance(axis, (list, tuple)):
            raise TypeError(f"axis is given as {type(axis).__name__}, "
                            f"but only one axis is allowed to be split")
        else:
            raise TypeError(f"{type(axis).__name__} is not a string")
        space = f"{axis}_space"

        # Initialize epoch and full_grid
        if last_step == -1:
            if not hasattr(grid, space):
                raise KeyError(f"space {space} is not specified "
                               f"in grid {type(grid).__name__}")
            setattr(grid, f"initial_{space}", getattr(grid, space))
            setattr(grid, "initial_full_grid", grid.full_grid())
        else:
            if not hasattr(grid, f"initial_full_grid"):
                raise KeyError(f"initial_full_grid is not specified "
                               f"in grid {type(grid).__name__} when resuming "
                               f"training, but is needed for multiple purposes "
                               f"among other things, testing")
            if not hasattr(grid, f"initial_{space}"):
                raise KeyError(f"initial_{space} is not specified in grid "
                               f"{type(grid).__name__} when resuming training")
        self.space = space
        self.last_step = last_step

        # n_axis, axis dimension,
        n_axis = f"n{axis}"
        if not hasattr(grid, n_axis):
            raise KeyError(f"axis length counter {n_axis} is not specified "
                           f"in grid {type(grid).__name__}")

        # splits
        if splits < 1:
            raise ValueError(f"expected at least 1 split, but got {splits}")
        if getattr(self.grid, n_axis) % splits != 0:
            raise ValueError(f"expected length of axis "
                             f"{getattr(self.grid, n_axis)}, "
                             f"be multiple of number of splits {splits}")
        self._n = int(getattr(self.grid, n_axis) / splits)
        setattr(grid, n_axis, self._n)
        self.splits = splits
        self.verbose = verbose
        self.step()

    def print_step(self, is_verbose):
        if is_verbose:
            print(f"Adjusting to grid split"
                  f" {self.last_step + 1} of {self.splits} splits.")

    def get_space(self):
        initial_space = getattr(self.grid, f"initial_{self.space}")

        if self.last_step < self.splits:
            return initial_space[(self._n * self.last_step):(
                        self._n * (self.last_step + 1) + 1)]

        return initial_space[(self._n * (self.splits - 1)):]

    def step(self):
        self.last_step += 1
        space = self.get_space()

        setattr(self.grid, self.space, space)
        self.print_step(self.verbose)
