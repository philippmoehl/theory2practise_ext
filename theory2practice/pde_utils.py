from abc import ABC, abstractmethod
import numpy as np

import torch


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
    Base class for different pdes.
    """

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def u0(self, nx):
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
    def to(self, dtype, device):
        pass


class ConvectionDiffusion(PDE):
    """
    nu: viscosity coefficient
    beta: wavespeed coefficient
    """

    def __init__(self, system, u0, params, source):
        self.system = system
        self._u0 = u0
        self.params = params
        self.source = source

        self._device = None
        self._dtype = None
        self.beta = 0.0
        self.nu = 0.0

        self.initialize()

    def initialize(self):
        # params
        if self.system == "convection":
            if "beta" in self.params:
                self.beta = self.params["beta"]
        elif self.system == "diffusion":
            if "nu" in self.params:
                self.nu = self.params["nu"]
        else:
            raise ValueError(
                "System needs to be chosen either 'convection' or 'diffusion'."
            )

    def u0(self, nx, x_lb=0.0, x_ub=2 * torch.pi):
        x_space = torch.arange(x_lb, x_ub, (x_ub - x_lb) / nx)
        x_space = x_space.to(device=self._device, dtype=self._dtype)

        return self._u0(x_space).view(-1, 1)

    def u(self, x, t, full_grid):
        u_full = self.solver(full_grid)

        x_t_ = torch.cat([x, t], dim=1)
        idx, idx_perm = torch.where((full_grid[:, None] == x_t_).all(dim=-1))
        # invert index permutation
        idx_inv = idx[torch.argsort(idx_perm)]
        return u_full[idx_inv]

    def solver(self, full_grid):
        """Calculate the u solution for convection/diffusion, assuming PBCs.
            Args:
            Returns:
            """
        x_mesh_stacked, t_mesh_stacked = torch.unbind(full_grid, dim=1)
        x_space = x_mesh_stacked.unique().sort()[0]

        t_mesh = t_mesh_stacked.view(-1, x_space.size(dim=0))
        uhat0 = torch.fft.fft(self._u0(x_space))
        g = torch.zeros_like(x_space) + self.source
        ikx = j_tensor_grid(x_space.size(dim=0))

        nu_factor = torch.exp(self.nu * ikx ** 2 * t_mesh - self.beta * ikx * t_mesh)
        a = uhat0 - torch.fft.fft(g) * 0  # at t=0, second term goes away
        uhat = a * nu_factor + torch.fft.fft(g) * t_mesh  # for constant, fft(p) dt = fft(p)*T

        u_full = torch.real(torch.fft.ifft(uhat))
        u_full.to(device=self._device, dtype=self._dtype)

        return u_full.flatten().view(-1, 1)

    def enforcer(self, u, x, t):
        """u_t - \nu * u_xx + \beta * u_x - G"""
        g = torch.full_like(u, self.source)
        return grad(u, t) - self.nu * grad(grad(u, x), x) + self.beta * grad(u, x) - g

    def to(self, dtype=None, device=None):
        """Sets the device and dtype."""
        self._device = device
        self._dtype = dtype


class Reaction(PDE):
    """
    rho
    """

    def __init__(self, u0, params):
        self._u0 = u0
        self.params = params

        self._device = None
        self._dtype = None
        self.rho = 0.0

        self.initialize()

    def initialize(self):
        # params
        if "rho" in self.params:
            self.rho = self.params["rho"]

    def u0(self, nx, x_lb=0.0, x_ub=2 * torch.pi):
        x_space = torch.arange(x_lb, x_ub, (x_ub - x_lb) / nx)
        x_space = x_space.to(device=self._device, dtype=self._dtype)

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
        return grad(u, t) - self.rho * u + self.rho * u ** 2

    def to(self, dtype=None, device=None):
        """Sets the device and dtype."""
        self._device = device
        self._dtype = dtype


class ReactionDiffusion(PDE):
    """
    nu: diffusion coefficient
    rho: reaction coefficient
    """

    def __init__(self, u0, params):
        self._u0 = u0
        self.params = params

        self._device = None
        self._dtype = None
        self.nu = 0.0
        self.rho = 0.0

        self.initialize()

    def initialize(self):
        # params
        if "nu" in self.params:
            self.nu = self.params["nu"]
        if "rho" in self.params:
            self.rho = self.params["rho"]

    def u0(self, nx, x_lb=0.0, x_ub=2 * torch.pi):
        x_space = torch.arange(x_lb, x_ub, (x_ub - x_lb) / nx)
        x_space = x_space.to(device=self._device, dtype=self._dtype)

        return self._u0(x_space).view(-1, 1)

    def u(self, x, t, full_grid):
        u_full = self.solver(full_grid)

        x_t_ = torch.cat([x, t], dim=1)
        idx, idx_perm = torch.where((full_grid[:, None] == x_t_).all(dim=-1))
        # invert index permutation
        idx_inv = idx[torch.argsort(idx_perm)]
        return u_full[idx_inv]

    def solver(self, full_grid):
        """ Computes the discrete solution of the reaction-diffusion PDE using
                pseudo-spectral operator splitting.
            Args:

            Returns:
            """
        x_mesh_stacked, t_mesh_stacekd = torch.unbind(full_grid, dim=1)
        x_space = x_mesh_stacked.unique().sort()[0]

        t_mesh = t_mesh_stacekd.view(-1, x_space.size(dim=0))
        u_full = torch.zeros_like(t_mesh.t())
        dt = 1 / t_mesh.size(dim=0)

        ikx = j_tensor_grid(x_space.size(dim=0))

        u_full[:, 0] = self._u0(x_space)
        u_ = self._u0(x_space)

        for i in range(t_mesh.size(dim=0) - 1):
            u_ = reaction(u_, 1, dt)

            u_ = diffusion(u_, 1, dt, ikx ** 2)
            u_full[:, i + 1] = u_

        u_full.to(device=self._device, dtype=self._dtype)

        return u_full.t().flatten().view(-1, 1)

    def enforcer(self, u, x, t):
        """u_t - \nu * u_xx - \rho * u + \rho * u**2"""
        return grad(u, t) - self.nu * grad(grad(u, x), x) - self.rho * u + self.rho * u ** 2

    def to(self, dtype=None, device=None):
        """Sets the device and dtype."""
        self._device = device
        self._dtype = dtype


def reaction(u, rho, dt):
    """ du/dt = rho*u*(1-u)
    """
    factor_1 = u * torch.exp(torch.tensor(rho * dt))
    factor_2 = (1 - u)
    u = factor_1 / (factor_2 + factor_1)
    return u


def diffusion(u, nu, dt, ikx2):
    """ du/dt = nu*d2u/dx2
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


class CurriculumScheduler:
    def __init__(self, pde, params, warmup_factor=0., start_factor=0., end_factor=1.0, warmup_iters=0, total_iters=0,
                 last_step=-1, verbose=False):

        if not isinstance(pde, PDE):
            raise TypeError(f'{type(pde).__name__} is not a PDE')
        self.pde = pde

        # TODO: param specific warmup and factors
        if not isinstance(params, list) and not isinstance(params, tuple):
            self.params = [params]
        else:
            # TODO: restructure PDE base class to have property "params"
            if not 0 < len(params) <= len(self.pde.params):
                raise ValueError(f"Expected at least 1 and at most {len(self.pde.params)} parameters, "
                                 f"but got {len(params)}")
            self.params = list(params)

        # Initialize epoch and base parameters
        if last_step == -1:
            for param in self.params:
                if not hasattr(pde, param):
                    raise KeyError(f"param {self.params} is not specified "
                                   f"in PDE {pde}")
                setattr(pde, f'initial_{param}', getattr(pde, param))
        else:
            for param in self.params:
                if not hasattr(pde, f"initial_{param}"):
                    raise KeyError(f"param initial_{param} is not specified "
                                   f"in PDE {pde} when resuming training")
        self.base_params = [getattr(pde, f"initial_{param}") for param in params]
        self.last_step = last_step

        # rates and steps
        if start_factor > 1.0 or start_factor < 0:
            raise ValueError(f"Starting multiplicative factor expected to be between 0 and 1, "
                             f"but got {start_factor}")

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError(f"Ending multiplicative factor expected to be between 0 and 1, "
                             f"but got {end_factor}")

        # warmup rates and steps
        if warmup_iters > total_iters:
            raise ValueError(f"warmup iterations expected to be below total iterations {total_iters}, "
                             f"but got {warmup_iters}")

        if warmup_iters > 0:
            if warmup_factor > max(start_factor, end_factor) or warmup_factor < min(start_factor, end_factor):
                raise ValueError(f"Warmup multiplicative factor expected to be between starting factor "
                                 f"{start_factor} and ending factor {end_factor}, "
                                 f"but got {warmup_factor}")

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters

        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor if (warmup_iters > 0) else start_factor

        self.verbose = verbose
        self._last_params = self.base_params

        self.step()

    def get_last_params(self):
        return self._last_params

    def get_params(self):

        if self.last_step == 0:
            return [base_param * self.start_factor
                    for base_param in self.base_params]

        if self.last_step > self.total_iters:
            return self.base_params

        if self.last_step <= self.warmup_iters:
            return [base_param * (self.start_factor + self.last_step *
                    (self.warmup_factor - self.start_factor) / self.warmup_iters )
                    for base_param in self.base_params]

        return [base_param * (self.warmup_factor + (self.last_step - self.warmup_iters) *
                (self.end_factor - self.warmup_factor)/(self.total_iters - self.warmup_iters))
                for base_param in self.base_params]

    @staticmethod
    def print_params(is_verbose, param, value):
        if is_verbose:
            # TODO: logging info level instead, if verbose
            print(f"Adjusting parameter"
                  f" {param} to {value:.2f}.")

    def step(self):
        self.last_step += 1
        values = self.get_params()

        for param, value in zip(self.params, values):
            setattr(self.pde, param, value)
            self.print_params(self.verbose, param, value)

        self._last_params = [getattr(self.pde, param) for param in self.params]
