from abc import ABC, abstractmethod
import numpy as np
import torch

PDE_PARAMS = {
    'convection': ['beta'],
    "diffusion": ['nu'],
    "reaction": ['rho'],
    "rd": ['rho', 'nu'],
}

CONDS = ["x_ic", "t_ic", "x_bc_lb", "t_bc_lb", "x_bc_ub", "t_bc_ub", "x_f", "t_f", "u_ic"]


class PdeDataset(torch.utils.data.Dataset):
    """Pde dataset."""

    def __init__(self):
        """
        Args:
            needed pde components.
        """
        pass

    def __len__(self):
        """Length for this is one, as no batching on these inputs is done."""
        return 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return None


def create_tensor_grid(nx, nt, x_lb, x_ub, t_min, t_max):
    """Grid of space, time."""
    _X = torch.linspace(x_lb, x_ub, steps=nx + 1)
    _T = torch.linspace(t_min, t_max, steps=nt)
    return torch.meshgrid(_X, _T, indexing='xy')


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
    def u0(self, x_grid):
        pass

    @abstractmethod
    def solver(self, _x_star):
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

    def __init__(self, system, _u0, params, source):
        self.system = system
        self._u0 = _u0
        self.params = params
        self.source = source

        self.device = None
        self.dtype = None
        self.beta = 0.0
        self.nu = 0.0

        self.initialize()

    def initialize(self):
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
        _X = torch.arange(x_lb, x_ub, (x_ub - x_lb) / nx)
        _X = _X.to(device=self.device, dtype=self.dtype)

        return self._u0(_X).view(-1, 1)

    def solver(self, _x_star):
        """Calculate the u solution for convection/diffusion, assuming PBCs.
            Args:
            Returns:
            """
        stacked_X, stacked_T = torch.unbind(_x_star, dim=1)
        _X = stacked_X.unique().sort()[0]

        T = stacked_T.view(-1, _X.size(dim=0))
        uhat0 = torch.fft.fft(self._u0(_X))
        G = torch.zeros_like(_X) + self.source
        IKX = j_tensor_grid(_X.size(dim=0))

        nu_factor = torch.exp(self.nu * IKX ** 2 * T - self.beta * IKX * T)
        A = uhat0 - torch.fft.fft(G) * 0  # at t=0, second term goes away
        uhat = A * nu_factor + torch.fft.fft(G) * T  # for constant, fft(p) dt = fft(p)*T

        u = torch.real(torch.fft.ifft(uhat))
        u.to(device=self.device, dtype=self.dtype)

        return u.flatten().view(-1, 1)

    def enforcer(self, u, x, t):
        G = torch.full_like(u, self.source)
        return grad(u, t) - self.nu * grad(grad(u, x), x) + self.beta * grad(u, x) - G

    def to(self, dtype=None, device=None):
        """Sets the device and dtype."""
        self.device = device
        self.dtype = dtype


class Reaction(PDE):
    """
    rho
    """

    def __init__(self, _u0, params):
        self._u0 = _u0
        self.params = params

        self.device = None
        self.dtype = None
        self.rho = 0.0

        self.initialize()

    def initialize(self):
        if "rho" in self.params:
            self.rho = self.params["rho"]

    def u0(self, nx, x_lb=0.0, x_ub=2 * torch.pi):
        _X = torch.arange(x_lb, x_ub, (x_ub - x_lb) / nx)
        _X = _X.to(device=self.device, dtype=self.dtype)

        return self._u0(_X).view(-1, 1)

    def solver(self, _x_star):
        stacked_X, stacked_T = torch.unbind(_x_star, dim=1)
        _X = stacked_X.unique().sort()[0]
        T = stacked_T.view(-1, _X.size(dim=0))

        u = reaction(self._u0(_X), self.rho, T)
        u.to(device=self.device, dtype=self.dtype)

        return u.flatten().view(-1, 1)

    def enforcer(self, u, x, t):
        return grad(u, t) - self.rho * u + self.rho * u ** 2

    def to(self, dtype=None, device=None):
        """Sets the device and dtype."""
        self.device = device
        self.dtype = dtype


class ReactionDiffusion(PDE):
    """
    nu: diffusion coefficient
    rho: reaction coefficient
    """

    def __init__(self, _u0, params):
        self._u0 = _u0
        self.params = params

        self.device = None
        self.dtype = None
        self.nu = 0.0
        self.rho = 0.0

        self.initialize()

    def initialize(self):
        if "nu" in self.params:
            self.nu = self.params["nu"]
        if "rho" in self.params:
            self.rho = self.params["rho"]

    def u0(self, nx, x_lb=0.0, x_ub=2 * torch.pi):
        _X = torch.arange(x_lb, x_ub, (x_ub - x_lb) / nx)
        _X = _X.to(device=self.device, dtype=self.dtype)

        return self._u0(_X).view(-1, 1)

    def solver(self, _x_star):
        """ Computes the discrete solution of the reaction-diffusion PDE using
                pseudo-spectral operator splitting.
            Args:

            Returns:
            """
        stacked_X, stacked_T = torch.unbind(_x_star, dim=1)
        _X = stacked_X.unique().sort()[0]

        T = stacked_T.view(-1, _X.size(dim=0))
        u = torch.zeros_like(T.t())
        dt = 1 / T.size(dim=0)

        IKX = j_tensor_grid(_X.size(dim=0))

        u[:, 0] = self._u0(_X)
        u_ = self._u0(_X)

        for i in range(T.size(dim=0) - 1):
            u_ = reaction(u_, 1, dt)

            u_ = diffusion(u_, 1, dt, IKX ** 2)
            u[:, i + 1] = u_

        u.to(device=self.device, dtype=self.dtype)

        return u.t().flatten().view(-1, 1)

    def enforcer(self, u, x, t):
        return grad(u, t) - self.nu * grad(grad(u, x), x) - self.rho * u + self.rho * u ** 2

    def to(self, dtype=None, device=None):
        """Sets the device and dtype."""
        self.device = device
        self.dtype = dtype


def reaction(u, rho, dt):
    """ du/dt = rho*u*(1-u)
    """
    factor_1 = u * torch.exp(torch.tensor(rho * dt))
    factor_2 = (1 - u)
    u = factor_1 / (factor_2 + factor_1)
    return u


def diffusion(u, nu, dt, IKX2):
    """ du/dt = nu*d2u/dx2
    """
    factor = torch.exp(nu * IKX2 * dt)
    u_hat = torch.fft.fft(u)
    u_hat *= factor
    u = torch.real(torch.fft.ifft(u_hat))
    return u


def j_tensor_grid(nx):
    IKX_pos = 1j * torch.arange(0, nx/2+1, 1)
    IKX_neg = 1j * torch.arange(-nx/2+1, 0, 1)
    return torch.cat((IKX_pos, IKX_neg))
