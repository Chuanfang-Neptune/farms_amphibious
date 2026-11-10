"""Network"""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from scipy import integrate
from scipy.integrate._ode import ode as ODE

from farms_core.model.data import AnimatData
from farms_core import pylog

from .ode import ode_oscillators_sparse


class AnimatNetwork(ABC):
    """Animat network"""

    def __init__(self, data, n_iterations):
        super().__init__()
        self.data: AnimatData = data
        self.n_iterations = n_iterations

    @abstractmethod
    def step(
            self,
            iteration: int,
            time: float,
            timestep: float,
            **kwargs,
    ):
        """Step function called at each simulation iteration"""


class NetworkODE(AnimatNetwork):
    """NetworkODE"""

    def __init__(self, data, integrator='dopri5', **kwargs):
        state_array = data.state.array
        self.modulo: int = kwargs.pop('modulo', 1)
        super().__init__(data=data, n_iterations=np.shape(state_array)[0])
        self.dstate = np.zeros_like(data.state.array[0, :])
        self.ode: Callable = kwargs.pop('ode', ode_oscillators_sparse)
        self.integrator = integrator
        self.integrator_kwargs = kwargs
        if integrator != 'rk4':
            self.solver: ODE = integrate.ode(f=self.ode)
        self.initialize_episode()

    def initialize_episode(self):
        """Initialize episode"""
        if self.integrator == 'rk4':
            # Integrator time and state, and work arrays for the RK4 stages
            self.solver = None
            self._t = 0.0
            self._y = np.array(self.data.state.array[0, :], dtype=np.float64)
            self._tmp = np.zeros_like(self._y)
            self._k = [np.zeros_like(self._y) for _ in range(4)]
        else:
            self.solver: ODE = integrate.ode(f=self.ode)
            self.solver.set_integrator(self.integrator, **self.integrator_kwargs)
            self.solver.set_initial_value(y=self.data.state.array[0, :], t=0.0)
        self.data.state.array[1:, :] = 0
        self.dstate[:] = 0

    def _integrate_rk4(self, iteration: int, time: float, timestep: float):
        """Integrate the state with fixed-step classical RK4

        Integrates from the integrator time to time + timestep in equal
        steps no longer than timestep, calling self.ode directly. Does
        nothing if the integrator time is already at least
        time + 0.99*timestep, as in the scipy integrator path.
        """
        if self._t >= time + 0.99*timestep:
            return
        target = time + timestep
        n_steps = max(1, int(np.ceil((target - self._t)/timestep - 1e-9)))
        h = (target - self._t)/n_steps
        y, tmp, dstate, data, ode = self._y, self._tmp, self.dstate, self.data, self.ode
        k1, k2, k3, k4 = self._k
        t = self._t
        for _ in range(n_steps):
            ode(t, y, dstate, iteration, data)
            k1[:] = dstate
            np.multiply(k1, 0.5*h, out=tmp)
            tmp += y
            ode(t + 0.5*h, tmp, dstate, iteration, data)
            k2[:] = dstate
            np.multiply(k2, 0.5*h, out=tmp)
            tmp += y
            ode(t + 0.5*h, tmp, dstate, iteration, data)
            k3[:] = dstate
            np.multiply(k3, h, out=tmp)
            tmp += y
            ode(t + h, tmp, dstate, iteration, data)
            k4[:] = dstate
            # y += h/6*(k1 + 2 k2 + 2 k3 + k4)
            np.multiply(k2, 2.0, out=tmp)
            tmp += k1
            np.multiply(k3, 2.0, out=k2)
            tmp += k2
            tmp += k4
            tmp *= h/6.0
            y += tmp
            t += h
        self._t = target

    def copy_next_drive(self, iteration):
        """Set initial drive"""
        array = self.data.network.drives.array
        array[iteration+1] = array[iteration]

    def step(
            self,
            iteration: int,
            time: float,
            timestep: float,
            checks: bool = False,
            strict: bool = False,
    ):
        """Control step

        Called after obtaining all the current sensor data, and right before
        calling the physics.

        """
        if iteration == 0:
            self.copy_next_drive(iteration)
            return
        if self.integrator == 'rk4':
            if iteration % self.modulo:
                self.data.state.array[iteration, :] = (
                    self.data.state.array[iteration-1, :]
                )
            else:
                self._integrate_rk4(iteration, time, timestep)
                self.data.state.array[iteration, :] = self._y
            if iteration < self.n_iterations-1:
                self.copy_next_drive(iteration)
            return
        if checks:
            assert np.array_equal(
                self.solver.y,
                np.array(self.data.state.array[iteration-1, :]),
            ), (
                f'At {iteration=}:\n'
                f'{self.solver.y=}'
                ' != '
                f'{np.array(self.data.state.array[iteration-1, :])=}'
            )
        if iteration % self.modulo:
            self.data.state.array[iteration, :] = (
                self.data.state.array[iteration-1, :]
            )
        else:
            self.solver.set_f_params(self.dstate, iteration, self.data)
            while self.solver.successful() and self.solver.t < time+0.99*timestep:
                self.data.state.array[iteration, :] = (
                    self.solver.integrate(time+timestep)  # , step=True, relax=True
                )
            if not self.solver.successful():
                message = (
                    f'ODE not integrated properly at {iteration=}'
                    f' ({self.solver.t=} < {time+timestep=} [s])'
                    f'\nReturn code: {self.solver.get_return_code()=}'
                    f'\nState:\n{np.array(self.data.state.array[iteration, :])}'
                )
                if strict:
                    raise IntegrationException(message)
                pylog.warning('%s\n\nResetting to previous iteration', message)
                self.solver.set_initial_value(y=self.solver.y, t=time+timestep)
        # Handle drive
        if iteration < self.n_iterations-1:
            self.copy_next_drive(iteration)
        if checks:
            assert self.solver.successful(), (
                f'Solver was not successful at {iteration=}'
            )
            assert abs(time+timestep-self.solver.t) < 1e-6*timestep, (
                'ODE solver time: '
                f'{self.solver.t} [s] != Simulation time: {time+timestep} [s]'
            )
