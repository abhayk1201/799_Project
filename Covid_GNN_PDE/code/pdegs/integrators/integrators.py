import torch
from torchdiffeq import odeint_adjoint


class Integrator():
    def __init__(self):
        raise NotImplementedError()

    def integrate(self, *args, **kwargs):
        raise NotImplementedError()


class ODEAdjointIntegrator(Integrator):
    def __init__(self):
        pass

    def integrate(self, f, y0, t, **kwargs):
        return odeint_adjoint(f, y0, t, **kwargs)  # (t, |y|)


class RecurrentIntegrator(Integrator):
    def __init__(self):
        pass

    def integrate(self, f, y0, nsteps):
        nt = nsteps + 1
        states = torch.zeros(nt, *y0.shape)

        y = y0

        for i in range(0, nt):
            states[i] = y
            y = f(y)
    
        return states
