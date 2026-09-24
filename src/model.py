"""PGND from main.tex, using its quadratic-potential special case.

The first comparison resets every model on the same observation window and
predicts the next force. Linear interpolation uses observed samples only;
the final, unobserved prediction interval holds the last encoding constant.
Time is expressed in units of time_unit seconds (default: milliseconds).
"""

import torch
from torch import nn
from torchdiffeq import odeint


class PGNDDynamics(nn.Module):
    """dq/dt=v; dv/dt=-Dv-grad V(q)+Bh+r(q,v,h,t)."""

    def __init__(self, latent_dim=16, encoding_dim=16, hidden_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.L_D = nn.Parameter(0.1 * torch.eye(latent_dim))
        self.L_K = nn.Parameter(0.1 * torch.eye(latent_dim))
        self.drive = nn.Linear(encoding_dim, latent_dim, bias=False)
        self.residual = nn.Sequential(
            nn.Linear(2 * latent_dim + encoding_dim + 1, hidden_dim),
            nn.Tanh(), nn.Linear(hidden_dim, latent_dim),
        )
        # Start from the structured dynamics; learn the residual correction.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def damping(self):
        # main.tex: eq:damping_constraint.
        return self.L_D @ self.L_D.T

    def stiffness(self):
        # A positive-semidefinite quadratic potential: eq:quadratic_potential.
        return self.L_K @ self.L_K.T

    def potential(self, q):
        return 0.5 * (q * (q @ self.stiffness())).sum(dim=-1)

    def residual_force(self, t, state, encoded_observation):
        # main.tex: eq:neural_residual; t broadcasts over batch/trajectory axes.
        time = torch.ones_like(state[..., :1]) * t
        return self.residual(torch.cat((state, encoded_observation, time), dim=-1))

    def forward(self, t, state, encoded_observation):
        q, v = state.split(self.latent_dim, dim=-1)
        # grad_q (q^T K q / 2) = Kq; analytic gradient retains autograd to K,q.
        acceleration = (-v @ self.damping() - q @ self.stiffness()
                        + self.drive(encoded_observation)
                        + self.residual_force(t, state, encoded_observation))
        return torch.cat((v, acceleration), dim=-1)


class PGNDModel(nn.Module):
    """Observation encoding, initialization, ODE evolution, and force decoding.

    x: (batch, observed_steps, 2), with no observation at the target instant.
    times: optional increasing physical seconds, length observed_steps + 1.
    Output: next-sample force (batch, 1). return_details also exposes the
    complete decoded trajectory and the manuscript's residual penalty.
    """

    def __init__(self, latent_dim=16, encoding_dim=16, hidden_dim=32,
                 sample_interval=0.001, time_unit=0.001, method="rk4",
                 step_size=1.0, rtol=1e-5, atol=1e-7):
        super().__init__()
        if min(sample_interval, time_unit, step_size, rtol, atol) <= 0:
            raise ValueError("Time intervals, solver step and tolerances must be positive.")
        self.sample_interval = sample_interval
        self.time_unit = time_unit
        self.method, self.step_size = method, step_size
        self.rtol, self.atol = rtol, atol
        self.encoder = nn.Sequential(nn.Linear(2, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, encoding_dim))
        # main.tex: eq:initial_state, single initial observation (K0=0).
        self.initial_state = nn.Sequential(nn.Linear(2, hidden_dim), nn.Tanh(),
                                          nn.Linear(hidden_dim, 2 * latent_dim))
        self.dynamics = PGNDDynamics(latent_dim, encoding_dim, hidden_dim)
        self.decoder = nn.Sequential(nn.Linear(2 * latent_dim, hidden_dim),
                                     nn.Tanh(), nn.Linear(hidden_dim, 1))

    @staticmethod
    def interpolate(t, observation_times, encoded):
        """eq:linear_interpolation, with a constant extension after the last input."""
        left = (torch.searchsorted(observation_times, t) - 1).clamp(
            0, len(observation_times) - 2)
        fraction = ((t - observation_times[left])
                    / (observation_times[left + 1] - observation_times[left])).clamp(0, 1)
        return encoded[:, left] + fraction * (encoded[:, left + 1] - encoded[:, left])

    def forward(self, x, times=None, return_details=False):
        if x.ndim != 3 or x.shape[-1] != 2 or x.shape[1] < 2:
            raise ValueError("PGND expects (batch, at least 2 observed steps, 2).")
        if times is None:
            times = torch.arange(x.shape[1] + 1, device=x.device,
                                 dtype=x.dtype) * self.sample_interval
        else:
            times = torch.as_tensor(times, device=x.device, dtype=x.dtype)
        if times.ndim != 1 or len(times) != x.shape[1] + 1 or not (times.diff() > 0).all():
            raise ValueError("times must increase and include one future target instant.")
        grid = (times - times[0]) / self.time_unit
        encoded = self.encoder(x)  # eq:observation_encoder
        initial = self.initial_state(x[:, 0])

        def vector_field(t, state):
            h = self.interpolate(t, grid[:-1], encoded)
            return self.dynamics(t, state, h)

        options = {"step_size": self.step_size} if self.method == "rk4" else None
        # Direct differentiation through a standard solver; no custom integrator.
        states = odeint(vector_field, initial, grid, method=self.method,
                        rtol=self.rtol, atol=self.atol, options=options).transpose(0, 1)
        forces = self.decoder(states).squeeze(-1)  # eq:force_decoder
        prediction = forces[:, -1:]
        if not return_details:
            return prediction
        h_at_states = torch.cat((encoded, encoded[:, -1:]), dim=1)
        residual = self.dynamics.residual_force(
            grid[None, 1:, None], states[:, 1:], h_at_states[:, 1:])
        # eq:residual_loss: mean over times/samples of the SUM of squared components.
        return prediction, {"residual_loss": residual.square().sum(dim=-1).mean(),
                            "states": states, "force_trajectory": forces, "times": times}
