"""PGND from manuscript/main.tex, using its quadratic-potential special case.

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
        time = torch.as_tensor(t, dtype=state.dtype, device=state.device).expand(*state.shape[:-1], 1)
        return self.residual(torch.cat((state, encoded_observation, time), dim=-1))

    def forward(self, t, state, encoded_observation, damping=None, stiffness=None):
        q, v = state.split(self.latent_dim, dim=-1)
        damping = self.damping() if damping is None else damping
        stiffness = self.stiffness() if stiffness is None else stiffness
        # grad_q (q^T K q / 2) = Kq; analytic gradient retains autograd to K,q.
        acceleration = (-v @ damping - q @ stiffness
                        + self.drive(encoded_observation)
                        + self.residual_force(t, state, encoded_observation))
        return torch.cat((v, acceleration), dim=-1)


class PGNDModel(nn.Module):
    """Observation encoding, initialization, ODE evolution, and force decoding.

    x: (batch, observed_steps, 2), with no observation at the target instant.
    times: optional increasing physical seconds, length observed_steps + 1.
    Output: next-sample force (batch, 1). return_details also exposes the
    complete decoded trajectory and the manuscript's residual penalty.
    return_residual returns only (prediction, penalty) for efficient training.
    observation_readout retains G(h_last) for historical checkpoint evaluation;
    it is not exposed by the normal training CLI.
    Intermediate readouts are diagnostics, not separately scored forecasts.
    """

    def __init__(self, latent_dim=16, encoding_dim=16, hidden_dim=32,
                 sample_interval=0.001, time_unit=0.001, method="rk4",
                 step_size=1.0, rtol=1e-5, atol=1e-7, observation_readout=False):
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
        # Construct after the original modules to preserve their seeded weights.
        # Default False keeps historical PGND state_dicts and behavior unchanged.
        self.direct_readout = None
        if observation_readout:
            self.direct_readout = nn.Sequential(
                nn.Linear(encoding_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
            # The revised model initially predicts exactly the PGND-0 output.
            nn.init.zeros_(self.direct_readout[-1].weight)
            nn.init.zeros_(self.direct_readout[-1].bias)

    @staticmethod
    def interpolate(t, observation_times, encoded):
        """eq:linear_interpolation, with a constant extension after the last input."""
        left = (torch.searchsorted(observation_times, t) - 1).clamp(
            0, len(observation_times) - 2).reshape(1)
        # index_select keeps the index on-device; scalar CUDA indexing can
        # convert the index to a Python integer and synchronize every ODE stage.
        t_left = observation_times.index_select(0, left)
        t_right = observation_times.index_select(0, left + 1)
        fraction = ((t - t_left) / (t_right - t_left)).clamp(0, 1)
        h_left = encoded.index_select(1, left).squeeze(1)
        h_right = encoded.index_select(1, left + 1).squeeze(1)
        return h_left + fraction * (h_right - h_left)

    def forward(self, x, times=None, return_details=False, return_residual=False):
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
        # Invariant within this solve, but rebuilt with autograd every forward.
        # Never detach or cache these matrices across optimizer steps.
        damping = self.dynamics.damping()
        stiffness = self.dynamics.stiffness()

        def vector_field(t, state):
            h = self.interpolate(t, grid[:-1], encoded)
            return self.dynamics(t, state, h, damping, stiffness)

        options = {"step_size": self.step_size} if self.method == "rk4" else None
        # Direct differentiation through a standard solver; no custom integrator.
        states = odeint(vector_field, initial, grid, method=self.method,
                        rtol=self.rtol, atol=self.atol, options=options).transpose(0, 1)
        if return_details:
            forces = self.decoder(states).squeeze(-1)
            if self.direct_readout is not None:
                # Encodings at diagnostic state times; hold the final input.
                h_at_states = torch.cat((encoded, encoded[:, -1:]), dim=1)
                forces = forces + self.direct_readout(h_at_states).squeeze(-1)
            prediction = forces[:, -1:]
        else:
            prediction = self.decoder(states[:, -1])
            if self.direct_readout is not None:
                prediction = prediction + self.direct_readout(encoded[:, -1])
        if not (return_details or return_residual):
            return prediction
        h_at_states = torch.cat((encoded, encoded[:, -1:]), dim=1)
        residual = self.dynamics.residual_force(
            grid[None, 1:, None], states[:, 1:], h_at_states[:, 1:])
        # eq:residual_loss: mean over times/samples of the SUM of squared components.
        residual_loss = residual.square().sum(dim=-1).mean()
        if not return_details:
            return prediction, residual_loss
        return prediction, {"residual_loss": residual_loss,
                            "states": states, "force_trajectory": forces, "times": times}
