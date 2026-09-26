"""Frozen PGND pilot models, retained for old-checkpoint evaluation only.

These are NOT the previous paper's Keras baselines. Definitions were moved
out of src/ after the readout/modal pilots and Priority 1–3 diagnostic screen.
Numerical behavior is retained; no pilot is promoted by this archival move.
See experiment_notes.md for settings, limitations and negative results.
Do not add new experimental variants here or import these into src.train.
"""

import torch
from torch import nn
from torchdiffeq import odeint

from src.model import PGNDModel


# Frozen Priority 1–3 definitions (2026-09-25). Kept together deliberately:
# old checkpoints must not inherit subsequent numerical changes to PGND.
# Subclassing is only for the existing evaluator's PGND diagnostic checks;
# constructor, dynamics, interpolation and forward computation are frozen.
class _ArchivedPGNDDynamics(nn.Module):
    """dq/dt=v; dv/dt=-Dv-grad V(q)+Bh+r(q,v,h,t)."""

    def __init__(self, latent_dim=16, encoding_dim=16, hidden_dim=32, use_residual=True):
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
        # Remove parameters as well as the force. Construct first so the shared
        # modules keep identical seeded initialization in the r=0 ablation.
        if not use_residual:
            self.residual = None

    def damping(self):
        # main.tex: eq:damping_constraint.
        return self.L_D @ self.L_D.T

    def stiffness(self):
        # A positive-semidefinite quadratic potential: eq:quadratic_potential.
        return self.L_K @ self.L_K.T

    def potential(self, q):
        return 0.5 * (q * (q @ self.stiffness())).sum(dim=-1)

    def residual_force(self, t, state, encoded_observation):
        if self.residual is None:
            return torch.zeros_like(state[..., :self.latent_dim])
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


class NeuralDynamics(nn.Module):
    """Diagnostic controls: dq=v, dv=f(z,h,t), or unrestricted dz=f(z,h,t).

    Same 2d state and observation conditioning as PGND, without D, K, B or a
    residual penalty. These are ablations, not calibrated mechanical models.
    """

    def __init__(self, latent_dim, encoding_dim, hidden_dim, second_order):
        super().__init__()
        self.second_order = second_order
        self.field = nn.Sequential(
            nn.Linear(2 * latent_dim + encoding_dim + 1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim if second_order else 2 * latent_dim))

    def forward(self, t, state, encoded_observation, damping=None, stiffness=None):
        time = torch.as_tensor(t, dtype=state.dtype, device=state.device).expand(*state.shape[:-1], 1)
        learned = self.field(torch.cat((state, encoded_observation, time), dim=-1))
        return torch.cat((state.chunk(2, dim=-1)[1], learned), dim=-1) if self.second_order else learned


class AblationPGND(PGNDModel):
    """Observation encoding, initialization, ODE evolution, and force decoding.

    x: (batch, observed_steps, 2), with no observation at the target instant.
    times: optional increasing physical seconds, length observed_steps + 1.
    Output: next-sample force (batch, 1). return_details also exposes the
    decoded trajectory from the initialization instant and the residual penalty.
    return_residual returns only (prediction, penalty) for efficient training.
    observation_readout retains G(h_last) for historical checkpoint evaluation;
    it is not exposed by the normal training CLI.
    Intermediate readouts are diagnostics, not separately scored forecasts.
    Delayed/history initialization integrates from the end of a causal prefix;
    generic dynamics controls have no mechanical residual penalty.
    """

    def __init__(self, latent_dim=16, encoding_dim=16, hidden_dim=32,
                 sample_interval=0.001, time_unit=0.001, method="rk4",
                 step_size=1.0, rtol=1e-5, atol=1e-7, observation_readout=False,
                 initialization="first", warmup_steps=8, dynamics_kind="structured",
                 use_residual=True):
        nn.Module.__init__(self)
        if min(sample_interval, time_unit, step_size, rtol, atol) <= 0:
            raise ValueError("Time intervals, solver step and tolerances must be positive.")
        if initialization not in ("first", "delayed", "history") or warmup_steps < 2:
            raise ValueError("Use first/delayed/history initialization and warmup_steps >= 2.")
        if dynamics_kind not in ("structured", "second_order", "first_order"):
            raise ValueError("Unknown dynamics control.")
        self.initialization, self.warmup_steps = initialization, warmup_steps
        self.initialization_index = 0 if initialization == "first" else warmup_steps - 1
        self.dynamics_kind = dynamics_kind
        self.use_residual = use_residual and dynamics_kind == "structured"
        self.sample_interval = sample_interval
        self.time_unit = time_unit
        self.method, self.step_size = method, step_size
        self.rtol, self.atol = rtol, atol
        self.encoder = nn.Sequential(nn.Linear(2, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, encoding_dim))
        # main.tex: eq:initial_state, single initial observation (K0=0).
        self.initial_state = nn.Sequential(nn.Linear(2, hidden_dim), nn.Tanh(),
                                          nn.Linear(hidden_dim, 2 * latent_dim))
        self.dynamics = _ArchivedPGNDDynamics(latent_dim, encoding_dim, hidden_dim, self.use_residual)
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
        # Experimental replacements follow ALL original modules: encoder,
        # decoder and shared initializer weights stay seed-matched to PGND-0.
        if dynamics_kind != "structured":
            self.dynamics = NeuralDynamics(latent_dim, encoding_dim, hidden_dim,
                                           second_order=dynamics_kind == "second_order")
        if initialization == "history":
            self.initial_state[0] = nn.Linear(2 * warmup_steps, hidden_dim)

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
        start = self.initialization_index
        if start >= x.shape[1]:
            raise ValueError("Warm-up must fit inside the observed window (never include the target).")
        if times is None:
            times = torch.arange(x.shape[1] + 1, device=x.device,
                                 dtype=x.dtype) * self.sample_interval
        else:
            times = torch.as_tensor(times, device=x.device, dtype=x.dtype)
            if times.ndim != 1 or len(times) != x.shape[1] + 1 or not (times.diff() > 0).all():
                raise ValueError("times must increase and include one future target instant.")
        grid = (times - times[0]) / self.time_unit
        encoded = self.encoder(x)  # eq:observation_encoder
        initial_input = (x[:, :self.warmup_steps].flatten(1)
                         if self.initialization == "history" else x[:, start])
        initial = self.initial_state(initial_input)
        # Invariant within this solve, but rebuilt with autograd every forward.
        # Never detach or cache these matrices across optimizer steps.
        damping = self.dynamics.damping() if self.dynamics_kind == "structured" else None
        stiffness = self.dynamics.stiffness() if self.dynamics_kind == "structured" else None

        def vector_field(t, state):
            h = self.interpolate(t, grid[:-1], encoded)
            return self.dynamics(t, state, h, damping, stiffness)

        options = {"step_size": self.step_size} if self.method == "rk4" else None
        # Direct differentiation through a standard solver; no custom integrator.
        # A history initializer is placed at the END of its causal warm-up;
        # it must not initialize at t0 with knowledge of later observations.
        states = odeint(vector_field, initial, grid[start:], method=self.method,
                        rtol=self.rtol, atol=self.atol, options=options).transpose(0, 1)
        if return_details:
            forces = self.decoder(states).squeeze(-1)
            if self.direct_readout is not None:
                # Encodings at diagnostic state times; hold the final input.
                h_at_states = torch.cat((encoded, encoded[:, -1:]), dim=1)
                forces = forces + self.direct_readout(h_at_states[:, start:]).squeeze(-1)
            prediction = forces[:, -1:]
        else:
            prediction = self.decoder(states[:, -1])
            if self.direct_readout is not None:
                prediction = prediction + self.direct_readout(encoded[:, -1])
        if not (return_details or return_residual):
            return prediction
        h_at_states = torch.cat((encoded, encoded[:, -1:]), dim=1)
        # eq:residual_loss: mean over times/samples of the SUM of squared components.
        residual_loss = x.new_zeros(())
        if self.use_residual:
            residual = self.dynamics.residual_force(
                grid[None, start + 1:, None], states[:, 1:], h_at_states[:, start + 1:])
            residual_loss = residual.square().sum(dim=-1).mean()
        if not return_details:
            return prediction, residual_loss
        return prediction, {"residual_loss": residual_loss,
                            "states": states, "force_trajectory": forces,
                            "times": times[start:], "solver_times": grid[start:]}


class DirectReadout(nn.Module):
    """PGND's encoder and direct readout, without an ODE or history processing.

    This new ablation uses only the last available acceleration/uplift pair.
    It is not a baseline recovered from the previous paper. PyTorch default
    initialization matches PGND's MLP family; the final output starts at zero.
    """

    def __init__(self, encoding_dim=16, hidden_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, encoding_dim))
        self.direct_readout = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.direct_readout[-1].weight)
        nn.init.zeros_(self.direct_readout[-1].bias)

    def forward(self, x):
        return self.direct_readout(self.encoder(x[:, -1]))


class HistoryReadout(nn.Module):
    """New full-window linear/MLP control; not a legacy-paper architecture."""

    def __init__(self, sequence_length=64, hidden_dim=32, nonlinear=False):
        super().__init__()
        self.sequence_length = sequence_length
        self.output = (nn.Sequential(nn.Linear(2 * sequence_length, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, 1))
                       if nonlinear else nn.Linear(2 * sequence_length, 1))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.sequence_length, 2):
            raise ValueError("HistoryReadout requires its configured window length and 2 channels.")
        return self.output(x.flatten(1))


class ModalPGND(nn.Module):
    """Experimental damped modal memory, not a replacement for PGNDModel.

    Inspired by LinOSS / D-LinOSS (arxiv:2410.03943, 2505.12171), not their
    full architectures. Each mode obeys q'' + 2*a*q' + (a*a+w*w)*q =
    (a*a+w*w)*u, with positive learned a,w and nonlinear u=encoder(x).
    p=q'/sqrt(a*a+w*w) makes the two state coordinates similarly scaled.
    Exact zero-order-hold evolution replaces the nonlinear RK4 solve.
    A vectorized sum computes only the final state, with no temporal loop.

    The initial state is q=u(first input), p=0 (local equilibrium). Input k
    drives [t_k,t_{k+1}); the last observed input drives the prediction
    interval. No target-time observation or force history is used. These
    are latent oscillators, NOT identified physical pantograph parameters.
    Unlike original PGND, there is no state-dependent residual or penalty.
    """

    def __init__(self, latent_dim=32, hidden_dim=32, sample_interval=0.001):
        super().__init__()
        if min(latent_dim, hidden_dim, sample_interval) <= 0:
            raise ValueError("Dimensions and sample interval must be positive.")
        self.sample_interval = sample_interval
        # Declared initial memory bank: damped frequencies 1--50 Hz and
        # independent decay rates a=0.25*w. Neither is fitted to test data.
        frequency = torch.logspace(0, torch.log10(torch.tensor(50.)).item(), latent_dim)
        self.log_frequency = nn.Parameter(frequency.log())
        self.log_decay = nn.Parameter((0.25 * 2 * torch.pi * frequency).log())
        self.encoder = nn.Sequential(nn.Linear(2, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(2 * latent_dim + 2, hidden_dim),
                                     nn.Tanh(), nn.Linear(hidden_dim, 1))

    def transition(self, times):
        """Exact homogeneous transition exp(A*t), shape (..., modes, 2, 2)."""
        a = self.log_decay.exp()
        w = 2 * torch.pi * self.log_frequency.exp()
        omega = torch.sqrt(a.square() + w.square())
        t = times[..., None]
        decay, cosine = torch.exp(-a * t), torch.cos(w * t)
        # sin(w*t)/w is well behaved even for very small learned frequencies.
        sine_over_w = t * torch.sinc(w * t / torch.pi)
        return decay[..., None, None] * torch.stack((
            torch.stack((cosine + a * sine_over_w, omega * sine_over_w), -1),
            torch.stack((-omega * sine_over_w, cosine - a * sine_over_w), -1),
        ), -2)

    def forward(self, x, return_details=False):
        if x.ndim != 3 or x.shape[-1] != 2 or x.shape[1] < 2:
            raise ValueError("ModalPGND expects (batch, at least 2 steps, 2).")
        u = self.encoder(x)
        steps = x.shape[1]
        times = torch.arange(steps + 1, device=x.device, dtype=x.dtype) * self.sample_interval
        transitions = self.transition(times)
        # For equilibrium e=(1,0), the constant-input contribution over one
        # interval is (I-exp(A*dt))*e*u. At lag j it is (T_j-T_{j+1})*e*u.
        kernels = (transitions[:-1, :, :, 0] - transitions[1:, :, :, 0]).flip(0)
        state = torch.einsum("bln,lnc->bnc", u, kernels)
        state = state + u[:, 0, :, None] * transitions[-1, :, :, 0]
        prediction = self.decoder(torch.cat((state.flatten(1), x[:, -1]), dim=-1))
        if return_details:
            return prediction, {"state": state, "frequency_Hz": self.log_frequency.exp(),
                                "decay_per_second": self.log_decay.exp()}
        return prediction
