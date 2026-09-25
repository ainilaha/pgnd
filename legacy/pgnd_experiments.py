"""Frozen PGND pilot models, retained for old-checkpoint evaluation only.

These are NOT the previous paper's Keras baselines. Definitions were moved
unchanged out of src/ after the readout/modal pilots did not justify promotion.
See experiment_notes.md for settings, limitations and negative results.
Do not add new experimental variants here or import these into src.train.
"""

import torch
from torch import nn


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

