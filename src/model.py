"""PGND implementation site, following the repository-root main.tex.

The manuscript defines z=(q,v), dq/dt=v and
dv/dt=-(L_D L_D^T)v-grad_q V(q)+B h(t)+r(q,v,h,t), with h(t) an
interpolated observation encoding and force decoded from z(t). Training uses
force MSE plus residual-magnitude regularization.

Implementation is deferred until latent dimensions, network forms, physical
time coordinates, solver, causal observation access, initialization and the
experimental split are resolved. See legacy/README.md. Baseline recurrent
architectures are separate in baselines.py.
"""

import torch


class PGNDModel(torch.nn.Module):
    """Placeholder for the PGND model; the architecture is not yet defined."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("The PGND architecture has not yet been implemented.")
