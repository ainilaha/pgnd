"""PGND from manuscript/main.tex, using its quadratic-potential special case.

The first comparison resets every model on the same observation window and
predicts the next force. Linear interpolation uses observed samples only;
the final, unobserved prediction interval holds the last encoding constant.
Time is expressed in units of time_unit seconds (default: milliseconds).
Historical initialization/residual/ODE controls live in legacy/pgnd_experiments.py.
The default PGND checkpoint parameterization remains unchanged.
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
    decoded trajectory from the initialization instant and the residual penalty.
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
        self.initialization, self.initialization_index = "first", 0
        self.dynamics_kind, self.use_residual = "structured", True
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
        # eq:residual_loss: mean over times/samples of the SUM of squared components.
        residual = self.dynamics.residual_force(
            grid[None, 1:, None], states[:, 1:], h_at_states[:, 1:])
        residual_loss = residual.square().sum(dim=-1).mean()
        if not return_details:
            return prediction, residual_loss
        return prediction, {"residual_loss": residual_loss,
                            "states": states, "force_trajectory": forces,
                            "times": times, "solver_times": grid}

    def forward_events(self, values, times, lengths, target_time, return_residual=False):
        """Frozen-window irregular pilot: retained sensors at their actual times.

        times: (B, padded_events), seconds from the ORIGINAL window origin;
        target_time: scalar seconds in that same clock. Padding is ignored.
        Initialize at each first retained event, interpolate only its retained
        encodings, then hold the last encoding until the force target. No
        missing sensor values are reconstructed. This remains a window model,
        not a persistent recording-level observer. The original forward path
        and all checkpoint tensors are unchanged.
        Optional training penalty averages squared residuals on the original
        1-ms solver-output grid AFTER initialization, equally per window.
        """
        if self.initialization != "first" or self.dynamics_kind != "structured":
            raise ValueError("Event pilot supports the original first-observation PGND only.")
        times = torch.as_tensor(times, dtype=values.dtype, device=values.device)
        lengths = torch.as_tensor(lengths, dtype=torch.long, device=values.device)
        target = torch.as_tensor(target_time, dtype=values.dtype, device=values.device)
        if (values.ndim != 3 or values.shape[-1] != 2 or times.shape != values.shape[:2]
                or lengths.shape != (len(values),) or target.ndim != 0
                or (lengths < 2).any() or (lengths > values.shape[1]).any()):
            raise ValueError("Invalid packed event shapes/counts or target time.")
        valid = torch.arange(values.shape[1], device=values.device)[None, :] < lengths[:, None]
        if (not torch.isfinite(times[valid]).all() or not torch.isfinite(values[valid]).all()
                or (times[valid] < 0).any() or (times[valid] >= target).any()
                or not (times.diff(dim=1)[valid[:, 1:]] > 0).all()):
            raise ValueError("Retained times must increase strictly before the target; inputs must be finite.")
        # Encode real events only. Even arbitrary/NaN padding cannot influence h.
        encoded = values.new_zeros(*values.shape[:2], self.dynamics.drive.in_features)
        encoded[valid] = self.encoder(values[valid])
        origins = times[:, :1] / self.time_unit
        event_times = torch.where(valid, times / self.time_unit - origins, torch.inf).contiguous()
        durations = target / self.time_unit - origins[:, 0]
        # Different first observations imply different integration durations.
        # Unused post-target states in shorter batch members never enter output.
        grid, inverse = torch.unique(torch.cat((durations.new_zeros(1), durations)),
                                     sorted=True, return_inverse=True)
        if return_residual:
            steps = round(float(target_time) / self.sample_interval)
            penalty_grid = torch.arange(steps + 1, device=values.device, dtype=values.dtype)
            penalty_grid = penalty_grid * self.sample_interval / self.time_unit
            grid = torch.unique(torch.cat((grid, penalty_grid)), sorted=True)
            final_indices = torch.searchsorted(grid, durations)
        else:
            final_indices = inverse[1:]
        damping, stiffness = self.dynamics.damping(), self.dynamics.stiffness()

        def conditioning(query):
            query = query.contiguous()
            left = (torch.searchsorted(event_times, query) - 1).clamp(min=0)
            left = torch.minimum(left, (lengths - 2)[:, None])
            lo, hi = (event_times.gather(1, i) for i in (left, left + 1))
            weight = ((query - lo) / (hi - lo)).clamp(0, 1)[..., None]
            lo_h, hi_h = (encoded.gather(1, i[..., None].expand(-1, -1, encoded.shape[-1]))
                          for i in (left, left + 1))
            return lo_h + weight * (hi_h - lo_h)

        def field(t, state):
            h = conditioning(t.expand(len(values), 1)).squeeze(1)
            return self.dynamics(t + origins, state, h, damping, stiffness)

        options = {"step_size": self.step_size} if self.method == "rk4" else None
        states = odeint(field, self.initial_state(values[:, 0]), grid,
                        method=self.method, rtol=self.rtol, atol=self.atol, options=options)
        rows = torch.arange(len(values), device=values.device)
        prediction = self.decoder(states[final_indices, rows])
        if self.direct_readout is not None:
            prediction = prediction + self.direct_readout(encoded[rows, lengths - 1])
        if return_residual:
            penalty_times = penalty_grid[1:][None, :].expand(len(values), -1)
            z = states[torch.searchsorted(grid, penalty_grid[1:])].transpose(0, 1)
            h = conditioning(penalty_times)
            residual = self.dynamics.residual_force((penalty_times + origins)[..., None], z, h)
            # Ignore integration past each member's target. Tolerance only
            # admits rounding at a nominal-grid endpoint, not another step.
            selected = penalty_times <= durations[:, None] + 1e-5
            penalty = ((residual.square().sum(-1) * selected).sum(1) / selected.sum(1)).mean()
            return prediction, penalty
        return prediction

    def propagate(self, x, state=None, start_indices=0):
        """Assimilate available observations, preserving elapsed recording time.

        x includes both endpoints of each observed interval. Reusing the final
        state with the last observation as the next chunk's first observation
        continues the same trajectory; only state=None calls the initializer.
        This path is separate from forward(), whose historical behavior stays
        unchanged. Forecasts must branch BEFORE assimilating their target input.
        """
        if x.ndim != 3 or x.shape[1] < 2 or x.shape[2] != 2:
            raise ValueError("Propagation needs (batch, at least 2 observations, 2).")
        if state is None:
            state = self.initial_state(x[:, 0])
        grid = torch.arange(x.shape[1], dtype=x.dtype, device=x.device) * self.sample_interval / self.time_unit
        offsets = torch.as_tensor(start_indices, dtype=x.dtype, device=x.device).reshape(-1, 1)
        offsets = offsets * (self.sample_interval / self.time_unit)
        encoded = self.encoder(x)
        damping, stiffness = self.dynamics.damping(), self.dynamics.stiffness()

        def field(t, z):
            return self.dynamics(t + offsets, z, self.interpolate(t, grid, encoded), damping, stiffness)

        options = {"step_size": self.step_size} if self.method == "rk4" else None
        return odeint(field, state, grid, method=self.method, rtol=self.rtol,
                      atol=self.atol, options=options).transpose(0, 1)

    def recording_stream(self, recordings, batches, context_length, return_residual=False):
        """Yield indexed causal forecasts while carrying one state per recording.

        batches comes from data.recording_batches; no file I/O or force input.
        The first context_length observations warm the original initializer.
        Cache numerical states, detached at each microchunk for truncated BPTT.
        Within a chunk, forecasts use z_(k-1), h_(k-1) only; assimilated z_k may
        use x_k AFTER that forecast and is carried to subsequent predictions.
        Keep the original last-L-grid-point residual penalty per target.
        """
        if context_length < 2:
            raise ValueError("Recording context must contain at least two observations.")
        parameter = next(self.parameters())
        history = {}
        for indices, ids, starts, ends, pairs in batches:
            width = max(end - start + 1 for start, end in zip(starts, ends))
            inputs, previous = [], []
            for i, start, end in zip(ids, starts, ends):
                source = torch.as_tensor(recordings[i], dtype=parameter.dtype)
                positions = torch.arange(start - context_length + 1, start + width, device=source.device)
                # Left padding is never scored; right padding is never carried.
                inputs.append(source[positions.clamp(0, end)].to(parameter.device))
                if i not in history:
                    if start != 0:
                        raise ValueError("A new recording must begin at observation zero.")
                    initial = self.initial_state(inputs[-1][context_length - 1])
                    history[i] = initial.expand(context_length, -1)
                previous.append(history[i])
            inputs, previous = torch.stack(inputs), torch.stack(previous)
            states = self.propagate(inputs[:, context_length - 1:], previous[:, -1], starts)
            states = torch.cat((previous[:, :-1], states), dim=1)
            pair = torch.as_tensor(pairs, dtype=torch.long, device=parameter.device)
            rows, targets = pair[:, 0], pair[:, 1] + context_length - 1
            z = states[rows, targets - 1]
            h = self.encoder(inputs[rows, targets - 1])
            origins = torch.as_tensor(starts, dtype=parameter.dtype, device=parameter.device) - context_length + 1
            times = (origins[rows] + targets - 1)[:, None] * (self.sample_interval / self.time_unit)
            interval = parameter.new_tensor([0., self.sample_interval / self.time_unit])
            damping, stiffness = self.dynamics.damping(), self.dynamics.stiffness()

            def field(t, state):
                return self.dynamics(t + times, state, h, damping, stiffness)

            options = {"step_size": self.step_size} if self.method == "rk4" else None
            forecast = odeint(field, z, interval, method=self.method, rtol=self.rtol,
                              atol=self.atol, options=options)[-1]
            prediction = self.decoder(forecast).reshape(-1)
            if self.direct_readout is not None:
                prediction = prediction + self.direct_readout(h).reshape(-1)
            penalty = prediction.new_zeros(len(prediction))
            if return_residual:
                past = targets[:, None] - torch.arange(context_length - 1, 0, -1, device=parameter.device)
                past_states = states[rows[:, None], past]
                past_h = self.encoder(inputs[rows[:, None], past])
                past_times = (origins[rows, None] + past)[:, :, None] * (self.sample_interval / self.time_unit)
                past_r = self.dynamics.residual_force(past_times, past_states, past_h)
                final_r = self.dynamics.residual_force(times + interval[-1], forecast, h)
                penalty = (past_r.square().sum((1, 2)) + final_r.square().sum(1)) / context_length
            for row, (i, start, end) in enumerate(zip(ids, starts, ends)):
                last = end - start + context_length - 1
                history[i] = states[row, last - context_length + 1:last + 1].detach()
            if not (torch.isfinite(prediction).all() and torch.isfinite(penalty).all()
                    and all(torch.isfinite(history[i]).all() for i in ids)):
                raise ValueError("Persistent-state propagation became non-finite; no automatic reset.")
            yield indices, prediction, penalty
