# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class DiffS6(nn.Module):
    def __init__(
        self,
        d_model,
        i_layer,
        l_max,
        final_act,
        dropout,
        lr,
        n_ssm,
        transposed,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x1_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt1_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.x2_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt2_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        def dt_proj_init(dt_proj):
            # Initialize special dt projection to preserve variance at initialization
            dt_init_std = self.dt_rank**-0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))

            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True
            return dt_proj

        def A_init():
            # S4D real initialization
            A = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_log = torch.log(A)  # Keep A_log in fp32
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
            return A_log

        def D_init():
            # D "skip" parameter
            D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            D._no_weight_decay = True
            return D

        self.dt1_proj = dt_proj_init(self.dt1_proj)
        self.dt2_proj = dt_proj_init(self.dt2_proj)

        self.A1_log = A_init()
        self.A2_log = A_init()

        self.D1 = D_init()
        self.D2 = D_init()

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, *args, state=None, inference_params=None, **kwargs):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A1 = -torch.exp(self.A1_log.float())  # (d_inner, d_state)
        A2 = -torch.exp(self.A2_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        x, z = xz.chunk(2, dim=1)
        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x1_dbl = self.x1_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        x2_dbl = self.x2_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)

        dt1, B1, C1 = torch.split(x1_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt2, B2, C2 = torch.split(x2_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt1 = self.dt1_proj.weight @ dt1.t()
        dt2 = self.dt2_proj.weight @ dt2.t()

        dt1 = rearrange(dt1, "d (b l) -> b d l", l=seqlen)
        dt2 = rearrange(dt2, "d (b l) -> b d l", l=seqlen)

        B1 = rearrange(B1, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        B2 = rearrange(B2, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        C1 = rearrange(C1, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C2 = rearrange(C2, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        assert self.activation in ["silu", "swish"]
        y1 = selective_scan_fn(
            x,
            dt1,
            A1,
            B1,
            C1,
            self.D1.float(),
            z=z,
            delta_bias=self.dt1_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        y2 = selective_scan_fn(
            x,
            dt2,
            A2,
            B2,
            C2,
            self.D2.float(),
            z=z,
            delta_bias=self.dt2_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        y = y1-y2
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out, ssm_state

    def allocate_inference_cache(self, batch_size, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def default_state(self, *batch_shape, device=None):
        # kernel is not a SequenceModule so it doesn't need to adhere to same interface
        # the kernel will know the device of its own parameters
        return self.allocate_inference_cache(*batch_shape)

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

    @property
    def d_output(self):
        return self.d_model