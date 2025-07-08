import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
except ImportError:
    RMSNormGated, LayerNorm = None, None
from src.models.functional.rms_norm import RMSNorm

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined


class DiffS6Mamba2(nn.Module):
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
        d_state=64,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=108,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=False,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        self.lambda_init = 0.8 - 0.6 * torch.exp(torch.tensor(-0.3 * (i_layer - 1)))
        # self.lambda_q1 = nn.Parameter(torch.randn(d_model))
        self.lambda_q1 = nn.Parameter(torch.randn(1))

        self.subln = RMSNorm(self.d_inner, eps=1e-5, elementwise_affine=True)

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj1 = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        self.in_proj2 = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        def dt_init():
            # Initialize log dt bias
            dt = torch.exp(
                torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            dt = torch.clamp(dt, min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            dt_bias = nn.Parameter(inv_dt)
            # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
            # name.endswith("bias") in param_grouping.py
            dt_bias._no_weight_decay = True
            return dt, dt_bias

        def A_init():
            A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
            A_log = torch.log(A).to(dtype=dtype)
            A_log = nn.Parameter(A_log)
            # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
            A_log._no_weight_decay = True
            return A_log

        def D_init():
            # D "skip" parameter
            D = nn.Parameter(torch.ones(self.nheads, device=device))
            D._no_weight_decay = True
            return D

        self.dt1, self.dt1_bias = dt_init()
        self.dt2, self.dt2_bias = dt_init()

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        self.A1_log = A_init()
        self.A2_log = A_init()

        self.D1 = D_init()
        self.D2 = D_init()

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, u, *args, state=None, inference_params=None, seq_idx=None, **kwargs):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt1 = self.in_proj1(u)  # (B, L, d_in_proj)
        zxbcdt2 = self.in_proj2(u)  # (B, L, d_in_proj)
        A1 = -torch.exp(self.A1_log)  # (nheads) or (d_inner, d_state)
        A2 = -torch.exp(self.A2_log)
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.use_mem_eff_path:
            # Fully fused path
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight,
                rmsnorm_eps=self.norm.eps,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=False,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
        else:
            z1, xBC1, dt1 = torch.split(
                zxbcdt1, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
            )
            z2, xBC2, dt2 = torch.split(
                zxbcdt2, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
            )
            dt1 = F.softplus(dt1 + self.dt1_bias)  # (B, L, nheads)
            dt2 = F.softplus(dt2 + self.dt2_bias)  # (B, L, nheads)

            assert self.activation in ["silu", "swish"]

            # 1D Convolution
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                xBC1 = self.act(
                    self.conv1d(xBC1.transpose(1, 2)).transpose(1, 2)
                )  # (B, L, self.d_inner + 2 * ngroups * d_state)
                xBC1 = xBC1[:, :seqlen, :]
                xBC2 = self.act(
                    self.conv1d(xBC2.transpose(1, 2)).transpose(1, 2)
                )  # (B, L, self.d_inner + 2 * ngroups * d_state)
                xBC2 = xBC2[:, :seqlen, :]
            else:
                xBC1 = causal_conv1d_fn(
                    x=xBC1.transpose(1, 2),
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)
                xBC2 = causal_conv1d_fn(
                    x=xBC2.transpose(1, 2),
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)

            # Split into 3 main branches: X, B, C
            # These correspond to V, K, Q respectively in the SSM/attention duality
            x1, B1, C1 = torch.split(xBC1, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            x2, B2, C2 = torch.split(xBC2, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

            y1 = mamba_chunk_scan_combined(
                rearrange(x1, "b l (h p) -> b l h p", p=self.headdim),
                dt1,
                A1,
                rearrange(B1, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C1, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=self.D1,
                z=None,
                seq_idx=seq_idx,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
            y1 = rearrange(y1, "b l h p -> b l (h p)")
            y2 = mamba_chunk_scan_combined(
                rearrange(x2, "b l (h p) -> b l h p", p=self.headdim),
                dt2,
                A2,
                rearrange(B2, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C2, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=self.D2,
                z=None,
                seq_idx=seq_idx,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
            y2 = rearrange(y2, "b l h p -> b l (h p)")
            # Multiply "gate" branch and apply extra normalization layer
            y1 = self.norm(y1, z1)
            y2 = self.norm(y2, z2)

            # lambda_q1 = torch.sum(self.lambda_q1, dim=-1).float()
            lambda_q1 = self.lambda_q1.float()
            lambda_full = torch.sigmoid(lambda_q1) + self.lambda_init
            attn = y1 - lambda_full * y2
            attn = self.subln(attn)
            attn = attn * (1 - self.lambda_init)
            out = self.out_proj(attn)
        return out, None

    @property
    def d_output(self):
        return self.d_model