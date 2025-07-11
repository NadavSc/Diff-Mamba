"""Implements a full residual block around a black box layer.

Configurable options include:
normalization position: prenorm or postnorm
normalization type: batchnorm, layernorm etc.
subsampling/pooling
residual options: feedforward, residual, affine scalars, depth-dependent scaling, etc.
"""

from functools import partial
import torch
from torch import nn

from src.models.nn import Normalization, StochasticDepth, DropoutNd
from src.models.sequence import SequenceModule
from src.models.sequence.modules.pool import registry as pool_registry
from src.models.nn.residual import registry as residual_registry
import src.utils as utils
import src.utils.registry as registry

class SequenceResidualBlock(SequenceModule):
    """Flexible residual block design. See model.py for meaning of options."""

    def __init__(
            self,
            d_input,
            i_layer=None, # Only needs to be passed into certain residuals like Decay
            prenorm=True,
            bidirectional=False,
            dropout=0.0,
            tie_dropout=False,
            transposed=False,
            layer=None, # Config for black box module
            residual=None, # Config for residual function
            norm=None, # Config for normalization layer
            pool=None,
            drop_path=0.,
        ):
        super().__init__()

        self.zero_diff=False
        self.i_layer = i_layer
        self.d_input = d_input
        self.prenorm = prenorm
        self.bidirectional = bidirectional
        self.transposed = transposed

        if self.zero_diff:
            if layer['_name_'] == 'diffmamba2' and i_layer <= 8:
                layer = layer.copy()
                layer['_name_'] = 'mamba2'
        self.layer = utils.instantiate(registry.layer, layer, d_input, i_layer)
        if self.bidirectional:
            self.reverse_layer = utils.instantiate(registry.layer, layer, d_input)
            self.bidirectional_linear = nn.Linear(2*self.layer.d_output, self.layer.d_output)

        # Residual
        # d_residual is the output dimension after residual
        if residual is None:
            self.residual = None
            self.d_residual = self.layer.d_output
        else:
            self.residual = utils.instantiate(residual_registry, residual, i_layer, d_input, self.layer.d_output)
            self.d_residual = self.residual.d_output

        # Normalization
        d_norm = d_input if self.prenorm else self.d_residual
        # We don't use config to directly instantiate since Normalization has some special cases
        if norm is None:
            self.norm = None
        elif isinstance(norm, str):
            self.norm = Normalization(d_norm, transposed=self.transposed, _name_=norm)
        else:
            self.norm = Normalization(d_norm, transposed=self.transposed, **norm)

        # Pool
        self.pool = utils.instantiate(pool_registry, pool, self.d_residual, transposed=self.transposed)

        # Dropout
        dropout_cls = partial(DropoutNd, transposed=self.transposed) if tie_dropout else nn.Dropout
        self.drop = dropout_cls(dropout) if dropout > 0.0 else nn.Identity()

        # Stochastic depth
        self.drop_path = StochasticDepth(drop_path, mode='row') if drop_path > 0.0 else nn.Identity()


    @property
    def d_output(self):
        return self.pool.d_output if self.pool is not None else self.d_residual

    @property
    def d_state(self):
        return self.layer.d_state

    @property
    def state_to_tensor(self):
        return self.layer.state_to_tensor

    def default_state(self, *args, **kwargs):
        return self.layer.default_state(*args, **kwargs)

    def forward(self, x, state=None, **kwargs):
        y = x

        # Pre-norm
        if self.norm is not None and self.prenorm: y = self.norm(y)

        # Black box layer
        y_for, new_state = self.layer(y, state=state, **kwargs)
        if self.bidirectional:
            assert state is None
            y_rev, _ = self.reverse_layer(y, state=state, **kwargs)
            if self.transposed: y = torch.cat([y_for, y_rev], dim=1)
            else: y = torch.cat([y_for, y_rev], dim=-1)
            y = self.bidirectional_linear(y)
        else:
            y = y_for

        # Residual
        if self.residual is not None: y = self.residual(x, self.drop_path(self.drop(y)), self.transposed)

        # Post-norm
        if self.norm is not None and not self.prenorm: y = self.norm(y)

        # Pool
        if self.pool is not None: y, _ = self.pool(y)

        return y, state

    def step(self, x, state, **kwargs):
        assert not self.bidirectional
        y = x

        # Pre-norm
        if self.norm is not None and self.prenorm:
            y = self.norm.step(y)

        # Black box layer
        y, state = self.layer.step(y, state, **kwargs)

        # Residual
        if self.residual is not None: y = self.residual(x, y, transposed=False) # NOTE this would not work with concat residual function (catformer)

        # Post-norm
        if self.norm is not None and not self.prenorm:
            y = self.norm.step(y)

        # Pool
        if self.pool is not None: y, _ = self.pool(y)

        return y, state
