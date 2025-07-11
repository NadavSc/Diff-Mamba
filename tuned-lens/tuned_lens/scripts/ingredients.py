"""Shared configuration for the scripts."""
import enum
import logging
import os
import torch
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from typing import Optional, Union
from omegaconf import OmegaConf

import torch as th
import pytorch_lightning as pl
import torch.distributed as dist
from datasets import Dataset, DatasetDict, load_dataset
from simple_parsing import field
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torchdata import dataloader2, datapipes

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)
from typing_extensions import Literal

from tuned_lens.data import (
    chunk_and_tokenize
)
from tuned_lens.model_surgery import get_transformer_layers
from tuned_lens.nn.lenses import Lens
from tuned_lens.utils import (
    TreeType,
    handle_name_conflicts,
    send_to_device,
)

import sys
project_path = os.path.abspath(os.path.join(__file__, "../../../"))
projects_path = os.path.abspath(os.path.join(__file__, "../../../../"))
sys.path.insert(0, project_path)
sys.path.insert(1, projects_path)
from transformers.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM
from modules.modeling_diffmamba2 import DiffMamba2ForCausalLM
logger = logging.getLogger(__name__)


@dataclass
class Data:
    """Configuration for the dataset."""

    name: list[str] = field(default_factory=lambda: ["the_pile", "all"], nargs="*")
    """Name of dataset to use. Can either be a local .jsonl file or a name
    suitable to be passed to the HuggingFace load_dataset function."""

    split: str = "validation"
    """Split of the dataset to use."""

    text_column: str = "text"
    """Column of the dataset containing text to run the model on."""

    revision: Optional[str] = None
    """The revision of the dataset to use"""

    max_seq_len: int = 2048
    """The maximum length of the input sequences."""

    dataset_shuffle: bool = False
    """Whether to shuffle the dataset prior to tokenization."""

    dataset_shuffle_seed: int = 42
    """Seed to use for shuffling the dataset"""

    def convert_txt_to_jsonl(self, name_json):
        import json
        with open(self.name[0], "r") as f:
            lines = f.readlines()
        data = []
        for i, line in enumerate(lines):
            text = line.strip()
            data.append({"input_ids": i, "text": text})
        with open(name_json, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    def load(self, model, tokenizer: PreTrainedTokenizerBase, ssm: bool = False) -> tuple[Dataset, float]:
        """Load the dataset, tokenize it and compute nats_to_bpb."""
        logger.info(f"Loading dataset '{' '.join(self.name)}'")
        logger.debug(f"Using split '{self.split}', revision '{self.revision}'")
        if self.name[0].endswith(".txt"):
            self.name[0] = os.path.join(project_path, self.name[0])
            name_json = self.name[0].replace('.txt', '.jsonl')
            if not os.path.exists(name_json):
               self.convert_txt_to_jsonl(name_json)
            self.name[0] = name_json
            dataset = Dataset.from_json(self.name[0])
            # dataset = model.val_dataloader()[0]
        elif len(self.name) == 1 and self.name[0].endswith(".jsonl"):
            self.name[0] = os.path.join(project_path, self.name[0])
            dataset = Dataset.from_json(self.name[0])
            assert isinstance(dataset, Dataset)
        else:
            dataset = load_dataset(*self.name, '1k')
            # dataset = load_dataset(*self.name, split=self.split, revision=self.revision)
            if not isinstance(dataset, (Dataset, DatasetDict)):
                raise ValueError(
                    "Only Dataset and DatasetDict instances are supported."
                )

        logger.debug(f"Dataset has {len(dataset)} samples.")
        if not ssm:
            logger.debug(f"Dataset columns: {dataset.column_names}")

        if self.dataset_shuffle:
            logger.debug(f"Shuffling dataset with seed: {self.dataset_shuffle_seed}")
            dataset = dataset.shuffle(self.dataset_shuffle_seed)

        if ssm:
            print('Not Implemented')
        else:
            logger.debug("Beginning tokenization...")
            processed, nats_to_bpb = chunk_and_tokenize(
                dataset,
                tokenizer,
                text_key=self.text_column,
                max_seq_len=self.max_seq_len,
                babilong=False
            )

        logger.info(f"Using nats per token to bits per byte ratio: {nats_to_bpb}")

        if not ssm:
            assert isinstance(processed, Dataset)

        return processed, nats_to_bpb


def apply_diffmamba(args, model):
    nlayers = len(model.backbone.layers)
    n_difflayers = nlayers//4
    print(f'Number of layers: {nlayers}, converting last {n_difflayers} to DiffMamba2Blocks')
    for layer_idx, pretrained_block in enumerate(model.backbone.layers):
        if layer_idx >= nlayers - n_difflayers:
            diffmamba_block = DiffMamba2Block.from_pretrained_block(pretrained_block, args)
            model.backbone.layers[layer_idx] = diffmamba_block

    print('Applied DiffMamba successfully')
    return model


def load_diffmamba2(args, base_model, ckpt_path):
    model = apply_diffmamba(args, base_model)
    model.load_state_dict(torch.load(f"{ckpt_path}/pytorch_model.bin"))
    return model.to('cuda')


@dataclass
class Model:
    """Configuration for the model and tokenizer."""

    name: str
    """Name of model to use in the Huggingface Hub."""

    ssm: bool = False
    """Use the SSM S4 framework."""

    ckpt_path: str = None

    precision: Literal["auto", "bfloat16", "float16", "float32", "int8"] = "auto"
    """Precision in which to load the model weights."""

    revision: str = "main"
    """Git revision to use for pretrained models."""

    slow_tokenizer: bool = field(action="store_true")
    """Use a slow tokenizer."""

    tokenizer: Optional[str] = None
    """Name of pretrained tokenizer to use from the Huggingface Hub. If None, will use
    AutoTokenizer.from_pretrained('<model name>')."""

    tokenizer_type: Optional[str] = None
    """Name of tokenizer class to use. If None, will use AutoTokenizer."""

    def load_tokenizer(self, must_use_cache: bool = False) -> PreTrainedTokenizerBase:
        """Load the tokenizer from huggingface hub."""
        with handle_name_conflicts():
            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer or self.name,
                revision=self.revision,
                use_fast=not self.slow_tokenizer,
                tokenizer_type=self.tokenizer_type,
                local_files_only=must_use_cache,
            )

        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        print(f'tokenizer {self.name} has been loaded')
        return tokenizer

    def load(
        self, device: Optional[th.device], must_use_cache: bool = False
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load the model and tokenizer.

        Args:
            device: The device to load the model on. Implemented with the `device_map`
                argument of `AutoModelForCausalLM.from_pretrained`.
            must_use_cache: If True, will raise an error if the model is not cached.
        """
        logger.info(f"Loading pretrained weights for '{self.name}'...")
        logger.debug(
            "Using revision {revision} dtype {dtype}, and device {device}".format(
                revision=self.revision, dtype=self.precision, device=device
            )
        )

        try:
            dtype = {
                "auto": "auto",
                "bfloat16": th.bfloat16,
                "float16": th.float16,
                "float32": th.float32,
                # `bitsandbytes` requires weights to initially be in fp16
                "int8": th.float16,
            }[self.precision]
        except KeyError as e:
            raise ValueError(f"Unknown precision: {self.precision}") from e

        with handle_name_conflicts():
            if self.ssm:
                diffmamba = 'diff' in self.name
                config = AutoConfig.from_pretrained(self.name)
                model = DiffMamba2ForCausalLM(config) if diffmamba else Mamba2ForCausalLM(config)
                model = model.from_pretrained(self.name, 
                                                device_map={"": device} if device is not None else None,
                                                load_in_8bit=self.precision == "int8",
                                                low_cpu_mem_usage=True,
                                                revision=self.revision,
                                                torch_dtype=dtype,
                                                local_files_only=must_use_cache)
                print(f'model {self.name} has been loaded')
            else:
                model = AutoModelForCausalLM.from_pretrained(  # type: ignore
                    self.name,
                    device_map={"": device} if device is not None else None,
                    load_in_8bit=self.precision == "int8",
                    low_cpu_mem_usage=True,
                    revision=self.revision,
                    torch_dtype=dtype,
                    local_files_only=must_use_cache,
                )

        assert isinstance(model, PreTrainedModel)
        model.eval()
        model.requires_grad_(False)

        return model, self.load_tokenizer(must_use_cache=must_use_cache)


class OptimizerOption(enum.Enum):
    """Options for the optimizer to use when training the model."""

    ADAM = "adam"
    SGD = "sgd"


@dataclass
class Optimizer:
    """Configuration for the optimizer."""

    weight_decay: float = 1e-3
    """Weight decay coefficient."""

    lr_scale: float = 1.0
    """The default LR (1e-3 for Adam, 1.0 for SGD) is scaled by this factor."""

    momentum: float = 0.9
    """Momentum coefficient for SGD, or beta1 for Adam."""

    zero: Optional[bool] = field(action="store_true")
    """Use ZeroRedundancyOptimizer."""

    optimizer: OptimizerOption = OptimizerOption.SGD
    """The type of optimizer to use."""

    warmup_steps: Optional[int] = None
    """Number of warmup steps. Defaults to min(0.2 * num_steps, 1000) for Adam and 0
    for SGD."""

    def create_scheduler(
        self, opt: th.optim.Optimizer, num_steps: int
    ) -> th.optim.lr_scheduler.LambdaLR:
        """Create the LR scheduler."""
        if self.warmup_steps is None:
            # Adam generally performs poorly without an LR warmup
            if self.optimizer == "adam":
                self.warmup_steps = min(1000, num_steps // 5)
                logger.info(f"Using {self.warmup_steps} LR warmup steps for Adam")
            else:
                self.warmup_steps = 0

        scheduler = get_linear_schedule_with_warmup(
            opt, self.warmup_steps, num_steps - self.warmup_steps
        )

        return scheduler

    def create_optim(self, params: list[th.nn.Parameter]) -> th.optim.Optimizer:
        """Create the optimizer."""
        # Don't train things that don't need gradients
        beta = self.momentum
        if self.optimizer == OptimizerOption.SGD:
            config = dict(
                # PyTorch's implementation effectively scales the LR by 1 / (1 - β),
                # so we undo that here. See https://www.youtube.com/watch?v=k8fTYJPd3_I
                # for discussion. Once we do this, the optimal LR seems to be unity.
                lr=self.lr_scale * (1 - beta),
                momentum=beta,
                # Empirically Nesterov momentum seems to improve convergence speed.
                nesterov=True,
                weight_decay=self.weight_decay,
            )
            opt_class = th.optim.SGD
        elif self.optimizer == OptimizerOption.ADAM:
            config = dict(
                # Helps convergence slightly by ensuring that the LR actually decays
                amsgrad=True,
                betas=(beta, 0.999),
                lr=self.lr_scale * 1e-3,
                weight_decay=self.weight_decay,
            )
            opt_class = th.optim.Adam
        else:
            raise ValueError(f"Unknown optimizer '{self.optimizer}'")

        if self.zero:
            opt = ZeroRedundancyOptimizer(params, optimizer_class=opt_class, **config)
        else:
            opt = opt_class(params, **config)  # type: ignore[call-arg]

        return opt

    def per_parameter_optim_state_size(self) -> int:
        """The number of elements in the optimizer state per parameter."""
        return 2 if self.optimizer == OptimizerOption.ADAM else 1


@dataclass
class Distributed:
    """Configuration and utilities for distributing the model."""

    fsdp: bool = field(action="store_true")
    """Run the model with Fully Sharded Data Parallelism."""

    cpu_offload: bool = field(action="store_true")
    """Use CPU offloading. Must be combined with fsdp"""

    nccl_timeout: int = 1200  # 20 minutes
    """Timeout for NCCL operations in seconds."""

    per_gpu_batch_size: int = 1
    """The batch size per GPU."""

    dataloader_shuffle: bool = True
    """Whether to shuffle the batches of tokenized data as they are loaded."""

    @property
    def rank(self) -> int:
        """The rank of this process.

        Note that in general this is not the same as the local rank.
        However, for single-node training, the local rank is the same as the
        global rank.
        """
        return int(os.environ["RANK"]) if dist.is_initialized() else 0

    @property
    def local_rank(self) -> int:
        """The local rank of this process."""
        return int(os.environ["LOCAL_RANK"]) if dist.is_initialized() else 0

    @property
    def world_size(self) -> int:
        """Get the world size from torch.distributed."""
        return int(os.environ["WORLD_SIZE"]) if dist.is_initialized() else 1

    @property
    def primary(self) -> bool:
        """Whether this is the rank 0 process."""
        return self.rank == 0

    @property
    def device(self) -> th.device:
        """The device associated with this process."""
        return (
            th.device("cuda", self.local_rank)
            if th.cuda.is_available()
            else th.device("cpu")
        )

    def shard_model(
        self, model: PreTrainedModel
    ) -> Union[FullyShardedDataParallel, PreTrainedModel]:
        """Shard the model using Fully Sharded Data Parallelism if needed."""
        if self.fsdp:
            _, layers = get_transformer_layers(model)
            layer_cls = type(layers[0])
            logger.info(
                f"Using '{layer_cls.__name__}' for transformer_auto_wrap_policy."
            )
            return FullyShardedDataParallel(
                model,
                auto_wrap_policy=partial(
                    transformer_auto_wrap_policy, transformer_layer_cls={layer_cls}
                ),
                cpu_offload=CPUOffload(offload_params=self.cpu_offload),
                device_id=self.rank,
                # This turns out to be important for training speed
                forward_prefetch=True,
                mixed_precision=MixedPrecision(
                    param_dtype=th.float16,
                    reduce_dtype=th.float16,
                    buffer_dtype=th.float16,
                ),
            )
        elif self.cpu_offload:
            raise ValueError("CPU offload requires FSDP.")
        else:
            return model

    def distribute_lens(self, lens: Lens) -> Union[DDP, Lens]:
        """Distribute the lens using DistributedDataParallel and send lens to device."""
        logger.debug(f"Sending Lens to device {self.device}")
        if self.world_size > 1:
            lens.to(self.device)
            logger.debug("Distributing the lens across the GPUS using DDP ...")
            return DDP(lens, device_ids=[self.local_rank], find_unused_parameters=True)
        else:
            return lens.to(self.device)

    def dataloader(
        self,
        dataset: Dataset,
    ) -> dataloader2.DataLoader2:
        """Shard the dataset based on local rank."""
        dp = datapipes.iter.IterableWrapper(dataset)
        if self.world_size > 1:
            rs = dataloader2.DistributedReadingService()
        else:
            rs = None

        if self.dataloader_shuffle:
            dp = dp.shuffle()

        dp = dp.sharding_filter()
        dp = dp.batch(self.per_gpu_batch_size)
        dp = dp.collate()
        return dataloader2.DataLoader2(dp, reading_service=rs)

    def init(self):
        """Initialize distributed process group if started with elastic launch."""
        # Support both distributed and non-distributed training
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            dist.init_process_group(
                "nccl", timeout=timedelta(seconds=self.nccl_timeout)
            )
            assert (
                th.cuda.is_available()
            ), "CUDA must be available for distributed training"
            th.cuda.set_device(self.local_rank)

    def barrier(self) -> None:
        """Barrier for all processes."""
        if dist.is_initialized():
            dist.barrier()

    def send_to_device(self, pytree: TreeType) -> TreeType:
        """Move pytree to the current device."""
        return send_to_device(pytree, self.device)
