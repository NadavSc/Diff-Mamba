import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from typing import Callable, Dict, Union, Optional, Tuple, NamedTuple, Any, List
import logging
from pathlib import Path
import rich
import rich.syntax

import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import os.path as osp
from torch import nn
from datetime import datetime
import lightning as L

import json
import pprint
import argparse
from long_context.forgetting_transformer.src.forgetting_transformer.datamodule.longcrawl64 import LongCrawl64DataModule
from transformers import AutoConfig
from transformers.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM
from modules.modeling_diffmamba2 import DiffMamba2ForCausalLM
import pickle
import numpy as np
import tqdm
import re


def get_step_from_filename(filename: str):
    # filename: step-000000000016384.<suffix>
    # Check format
    assert re.match(r"^step-(\d+)\..+", filename), (
        "Filename should be of format `step-<step>.<suffix> where <step> should be at"
        f" least 1 digit, but got {filename}."
    )
    stem, _, _ = filename.partition('.')
    _, _, digits = stem.partition('-')
    return int(digits)

def get_checkpoint_filename(step, min_num_digits=15):
    num_digits = max(min_num_digits, len(str(step)))
    return f"step-{step:0{num_digits}d}.pt"

def load_checkpoint(checkpoint_dir, device):
    steps = []
    for sentinel_path in checkpoint_dir.glob('*.pt.done'):
        steps.append(get_step_from_filename(sentinel_path.name))
    if len(steps) > 0:
        resume_step = max(steps)
    else:
        resume_step = None
    if resume_step is not None:
        path = checkpoint_dir / get_checkpoint_filename(resume_step)
        print(f"Loading checkpoint from {path}")
        with path.open("rb") as f:
            state = torch.load(f, map_location=device)
        return resume_step, state
    else:
        return None, None

def delete_checkpoints(checkpoint_dir):
    print("Not resuming. Deleting existing checkpoints...")
    # Delete sentinels first
    for path in checkpoint_dir.glob('*.pt.done'):
        path.unlink()
    # Actual checkpoint
    for path in checkpoint_dir.glob('*.pt'):
        path.unlink()

def save_checkpoint(
    checkpoint_dir: Path,
    state: Dict,
    step: int,
):
    """Save checkpoint for a step.

    The save checkpoint is like `{checkpoint_dir}/step-000000000016384.pt`

    Args:
        - keep: normally we only keep the latest checkpoint. If keep is True
          then this checkpoint is permanent
    """
    # file name is typically like "step-000000000016384.pt"
    path = checkpoint_dir / get_checkpoint_filename(step)
    print(f"Saving checkpoint to {path}...")
    with path.open("wb") as f:
        torch.save(
            state,
            f
        )
    sentinel_path = Path(f"{path}.done")
    with sentinel_path.open("w"):
        pass

    print(f"Checkpoint saved to {path}.")

    # Delete previous checkpoints without keep flag
    for delete_path in checkpoint_dir.glob('*.pt'):
        delete_step = get_step_from_filename(delete_path.name)
        delete = False
        if not Path(f"{delete_path}.done").is_file() or delete_step < step:
            delete = True
        elif delete_step > step :
            raise ValueError(f"Currently saving checkpoint at step {step} but found checkpoint at step {step} at path {delete_path}. Aborting.")
        else:
            assert delete_step == step, "What??"

        if delete:
            print(f"Deleting outdated or invalid checkpoint {delete_path}")
            sentinel_path = Path(f"{delete_path}.done")
            sentinel_path.unlink()
            delete_path.unlink()

def apply_diffmamba(args, model):
    nlayers = len(model.backbone.layers)
    n_difflayers = nlayers
    print(f'Number of layers: {nlayers}, converting last {n_difflayers} to DiffMamba2Blocks')
    for layer_idx, pretrained_block in enumerate(model.backbone.layers):
        if layer_idx >= nlayers - n_difflayers and layer_idx % 2 == 0:
            diffmamba_block = DiffMamba2Block.from_pretrained_block(pretrained_block, args)
            model.backbone.layers[layer_idx] = diffmamba_block

    print('Applied DiffMamba successfully')
    return model


def model_load(args):
    config = AutoConfig.from_pretrained(args.model_path)
    model = DiffMamba2ForCausalLM(config) if args.diffmamba else Mamba2ForCausalLM(config)
    if args.diffmamba:
        model = DiffMamba2ForCausalLM.from_pretrained(args.ckpt_path, config)
    else:
        model = Mamba2ForCausalLM.from_pretrained(args.ckpt_path)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='diffmamba2-370M')
    parser.add_argument('--diffmamba', action='store_true', default=True)
    parser.add_argument('--model_path', type=str, default='outputs/diffmamba2-370M')
    parser.add_argument('--data_path', type=str, default='data/longcrawl64')
    parser.add_argument('--save_dir', type=str, default='forgetting-transformer/eval/per_token_loss/results')
    parser.add_argument('--resume', action="store_true", default=True)
    parser.add_argument('--save_interval', type=int, default=128)
    parser.add_argument('--eval_len', type=int, default=65536)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--eval_tokens', type=int, default=1 * 2 ** 30)
    parser.add_argument('--local_batch_size', type=int, default=1)
    args = parser.parse_args()

    model_name = args.model
    save_dir = Path(args.save_dir) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    resume = args.resume
    local_batch_size = args.local_batch_size
    eval_tokens = args.eval_tokens
    assert resume
    data_path = Path(args.data_path)
    # device_id = Path(args.device_id)
    save_interval = args.save_interval
    print(f"Evaluating {model_name}")

    assert data_path.is_dir()

    fabric = L.Fabric(
        # strategy=L.fabric.strategies.FSDPStrategy(cpu_offload=True),
        strategy="ddp",
        devices="auto", precision="32-true")
    # device = torch.device(f"cuda:{device_id}")



    with torch.cuda.device(fabric.device):
        model = model_load(args).to(fabric.device)

    EVAL_LEN = args.eval_len
    datamodule = LongCrawl64DataModule(
        data_dir=data_path,
        world_size=fabric.world_size,
        rank=fabric.global_rank,
        train_seq_len=None,
        train_batch_len=16384,
        train_batch_size=16384,
        train_num_workers=1,

        eval_tokens=eval_tokens,
        eval_seq_len=None,
        eval_batch_len=EVAL_LEN,
        eval_local_batch_size=local_batch_size,
        eval_num_workers=2,

        # CRUCIAL
        eval_stateful=True
    )
    assert EVAL_LEN == 65536

    val_dataloader, val_data_info = datamodule.val_dataloader()
    model = fabric.setup_module(model)
    val_dataloader = fabric.setup_dataloaders(val_dataloader, use_distributed_sampler=False)

    if resume:
        resume_step, checkpoint = load_checkpoint(save_dir, device=fabric.device)
    else:
        if fabric.is_global_zero:
            delete_checkpoints(save_dir)
        resume_step, checkpoint = None, None
    fabric.barrier()

    if checkpoint is None:
        print(f"Starting from scratch")
        total_loss = torch.zeros(
            size=(EVAL_LEN-1,), dtype=torch.float64, device=fabric.device
        )
        seq_count = 0
        batch_count = 0
    else:
        print(f"Resuming from step {resume_step}")
        total_loss = checkpoint["total_loss"]
        seq_count = checkpoint["seq_count"]
        batch_count = checkpoint["batch_count"]
        assert resume_step <= len(val_dataloader), resume_step
    val_dataloader.sampler.load_state_dict({"batch_id": batch_count})


    # (T)
    if fabric.is_global_zero:
        pbar = tqdm.tqdm(total=len(val_dataloader))
        pbar.update(batch_count)
    for batch in val_dataloader:
        # assert batch_count == seq_count, (batch_count, seq_count)
        input_ids = batch.input_ids.to(fabric.device)
        labels = batch.labels.to(fabric.device)

        with torch.cuda.device(fabric.device):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    output = model(
                        input_ids=input_ids,
                        labels=labels,
                    )
                    local_loss = output.loss.double().unsqueeze(0).sum(axis=0)
                    total_loss += fabric.all_reduce(local_loss, reduce_op="sum")
                    assert val_data_info.global_batch_size == fabric.world_size
        seq_count += val_data_info.global_batch_size
        batch_count += 1

        if (batch_count % save_interval == 0 or batch_count == len(val_dataloader)):
            fabric.barrier()
            print(f'Batch {batch_count}: {total_loss}')
            if fabric.is_global_zero:
                save_checkpoint(
                    checkpoint_dir=save_dir,
                    state={
                        "total_loss": total_loss,
                        "seq_count": seq_count,
                        "batch_count": batch_count,
                    },
                    step=batch_count
                )
            fabric.barrier()
        if fabric.is_global_zero:
            pbar.update()

    fabric.barrier()
    if fabric.is_global_zero:
        loss_per_token = (total_loss / seq_count).cpu().numpy()
        MIN_VAL_LENGTH = 512
        length = val_data_info.seq_len
        metrics = {"eval_tokens": eval_tokens, "eval_len": EVAL_LEN}
        while length >= MIN_VAL_LENGTH:
            loss_avg = loss_per_token[:length].mean(axis=0)
            perplexity = np.exp(loss_avg)
            metrics[f"val/loss_avg_len_{length}"] = loss_avg
            metrics[f"val/perplexity_len_{length}"] = perplexity
            length = length // 2

        pprint.pprint(metrics)
        save_path = save_dir / "results.json"
        with save_path.open("w") as f:
            json.dump(metrics, f, indent=4)


        npz_save_path = save_dir / "results.npz"
        np.savez(
            npz_save_path,
            **{**metrics, "val/loss_per_token": loss_per_token},
        )
    # import ipdb; ipdb.set_trace()
if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
