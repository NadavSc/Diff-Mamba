import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
import random
import numpy as np

from datasets import load_dataset

from transformers import AutoConfig, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from transformers.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM
from modules.modeling_diffmamba2 import DiffMamba2ForCausalLM
from logger import info


try:
    local_rank = int(os.environ["LOCAL_RANK"])  # GPU number for this process (0 or 1)
    info(f'LOCAL RANK: {local_rank}')
except:
    info('1 GPU is utilized')


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune Mamba')
    parser.add_argument('--model_id', type=str, default='AntonV/mamba2-370m-hf',
                        help='Model identifier to load from HuggingFace')
    parser.add_argument('--ckpt_path', type=str,
                        default='./outputs/diffmamba2-370m-pg19-finetune',
                        help='ckpt path to load a specific model')
    parser.add_argument('--diffmamba', action='store_true', default=False,
                        help='Apply diffmamba to the pretrained model')
    parser.add_argument('--output_dir', type=str, default='./outputs/mamba2-370m-pg19-finetune',
                        help='Directory to save model outputs')
    parser.add_argument('--batch_size', type=int, default=6,
                        help='Training batch size per device')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate for training')
    parser.add_argument('--logging_steps', type=int, default=20,
                        help='Number of steps between logging events')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Number of steps between model saves'),
    parser.add_argument('--eval_steps', type=int, default=500,
                        help='Number of steps between PPL evaluations')
    parser.add_argument('--max_steps', type=int, default=4000,
                        help='Number of training steps'),
    parser.add_argument('--grad_accum_steps', type=int, default=9,
                        help='Gradient accumulation steps'),
    parser.add_argument('--warmup_steps', type=int, default=400,
                        help='Number of warmup steps in the scheduler'),
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay in AdamW'),
    parser.add_argument('--max_length', type=int, default=2048,
                        help='Max length in tokenizer'),
    parser.add_argument('--finetune', action='store_true', default=True,
                        help='if True load weights from model_id')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='if True eval ckpt on pg19 test')
    parser.add_argument('--wandb', type=str, default='wandb',
                        help='report to wandb or "none" to disable')
    parser.add_argument('--seed', type=int, default=0,
                        help='torch seed through the training')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Model identifier to load from HuggingFace')
    return parser.parse_args()


def tokenize_dataset(dataset, tokenizer, max_length):
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_overflowing_tokens=True,
        )

        sample_map = tokenized.pop("overflow_to_sample_mapping")
        for key, values in examples.items():
            tokenized[key] = [values[i] for i in sample_map]
        return tokenized

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1,
        remove_columns='text'
    )
    return tokenized


def main():
    args = parse_args()

    # Log the parsed arguments
    info(f"Output Directory: {args.output_dir}")
    info(f"Max Training Steps: {args.max_steps}")
    info(f"Batch Size: {args.batch_size}")
    info(f"Gradient Accumulation Steps: {args.grad_accum_steps}")
    info(f"Max Length: {args.max_length}")
    info(f"Logging Steps: {args.logging_steps}")
    info(f"Learning Rate: {args.lr}")
    info(f"Save Steps: {args.save_steps}")
    info(f"Eval Steps: {args.eval_steps}")
    info(f"Weights & Biases Integration: {'Enabled' if args.wandb == 'wandb' else 'Disabled'}")
    info(f"DiffMamba: {'Enabled' if args.diffmamba else 'Disabled'}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    info(f"Random Seed: {args.seed}")

    def model_init():
        config = AutoConfig.from_pretrained(args.model_id)
        model = DiffMamba2ForCausalLM(config) if args.diffmamba else Mamba2ForCausalLM(config)
        if args.finetune:
            model = model.from_pretrained(args.model_id)
            print('Model initialized: Finetune mode...')
        else:
            print('Model initialized: Pretraining mode...')
        if args.diffmamba:
            model.apply_diffmamba()
        return model

    def model_load():
        config = AutoConfig.from_pretrained(args.ckpt_path)
        if args.diffmamba:
            model = DiffMamba2ForCausalLM.from_pretrained(args.ckpt_path, config)
        else:
            model = Mamba2ForCausalLM.from_pretrained(args.ckpt_path)
        return model

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    info(f'Loading dataset PG19')
    train_dataset = load_dataset("emozilla/pg19", split='train', streaming=True)
    train_dataset = tokenize_dataset(train_dataset, tokenizer, args.max_length)

    eval_dataset = load_dataset("emozilla/pg19", split='test')
    eval_dataset = tokenize_dataset(eval_dataset, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=args.logging_steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        save_steps=args.save_steps,
        save_safetensors=False,
        save_total_limit=5,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        max_steps=args.max_steps,
        report_to=args.wandb,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        bf16_full_eval=True
    )

    trainer = Trainer(
        model_init=model_load if args.eval else model_init,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    if args.eval:
        trainer.evaluate()
    else:
        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
