import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "10000"
import argparse
import torch
import random
import numpy as np

from datasets import Dataset, load_dataset
from transformers import AutoConfig, AutoTokenizer, TrainingArguments, Trainer, set_seed, default_data_collator
from transformers.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM
from modules.modeling_diffmamba2 import DiffMamba2ForCausalLM
from logger import info


try:
    local_rank = int(os.environ["LOCAL_RANK"])  # GPU number for this process (0 or 1)
    info(f'LOCAL RANK: {local_rank}')
except:
    info('1 GPU is utilized')
set_seed(0)


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune Mamba')
    parser.add_argument('--model_id', type=str, default='AntonV/mamba2-370m-hf',
                        help='Model identifier to load from HuggingFace')
    parser.add_argument('--ckpt_path', type=str, default='./outputs/mamba2-370m-pg19-finetune',
                        help='ckpt path to load a specific model')
    parser.add_argument('--diffmamba', action='store_true', default=False,
                        help='Apply diffmamba to the pretrained model')
    parser.add_argument('--output_dir', type=str, default='./outputs/mamba2-370m-needle-finetune',
                        help='Directory to save model outputs')
    parser.add_argument('--batch_size', type=int, default=6,
                        help='Training batch size per device')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate for training')
    parser.add_argument('--logging_steps', type=int, default=20,
                        help='Number of steps between logging events')
    parser.add_argument('--save_steps', type=int, default=100,
                        help='Number of steps between model saves'),
    parser.add_argument('--eval_steps', type=int, default=100,
                        help='Number of steps between PPL evaluations')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Number of training steps'),
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps'),
    parser.add_argument('--warmup_steps', type=int, default=50,
                        help='Number of warmup steps in the scheduler'),
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay in AdamW'),
    parser.add_argument('--max_length', type=int, default=2048,
                        help='Max length in tokenizer'),
    parser.add_argument('--wandb', type=str, default='wandb',
                        help='report to wandb or "none" to disable')
    parser.add_argument('--seed', type=int, default=0,
                        help='torch seed through the training')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Model identifier to load from HuggingFace')
    return parser.parse_args()


def preprocess_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        for question, target, input in zip(examples['question'], examples['target'], examples['input']):
            prompt = f'<context> \n{input}\n</context>\n\nQuestion: {question} Answer: '
            answer = target
            full_text = prompt + answer
            # example: <context>\nJohn travelled to the hallway. Mary journeyed to the bathroom.\n</context>\n\nQuestion: Where is John?
            tokenized = tokenizer(full_text, truncation=True, padding="max_length", max_length=2048)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            # for 'left' padding tokenizer
            prompt_ids = []
            answer_ids = []
            answer = True
            for token in reversed(input_ids):
                if token == 27:
                    answer = False
                if token == 0:
                    break
                if answer:
                    answer_ids.append(token)
                else:
                    prompt_ids.append(token)

            answer_ids.reverse()
            prompt_ids.reverse()

            # Compute the number of padding tokens on the left
            padding_len = input_ids.count(0)

            # Shift label start position by the amount of left-padding
            start = padding_len + len(prompt_ids)
            labels = [-100] * 2048
            for i, token in enumerate(answer_ids):
                if start + i < len(labels):
                    labels[start + i] = token

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list
        }

    dataset = dataset.map(tokenize_function, batched=True)
    return dataset


def preprocess_babilong(dataset, train=False):
    data_dict = {'question': [], 'target': [], 'input': []}
    for i in range(1, 21):
        data_dict['question'] += dataset[f'qa{i}']['question'][:-100] if train else dataset[f'qa{i}']['question'][-100:]
        data_dict['target'] += dataset[f'qa{i}']['target'][:-100] if train else dataset[f'qa{i}']['target'][-100:]
        data_dict['input'] += dataset[f'qa{i}']['input'][:-100] if train else dataset[f'qa{i}']['input'][-100:]
    return Dataset.from_dict(data_dict)


def main():
    args = parse_args()
    args.output_dir = args.output_dir + f'-seed{args.seed}'

    # Log the parsed arguments
    info(f"Output Directory: {args.output_dir}")
    info(f"Checkpoint Path: {args.ckpt_path}")
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
    torch.cuda.manual_seed_all(args.seed)  # Ensures consistency on multi-GPU setups
    info(f"Random Seed: {args.seed}")

    def model_load():
        config = AutoConfig.from_pretrained(args.ckpt_path)
        model = DiffMamba2ForCausalLM(config) if args.diffmamba else Mamba2ForCausalLM(config)
        if args.diffmamba:
            model = model.from_pretrained(args.ckpt_path, config)
        else:
            model = model.from_pretrained(args.ckpt_path)
        return model

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    data_collator = default_data_collator

    info(f'Loading RMT-team/babilong-1k-samples dataset')
    dataset = load_dataset("RMT-team/babilong-1k-samples", '0k')

    train_dataset = preprocess_babilong(dataset, train=True)
    shuffle_train_dataset = train_dataset.shuffle(seed=42)
    train_dataset = preprocess_dataset(shuffle_train_dataset, tokenizer)
    info(f'train preprocessed')

    eval_dataset = preprocess_babilong(dataset, train=False)
    shuffle_eval_dataset = eval_dataset.shuffle(seed=42)
    eval_dataset = preprocess_dataset(shuffle_eval_dataset, tokenizer)
    info(f'eval preprocessed')

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
        seed=args.seed
    )

    trainer = Trainer(
        model_init=model_load,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()