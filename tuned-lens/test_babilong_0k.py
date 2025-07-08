import os
import sys
sys.path.append('/home/nadavsch/projects/Diff-Mamba-test')
import argparse
import torch
import numpy as np
from tuned_lens.nn.lenses import TunedLens
from transformers import AutoConfig, AutoTokenizer
from transformers.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM
from modules.modeling_diffmamba2 import DiffMamba2ForCausalLM

from tuned_lens.plotting import PredictionTrajectory
from datasets import Dataset, load_dataset

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate tuned lens model performance")
    parser.add_argument('--diffmamba', action='store_true', default=False, help='Use diff model variant')
    parser.add_argument('--ckpt_path', type=str, help='model checkpoint path')
    parser.add_argument('--num_examples', type=int, default=None, help='Number of examples to evaluate')
    return parser.parse_args()


def model_load(ckpt_path, diffmamba):
    config = AutoConfig.from_pretrained(ckpt_path)
    model = DiffMamba2ForCausalLM(config) if diffmamba else Mamba2ForCausalLM(config)
    if diffmamba:
        model = model.from_pretrained(ckpt_path, config)
    else:
        model = model.from_pretrained(ckpt_path)
    return model.to('cuda')


def load_model_and_lens(ckpt_path, diffmamba):
    model = model_load(ckpt_path, diffmamba)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    lens_resource_id = f'my_lenses/{os.path.basename(ckpt_path)}'
    print(lens_resource_id)

    tuned_lens = TunedLens.from_model_and_pretrained(model, lens_resource_id=lens_resource_id, map_location='cuda')
    tuned_lens = tuned_lens.to('cuda')
    return model, tokenizer, tuned_lens


def preprocess_babilong(dataset, train=False):
    data_dict = {'question': [], 'target': [], 'input': []}
    for i in range(1, 21):
        data_dict['question'] += dataset[f'qa{i}']['question'][:-100] if train else dataset[f'qa{i}']['question'][-100:]
        data_dict['target'] += dataset[f'qa{i}']['target'][:-100] if train else dataset[f'qa{i}']['target'][-100:]
        data_dict['input'] += dataset[f'qa{i}']['input'][:-100] if train else dataset[f'qa{i}']['input'][-100:]
    return Dataset.from_dict(data_dict)


def preprocess_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        examples['text'] = []
        for question, target, input in zip(examples['question'], examples['target'], examples['input']):
            examples['text'].append(f'<context> \n{input}\n</context>\n\nQuestion: {question} Answer:')
        return tokenizer(examples["text"])
    dataset = dataset.map(tokenize_function, batched=True)
    return dataset


def needle_prob(max_probs, target_id):
    for layer_idx in range(1, num_layers + 1):
        max_probs[layer_idx]['total'] += 1
        max_probs[layer_idx]['sum'] += predictition_traj_ring.probs[layer_idx][-1][target_id['input_ids'][0]]
        if np.isnan(max_probs[layer_idx]['sum']):
            import pdb; pdb.set_trace()
        max_probs[layer_idx]['avg'] = max_probs[layer_idx]['sum'] / max_probs[layer_idx]['total']


if __name__ == "__main__":
    args = parse_args()
    model, tokenizer, tuned_lens = load_model_and_lens(args.ckpt_path, args.diffmamba)

    num_layers = len(model.backbone.layers)
    dataset = load_dataset("RMT-team/babilong-1k-samples", '0k')

    eval_dataset = preprocess_babilong(dataset, train=False)
    shuffle_eval_dataset = eval_dataset.shuffle(seed=42)
    eval_dataset = preprocess_dataset(shuffle_eval_dataset, tokenizer)
    print(f'eval preprocessed')

    if args.num_examples is None:
        args.num_examples = len(eval_dataset)
        
    max_probs = {layer_idx: {'sum': 0, 'total': 0} for layer_idx in range(1, num_layers + 1)}
    for idx, example in enumerate(eval_dataset):
        if idx == args.num_examples:
            break
        input_ids = example['input_ids']
        target_id = tokenizer(' ' + example['target'])
        predictition_traj_ring = PredictionTrajectory.from_lens_and_model(
            tuned_lens,
            model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            targets=input_ids[1:] + [tokenizer.eos_token_id]
        )
        needle_prob(max_probs, target_id)
        if idx % 20 == 0:
            print(max_probs)

    print(max_probs)
    with open(f'{args.ckpt_path}-lens_eval.txt', 'w') as f:
        f.write(str(max_probs))
