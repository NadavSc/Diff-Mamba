import sys
import os
project_path = os.path.abspath(os.path.join(__file__, "../../../../"))
babilong_path = os.path.join(project_path, 'retrieval/babilong')
sys.path.insert(0, project_path)
sys.path.insert(1, babilong_path)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM

import datasets
from datasets import Dataset
from tqdm.auto import tqdm
import pandas as pd
import json
from pathlib import Path
import requests

from typing import List
from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input
from modules.modeling_diffmamba2 import DiffMamba2ForCausalLM


def load_mamba2(args, ckpt_path, dtype, quantization_config, diff=False):
    config = AutoConfig.from_pretrained(ckpt_path)
    if diff:
        model = DiffMamba2ForCausalLM(config)
        model = model.from_pretrained(ckpt_path, config, trust_remote_code=True,
                                      device_map='auto', torch_dtype=dtype,
                                      quantization_config=quantization_config)
    else:
        model = Mamba2ForCausalLM(config)
        model = model.from_pretrained(ckpt_path,  trust_remote_code=True,
                                      device_map='auto', torch_dtype=dtype,
                                      quantization_config=quantization_config)
    return model


def main(
    results_folder: str, model_name: str, model_path: str, tokenizer_name: str, tokenizer_path: str,
    tasks: List[str], split_names: List[str], dataset_name: str,
    use_chat_template: bool, api_url: str, use_instruction: bool, use_examples: bool, use_post_prompt: bool,
    system_prompt: str, load_in_8bit: bool, load_in_4bit: bool, mamba: bool, local: bool, diff: bool, args
) -> None:
    """
    Main function to get model predictions on babilong and save them.
    This function supports both local models using the transformers library and remote (local) models accessed via an
    API URL, running with llamacpp or vllm.

    Args:
        results_folder (str): Folder to store results.
        model_name (str): Name of the model to use (will be used in from_pretrained and to save model results)
        model_path (str): path to local model weights
        tokenizer_name (str): tokenizer to use in .from_pretrained, model_name is used if not specified
        tokenizer_path (str): path to tokenizer to use in .from_pretrained
        tasks (List[str]): List of tasks to evaluate.
        split_names (List[str]): List of lengths to evaluate.
        dataset_name (str): Dataset name from Hugging Face.
        use_chat_template (bool): Flag to use the tokenizer chat template.
        api_url (str): API endpoint for llama.cpp.
        use_instruction (bool): Flag to use instruction in prompt.
        use_examples (bool): Flag to use examples in prompt.
        use_post_prompt (bool): Flag to use post_prompt text in prompt.
        system_prompt (str): system prompt to use with chat template (e.g., "You are a helpful AI assistant.")
        load_in_8bit (bool): load in 8 bit with bitsandbytes
        load_in_4bit (bool): load in 4 bit with bitsandbytes
    """
    if model_path is None:
        # use model from transformerss
        model_path = model_name

    if tokenizer_path is None:
        tokenizer_path = tokenizer_name
        if tokenizer_path is None:
            tokenizer_path = model_path

    quantization_config = None
    if load_in_8bit or load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if not api_url:
        # load the model locally if llamacpp / vllm APIs are not used
        try:
            if mamba:
                if local:
                    print('trying to load local mamba')
                    model = load_mamba2(args, model_path, dtype, quantization_config, diff).to('cuda')
                else:
                    print('trying to load mamba')
                    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                                 device_map='auto', torch_dtype=dtype,
                                                                 quantization_config=quantization_config)
            else:
                print('trying to load model with flash attention 2...')
                model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                             device_map='auto', torch_dtype=dtype,
                                                             attn_implementation='flash_attention_2',
                                                             quantization_config=quantization_config)
        except ValueError as e:
            print(e)
            print('trying to load model without flash attention 2...')
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                         device_map='auto', torch_dtype=dtype,
                                                         quantization_config=quantization_config)

        model = model.eval()

    # define generation parameters
    generate_kwargs = {
        'max_new_tokens': 20,
        'max_length': None,
        'num_beams': 1,
        'do_sample': False,
        'temperature': None,
        'top_p': None,
        'top_k': None,
        'pad_token_id': tokenizer.pad_token_id
    }

    if mamba and local and diff:
        generate_kwargs['use_cache'] = False

    if generate_kwargs['pad_token_id'] is None:
        generate_kwargs['pad_token_id'] = tokenizer.eos_token_id

    print(f'prompt template:\n{DEFAULT_TEMPLATE}')
    print(generate_kwargs)

    def preprocess_babilong(dataset, train=False):
        data_dict = {}
        for i in range(1, 21):
            key = f'qa{i}'
            data_dict[key] = {'question': dataset[key]['question'][:-100] if train else dataset[key]['question'][-100:], 
                              'target': dataset[key]['target'][:-100] if train else dataset[key]['target'][-100:], 
                              'input': dataset[key]['input'][:-100] if train else dataset[key]['input'][-100:]}
        return data_dict
        #return Dataset.from_dict(data_dict)

    for task in tqdm(tasks, desc='tasks'):
        # configure the prompt
        prompt_cfg = {
            'instruction': DEFAULT_PROMPTS[task]['instruction'] if use_instruction else '',
            'examples': DEFAULT_PROMPTS[task]['examples'] if use_examples else '',
            'post_prompt': DEFAULT_PROMPTS[task]['post_prompt'] if use_post_prompt else '',
            'template': DEFAULT_TEMPLATE,
            'chat_template': use_chat_template,
            'system_prompt': system_prompt
        }
        prompt_name = [f'{k}_yes' if prompt_cfg[k] else f'{k}_no' for k in prompt_cfg if k != 'template']
        prompt_name = '_'.join(prompt_name)

        for split_name in tqdm(split_names, desc='lengths'):
            if args.eval0k:
                data = datasets.load_dataset(dataset_name, split_name)
                data = preprocess_babilong(data, train=False)
                task_data = data[task]
                task_data = Dataset.from_dict(task_data)
            else:
                data = datasets.load_dataset(dataset_name, split_name)
                task_data = data[task]
            
            # Prepare files with predictions, prompt, and generation configurations
            outfile = Path(f'{results_folder}/{model_name}/{task}_{split_name}_{prompt_name}.csv')
            outfile.parent.mkdir(parents=True, exist_ok=True)
            cfg_file = f'./{results_folder}/{model_name}/{task}_{split_name}_{prompt_name}.json'
            json.dump({'prompt': prompt_cfg, 'generate_kwargs': generate_kwargs}, open(cfg_file, 'w'), indent=4)

            df = pd.DataFrame({'target': [], 'output': [], 'question': []})
            for sample in tqdm(task_data, desc=f'task: {task} length: {split_name}'):
                target = sample['target']
                context = sample['input']
                question = sample['question']

                # format input text
                input_text = get_formatted_input(context, question, prompt_cfg['examples'],
                                                 prompt_cfg['instruction'], prompt_cfg['post_prompt'],
                                                 template=prompt_cfg['template'])

                if api_url:
                    # model is running via llamacpp's serve command
                    headers = {'Content-Type': 'application/json'}
                    if generate_kwargs['temperature'] is None:
                        generate_kwargs['temperature'] = 0.0

                    if use_chat_template:
                        input_text = [{'role': 'user', 'content': input_text}]
                        if system_prompt:
                            input_text = [{"role": "system", "content": system_prompt}] + input_text
                        model_inputs = tokenizer.apply_chat_template(input_text, tokenize=True,
                                                                     add_generation_prompt=True)
                    else:
                        model_inputs = tokenizer.encode(input_text, add_special_tokens=True)

                    request_data = {'prompt': model_inputs, 'temperature': generate_kwargs['temperature'],
                                    'model': model_name}
                    response = requests.post(api_url, headers=headers, json=request_data).json()

                    if 'content' in response:
                        # llamacpp
                        output = response['content'].strip()
                    elif 'choices' in response:
                        # openai compatible api (vllm)
                        output = response['choices'][0]['text'].strip()
                else:
                    # generate output using local model
                    if model.name_or_path in ['THUDM/chatglm3-6b-128k']:
                        # have to add special code to run chatglm as tokenizer.chat_template tokenization is not
                        # the same as in model.chat (recommended in https://huggingface.co/THUDM/chatglm3-6b-128k)
                        with torch.no_grad():
                            output, _ = model.chat(tokenizer, input_text, history=[], **generate_kwargs)
                    else:
                        if use_chat_template:
                            input_text = [{'role': 'user', 'content': input_text}]
                            model_inputs = tokenizer.apply_chat_template(input_text, add_generation_prompt=True,
                                                                         return_tensors='pt').to(model.device)
                            model_inputs = {'input_ids': model_inputs}
                        else:
                            model_inputs = tokenizer(input_text, return_tensors='pt',
                                                     add_special_tokens=True).to(model.device)

                        sample_length = model_inputs['input_ids'].shape[1]
                        with torch.no_grad():
                            output = model.generate(**model_inputs, **generate_kwargs)
                            # we need to reset memory states between samples for activation-beacon models
                            if 'activation-beacon' in model.name_or_path and hasattr(model, 'memory'):
                                model.memory.reset()

                        output = output[0][sample_length:]
                        output = tokenizer.decode(output, skip_special_tokens=True).strip()

                df.loc[len(df)] = [target, output, question]
                # write results to csv file
                df.to_csv(outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model evaluation')
    parser.add_argument('--results_folder', type=str, required=True, default='./babilong_evals',
                        help='Folder to store results')
    parser.add_argument('--dataset_name', type=str, required=True, default='RMT-team/babilong-1k-samples',
                        help='dataset name from huggingface')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--model_path', type=str, required=False, help='path to model, optional')
    parser.add_argument('--tokenizer_name', type=str, required=False, help='tokenizer to use in .from_pretrained')
    parser.add_argument('--tokenizer_path', type=str, required=False, help='path to tokenizer to use in .from_pretrained')
    parser.add_argument('--tasks', type=str, nargs='+', required=True, help='List of tasks to evaluate: qa1 qa2 ...')
    parser.add_argument('--lengths', type=str, nargs='+', required=True, help='List of lengths to evaluate: 0k 1k ...')
    parser.add_argument('--mamba', action='store_true', help='Use mamba')
    parser.add_argument('--local', action='store_true', help='Local model')
    parser.add_argument('--diff', action='store_true', help='Use diffmamba')
    parser.add_argument('--eval0k', action='store_true', default=False, help='Eval on the test 0k examples only')
    parser.add_argument('--use_chat_template', action='store_true', help='Use tokenizer chat template')
    parser.add_argument('--use_instruction', action='store_true', help='Use instruction in prompt')
    parser.add_argument('--use_examples', action='store_true', help='Use examples in prompt')
    parser.add_argument('--use_post_prompt', action='store_true', help='Use post prompt text in prompt')
    parser.add_argument('--system_prompt', type=str, required=False, default='',
                        help='system prompt to use in chat template')
    parser.add_argument('--api_url', type=str, required=True, default='', help='llamacpp/vllm api endpoint')
    parser.add_argument('--load_in_8bit', action='store_true', help='load in 8 bit with bitsandbytes')
    parser.add_argument('--load_in_4bit', action='store_true', help='load in 4 bit with bitsandbytes')
    parser.add_argument('--verbose', type=int, default=1, help='load in 4 bit with bitsandbytes')

    args = parser.parse_args()

    print(args)

    main(args.results_folder, args.model_name, args.model_path,  args.tokenizer_name, args.tokenizer_path,
         args.tasks, args.lengths, args.dataset_name,
         args.use_chat_template, args.api_url, args.use_instruction, args.use_examples, args.use_post_prompt,
         args.system_prompt, args.load_in_8bit, args.load_in_4bit, args.mamba, args.local, args.diff, args)
