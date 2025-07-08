import seaborn as sns
import matplotlib
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import pandas as pd
import numpy as np
import argparse

from babilong.metrics import compare_answers, TASK_LABELS

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Generate performance heatmap for model evaluations.")
    parser.add_argument('--results_folder', type=str, default='scripts/babilong_evals/diffmamba2-370m-needle-finetune-seed0',
                        help="Folder where the evaluation results are stored.")
    parser.add_argument('--model_name', type=str, default='diffmamba2-370m-needle-finetune-seed0', help="Name of the model being evaluated.")
    parser.add_argument('--prompt_name', type=str, default='instruction_no_examples_no_post_prompt_no_chat_template_no_system_prompt_no',
                        help="Prompt name used for the evaluations.")
    parser.add_argument('--tasks', type=str, nargs='+', default=['qa1', 'qa2', 'qa3', 'qa4', 'qa5'],
                        help="List of tasks to evaluate.")
    parser.add_argument('--lengths', type=str, nargs='+', default=['0k', '1k', '2k', '4k', '8k', '16k', '32k', '64k'],
                        help="List of context lengths to evaluate.")
    parser.add_argument('--output_file', type=str, default=None, help="Path to save the output PDF file (optional).")
    
    return parser.parse_args()

# Main execution
def main():
    args = parse_args()
    
    accuracy = np.zeros((len(args.tasks), len(args.lengths)))
    
    for j, task in enumerate(args.tasks):
        for i, ctx_length in enumerate(args.lengths):
            fname = f'{args.results_folder}/{args.model_name}/{task}_{ctx_length}_{args.prompt_name}.csv'
            if not os.path.isfile(fname):
                print(f'No such file: {fname}')
                continue

            print(fname)
            df = pd.read_csv(fname)

            if df['output'].dtype != object:
                df['output'] = df['output'].astype(str)
            df['output'] = df['output'].fillna('')

            df['correct'] = df.apply(lambda row: compare_answers(row['target'], row['output'],
                                                                 row['question'], TASK_LABELS[task]
                                                                 ), axis=1)
            score = df['correct'].sum()
            accuracy[j, i] = 100 * score / len(df) if len(df) > 0 else 0

    np.save(f'{args.results_folder}/{args.model_name}/acc_{args.model_name}.npy', accuracy)

    # Set large font sizes for better visibility in the PDF
    matplotlib.rc('font', size=14)

    # Create a colormap for the heatmap
    cmap = LinearSegmentedColormap.from_list('ryg', ["red", "yellow", "green"], N=256)

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(8, 5))  # Adjust the size as necessary
    sns.heatmap(accuracy, cmap=cmap, vmin=0, vmax=100, annot=True, fmt=".2f",
            linewidths=.5, xticklabels=args.lengths, yticklabels=args.tasks, ax=ax, annot_kws={"fontsize": 18})
    ax.set_xlabel('Context Length', fontsize=20)
    ax.set_ylabel('Tasks', fontsize=20)
    plt.xticks(rotation=0, fontsize=18)
    plt.yticks(rotation=0, fontsize=18)

    # Save the figure to a PDF
    output_file = args.output_file if args.output_file else f'{args.results_folder}/{args.model_name}/{args.model_name}.pdf'
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
