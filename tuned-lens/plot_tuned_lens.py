import os
import re
import ast
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'axes.titlesize': 18
})


def extract_seed(filename):
    match = re.search(r'seed(\d+)', filename)
    return int(match.group(1)) if match else None


# Function to load and parse a dictionary from a file
def load_data(filename):
    with open(filename, 'r') as file:
        raw_data = file.read()
    return ast.literal_eval(raw_data)


def main(args):
    db_type = 'needle' if 'needle' in args.diff_results_path else 'pg19'
    seed = extract_seed(args.diff_results_path)
    output = f'outputs/compared-lens-{db_type}-seed{seed}' if seed is not None else f'outputs/compared-lens-{db_type}'
    output += '-log.pdf' if args.log else '.pdf'

    # Load both dictionaries
    data1 = load_data(args.diff_results_path)
    data2 = load_data(args.mamba_results_path)

    # Define the layer range to display
    start_layer = args.start_layer
    end_layer = args.end_layer
    layers = list(range(start_layer, end_layer + 1))

    # Extract average values for the selected layers
    avg1 = [data1[layer]['avg'] for layer in layers]
    avg2 = [data2[layer]['avg'] for layer in layers]

    # Generate x positions
    x = np.arange(len(layers))
    bar_width = 0.4

    # Set Seaborn theme
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 6))

    # Use solid, warm colors for each dataset
    color1 = "#F5B041"
    color2 = "#5DADE2"

    # Plot bars with labels
    label1 = 'Diff-Mamba'
    label2 = 'Mamba'
    bars1 = plt.bar(x - bar_width/2, avg1, width=bar_width, color=color1, label=label1)
    bars2 = plt.bar(x + bar_width/2, avg2, width=bar_width, color=color2, label=label2)

    # Create legend patches manually
    legend_handles = [
        Patch(color=color1, label=label1),
        Patch(color=color2, label=label2)
    ]

    # Aesthetics
    plt.xlabel("Layer", fontsize=16)
    if args.log:
        plt.ylabel("Needle Probability (log scale) [%]", fontsize=16)
        plt.yscale('log')
    else:
        plt.ylabel("Needle Probability [%]", fontsize=16)
    plt.xticks(ticks=x, labels=layers, rotation=90, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(handles=legend_handles, fontsize=16)
    plt.tight_layout()

    # Save the plot
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Diff-Mamba vs Mamba layer-wise results")
    parser.add_argument('--diff_results_path', type=str, help='Path to first model result file (Diff-Mamba)')
    parser.add_argument('--mamba_results_path', type=str, help='Path to second model result file (Mamba)')
    parser.add_argument('--log', action='store_true', default=False, help='Whether to plot the graph with log scale or not')
    parser.add_argument('--start-layer', type=int, default=1, help='Start layer index')
    parser.add_argument('--end-layer', type=int, default=48, help='End layer index')

    args = parser.parse_args()
    main(args)
