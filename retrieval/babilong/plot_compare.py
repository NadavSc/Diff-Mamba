import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
import argparse
import os

# Set a modern style
sns.set_theme(style="whitegrid")
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 150
})

def parse_args():
    parser = argparse.ArgumentParser(description="Generate performance heatmap for model evaluations.")
    parser.add_argument('--results_folder', type=str, default='scripts/babilong_evals')
    parser.add_argument('--substract',  action='store_true', default=False)
    parser.add_argument('--ratio',  action='store_true', default=False)
    parser.add_argument('--model_name', type=str, default='diffmamba2-370m-needle-finetune')
    parser.add_argument('--seeds', type=str, nargs='+', default=['0', '42', '77'])
    parser.add_argument('--tasks', type=str, nargs='+', default=['qa1', 'qa2', 'qa3', 'qa4', 'qa5'])
    parser.add_argument('--lengths', type=str, nargs='+', default=['0k', '1k', '2k', '4k', '8k', '16k', '32k', '64k'])
    parser.add_argument('--output_file', type=str, default=None)
    return parser.parse_args()

def load_accuracy_matrix(results_folder, model_name):
    path = os.path.join(results_folder, model_name, f'acc_{model_name}.npy')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


def draw_performance_heatmap(accuracy, baseline, tasks, lengths, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw base heatmap in neutral background
    white_cmap = ListedColormap(["#ffffff"])
    sns.heatmap(accuracy, cmap=white_cmap, annot=True, fmt=".2f",
                xticklabels=lengths, yticklabels=tasks, ax=ax,
                cbar=False, linewidths=0.4, linecolor='lightgray',  annot_kws={"fontsize": 18})


    for (i, j), val in np.ndenumerate(accuracy):
        base_val = baseline[i, j]
        if val > base_val:
            cell_color = (0.85, 1.0, 0.85, 1.0)  # soft green
        elif val < base_val:
            cell_color = (1.0, 0.85, 0.85, 1.0)  # soft red
        else:
            continue
        ax.add_patch(Rectangle((j, i), 1, 1, fill=True, color=cell_color, lw=0))
        ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='black', linewidth=0.8))

    # Axis and title
    ax.set_xlabel('Context Length', labelpad=10, fontsize=20)
    ax.set_ylabel('Tasks', labelpad=10, fontsize=20)
    plt.xticks(rotation=0, fontsize=18)
    plt.yticks(rotation=0, fontsize=18)

    # Save and show
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()


def draw_ratio_heatmap(accuracy, baseline, tasks, lengths, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    ratio = np.zeros_like(accuracy, dtype=np.float32)
    mask = np.zeros_like(accuracy, dtype=bool)
    color_matrix = np.empty(accuracy.shape, dtype=object)
    base_green = np.array([144, 238, 144]) / 255
    dark_green = np.array([0, 100, 0]) / 255  # dark forest green
    light_red = np.array([255, 182, 193]) / 255     # light pink
    dark_red = np.array([139, 0, 0]) / 255          # dark red

    for (i, j), model_val in np.ndenumerate(accuracy):
        base_val = baseline[i, j]
        if base_val == 0 or model_val == 0:
            ratio[i, j] = 0
            color_matrix[i, j] = "#ffffff"
            continue

        max_val = max(model_val, base_val)
        min_val = min(model_val, base_val)
        ratio_val = max_val / min_val
        ratio[i, j] = ratio_val
        
        if model_val > base_val:
            # green hue: low = light green, high = dark green
            intensity = min(1.0, np.sqrt((ratio_val - 1) / 2))
            #intensity = min(1.0, (ratio_val - 1) / 5)
            color = (1 - intensity) * base_green + intensity * dark_green
            #color = (1 - intensity, 1, 1 - intensity)  # RGB for greenish
        else:
            # red hue: low = light red, high = dark red
            intensity = min(1.0, np.sqrt((ratio_val - 1) / 2))
            #intensity = min(1.0, (ratio_val - 1) / 5)
            color = (1 - intensity) * light_red + intensity * dark_red
            #color = (1, 1 - intensity, 1 - intensity)  # RGB for reddish

        color_matrix[i, j] = color

    # Draw base heatmap with ratio values
    sns.heatmap(ratio, annot=True, fmt=".2f", cmap=ListedColormap(["#ffffff"]),
                xticklabels=lengths, yticklabels=tasks, ax=ax,
                cbar=False, linewidths=0.4, linecolor='lightgray', annot_kws={"fontsize": 18})

    # Overlay colored cells manually
    for (i, j), color in np.ndenumerate(color_matrix):
        ax.add_patch(Rectangle((j, i), 1, 1, fill=True, color=color, lw=0))
        ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='black', linewidth=0.8))

    # Labels and title
    ax.set_xlabel('Context Length', labelpad=10, fontsize=20)
    ax.set_ylabel('Tasks', labelpad=10, fontsize=20)
    plt.xticks(rotation=0, fontsize=18)
    plt.yticks(rotation=0, fontsize=18)

    # Save and show
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()

def main():
    args = parse_args()
    baseline_model = args.model_name
    baseline_model = baseline_model.replace('diffmamba', 'mamba') if baseline_model.startswith('diff') else baseline_model.replace('mamba', 'diffmamba')
    
    if 'needle' in args.model_name:
        accuracies = []
        baselines = []
        for seed in args.seeds:
            accuracies.append(load_accuracy_matrix(args.results_folder, args.model_name + f'-seed{seed}'))
            baselines.append(load_accuracy_matrix(args.results_folder, baseline_model + f'-seed{seed}'))
    
        accuracy = np.mean(accuracies, axis=0)
        baseline = np.mean(baselines, axis=0)
    else:
        accuracy = load_accuracy_matrix(args.results_folder, args.model_name)
        baseline = load_accuracy_matrix(args.results_folder, baseline_model)
    

    if args.output_file is None:
        if 'pg19' not in args.model_name:
            output_file = f'{args.results_folder}/compare-{args.model_name.split("-")[2]}-seeds-{"-".join(str(seed) for seed in args.seeds)}'
        else:
            output_file = f'{args.results_folder}/compare-{args.model_name.split("-")[2]}'
    else:
        output_file = args.output_file

    if args.ratio:
        output_file += '-ratio.pdf'
        draw_ratio_heatmap(accuracy, baseline, args.tasks, args.lengths, output_file)
    else:
        output_file += '.pdf'
        draw_performance_heatmap(accuracy, baseline, args.tasks, args.lengths, output_file)

if __name__ == "__main__":
    main()


