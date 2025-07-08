import os
from datasets import load_dataset 

dataset = load_dataset("wikitext", "wikitext-103-v1")
output_folder = "../../data/wt103"
os.makedirs(output_folder, exist_ok=True)

# Save each split as a text file
for split in dataset.keys():
    if split.startswith('val'):
        split = 'valid.txt'
    file_path = os.path.join(output_folder, f"{split}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        for line in dataset[split]['text']:
            f.write(line + "\n")

print(f"Dataset saved in '{output_folder}'")
