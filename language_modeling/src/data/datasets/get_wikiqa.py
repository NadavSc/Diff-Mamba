import os
from datasets import load_dataset 

dataset = load_dataset("microsoft/wiki_qa")
output_folder = "../../data/wikiqa"
os.makedirs(output_folder, exist_ok=True)

# Save each split as a text file
for split in dataset.keys():
    file_path = os.path.join(output_folder, f"{split}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        for qline, aline in zip(dataset[split]['question'], dataset[split]['answer']):
            f.write(qline + " Answer: " + aline + "\n")
    if split.startswith('val'):
        split = 'valid'
        updated_path = os.path.join(output_folder, f"{split}.txt")
        os.rename(file_path, updated_path)

print(f"Dataset saved in '{output_folder}'")
