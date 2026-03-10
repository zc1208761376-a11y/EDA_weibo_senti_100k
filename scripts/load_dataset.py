from collections import Counter
from datasets import load_dataset
import random

def show_random_samples(ds, text_col, label_col, k=3, seed=42):
    print("\n=== random samples ===")
    random.seed(seed)

    indices = random.sample(range(ds.num_rows), k)
    for i, idx in enumerate(indices, start=1):
        row = ds[idx]
        print(f"\n--- sample {i} (idx={idx}) ---")
        print(f"label: {row[label_col]}")
        print(f"text: {row[text_col]}")

def summarize_dataset(ds, text_col, label_col):
    print("=== basic info ===")
    print(f"num_rows: {ds.num_rows}")
    print(f"columns: {ds.column_names}")

    labels = ds[label_col]
    label_counter = Counter(labels)

    print("\n=== label distribution ===")
    total = ds.num_rows
    for label, count in sorted(label_counter.items()):
        ratio = count / total
        print(f"label={label}: count={count}, ratio={ratio:.4f}")

def main():
    dataset = load_dataset("dirtycomputer/weibo_senti_100k")

    if "train" in dataset:
        ds = dataset["train"]
    else:
        first_split = list(dataset.keys())[0]
        ds = dataset[first_split]

    # 这里先根据你实际看到的字段名填写
    text_col = "review"
    label_col = "label"

    show_random_samples(ds, text_col, label_col, k=3, seed=42)
    summarize_dataset(ds, text_col, label_col)

if __name__ == "__main__":
    main()