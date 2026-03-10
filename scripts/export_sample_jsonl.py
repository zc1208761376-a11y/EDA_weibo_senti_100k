import json
import os
from datasets import load_dataset
import random

def export_sample_jsonl(ds, text_col, label_col, output_path, sample_size=100, seed=42):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    random.seed(seed)
    actual_size = min(sample_size, ds.num_rows)
    indices = random.sample(range(ds.num_rows), actual_size)

    with open(output_path, "w", encoding="utf-8") as f:
        for idx in indices:
            row = ds[idx]
            record = {
                "text": row[text_col],
                "label": row[label_col]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSaved sample to: {output_path}")
    print(f"Sample size: {actual_size}")
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
    export_sample_jsonl(
        ds,
        text_col=text_col,
        label_col=label_col,
        output_path="../data/sample.jsonl",
        sample_size=100,
        seed=42,
    )

if __name__ == "__main__":
    main()
