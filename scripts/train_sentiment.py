import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

# ========= 1. 基本配置 =========
DATASET_NAME = "dirtycomputer/weibo_senti_100k"
CACHE_DIR = "../.cache/huggingface"
MODEL_NAME = "google-bert/bert-base-chinese"
OUTPUT_DIR = "../models/weibo_sentiment_bert"

TEXT_COL = "review"      # 改成你的真实文本列
LABEL_COL = "label"    # 改成你的真实标签列

MAX_LENGTH = 128
SEED = 42


# ========= 2. 加载数据 =========
raw_dataset = load_dataset(DATASET_NAME, cache_dir=CACHE_DIR)

# 如果数据集只有一个 split，就自己切分
if "train" in raw_dataset and len(raw_dataset.keys()) == 1:
    full_ds = raw_dataset["train"]
elif "train" in raw_dataset and "test" in raw_dataset:
    # 如果已有 train/test，这里再从 train 里切一部分做 val
    train_valid = raw_dataset["train"].train_test_split(test_size=0.1, seed=SEED)
    dataset = {
        "train": train_valid["train"],
        "validation": train_valid["test"],
        "test": raw_dataset["test"],
    }
else:
    # 没有标准 split 时，取第一个 split
    first_split = list(raw_dataset.keys())[0]
    full_ds = raw_dataset[first_split]

# 如果前面没有直接构建 dataset，就从 full_ds 自己切
if "dataset" not in locals():
    train_test = full_ds.train_test_split(test_size=0.2, seed=SEED)
    test_valid = train_test["test"].train_test_split(test_size=0.5, seed=SEED)
    dataset = {
        "train": train_test["train"],
        "validation": test_valid["train"],
        "test": test_valid["test"],
    }

print("Dataset split sizes:")
for split_name, ds in dataset.items():
    print(split_name, ds.num_rows, ds.column_names)


# ========= 3. tokenizer =========
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    return tokenizer(
        examples[TEXT_COL],
        truncation=True,
        max_length=MAX_LENGTH,
    )

# batched=True 通常更高效
tokenized_dataset = {}
for split_name, ds in dataset.items():
    tokenized_dataset[split_name] = ds.map(
        preprocess_function,
        batched=True,
        remove_columns=[col for col in ds.column_names if col != LABEL_COL],
    )
    # Trainer 默认读 "labels"
    tokenized_dataset[split_name] = tokenized_dataset[split_name].rename_column(LABEL_COL, "labels")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# ========= 4. 指标 =========
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_metric.compute(predictions=preds, references=labels)
    f1 = f1_metric.compute(predictions=preds, references=labels, average="binary")

    return {
        "accuracy": acc["accuracy"],
        "f1": f1["f1"],
    }


# ========= 5. 模型 =========
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "negative", 1: "positive"},
    label2id={"negative": 0, "positive": 1},
)


# ========= 6. 训练参数 =========
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    seed=SEED,
    report_to="none",
)


# ========= 7. Trainer =========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ========= 8. 开始训练 =========
trainer.train()

# ========= 9. 保存模型 =========
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ========= 10. 在 test 上评估 =========
test_metrics = trainer.evaluate(tokenized_dataset["test"])
print("\nTest metrics:")
print(test_metrics)