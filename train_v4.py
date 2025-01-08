from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification, TrainerCallback
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
import json

class LRSchedulerCallback(TrainerCallback):
    def __init__(self):
        self.lr_logs = []

    def on_step_end(self, args, state, control, **kwargs):
        if hasattr(args, 'learning_rate'):
            self.lr_logs.append(args.learning_rate)

class LoggingCallback(TrainerCallback):
    def __init__(self):
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.logs.append((state.epoch, logs["loss"]))

def compute_metrics(p):
    predictions, labels = p
    preds = torch.argmax(torch.tensor(predictions), dim=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

def validate_csv_data(dataset):
    required_columns = {"text", "label"}
    if not required_columns.issubset(dataset.column_names):
        raise ValueError(f"CSV 文件缺少必要欄位: {required_columns - set(dataset.column_names)}")

def save_logs(logs, file_name):
    with open(file_name, 'w') as f:
        json.dump(logs, f)
    print(f"訓練記錄已保存至 {file_name}")

def train_keyword_model(csv_path, model_output_path):
    """
    使用多語言數據集訓練模型，並顯示收斂曲線。
    Args:
        csv_path (str): 數據集的 CSV 文件路徑。
        model_output_path (str): 微調後的模型保存路徑。
    """
    try:
        # 加載數據集
        dataset = load_dataset("csv", data_files=csv_path)
        dataset = dataset["train"].train_test_split(test_size=0.2)

        # 檢查數據有效性
        validate_csv_data(dataset["train"])

        # 加載多語言預訓練模型和分詞器
        model_name = "bert-base-multilingual-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        # 分詞處理
        tokenized_datasets = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

        # 訓練參數
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=1e-5,
            per_device_train_batch_size=32,
            num_train_epochs=10,
            save_total_limit=2,
            load_best_model_at_end=True,
            logging_dir='./logs',
            logging_steps=10,
            warmup_steps=3000,
            lr_scheduler_type="linear",
        )

        # 初始化回調
        logging_callback = LoggingCallback()
        lr_scheduler_callback = LRSchedulerCallback()

        # 初始化 Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[logging_callback, lr_scheduler_callback],
        )

        # 開始訓練
        trainer.train()

        # 保存微調後的模型
        model.save_pretrained(model_output_path)
        
        tokenizer.save_pretrained(model_output_path)
        print(f"模型訓練完成並保存至：{model_output_path}")

        # 繪製收斂曲線
        if logging_callback.logs:
            epochs, losses = zip(*logging_callback.logs)
            plt.plot(epochs, losses, label="Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss Convergence")
            plt.legend()
            plt.savefig("training_convergence1.png")
            print("收斂曲線已保存至 training_convergence1.png")

        # 保存學習率記錄
        if lr_scheduler_callback.lr_logs:
            save_logs(lr_scheduler_callback.lr_logs, "lr_logs.json")
        else:
            print("未記錄到學習率變化。")

    except Exception as e:
        print(f"訓練過程中出錯: {e}")

if __name__ == "__main__":
    csv_path = "keyword_dataset.csv"  # 包含中英文關鍵字的數據集文件路徑
    model_output_path = "./fine_tuned_model"  # 保存模型的路徑

    # 訓練模型
    train_keyword_model(csv_path, model_output_path)
