import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertModel

# === Load and preprocess data ===
df = pd.read_csv("simulated_error_logs.csv")
df = df.fillna(0.0)  # Replace NaN with 0.0

LABELS = list(df.columns[1:])  # all label columns
NUM_LABELS = len(LABELS)

# Split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# === Tokenizer and dataset ===
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class LogDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe["log"].tolist()
        self.labels = dataframe[LABELS].values.astype("float32")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx])
        }

train_dataset = LogDataset(train_df)
val_dataset = LogDataset(val_df)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# === Define the model ===
class MultiLabelClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        output = self.dropout(output)
        return self.classifier(output)

model = MultiLabelClassifier(NUM_LABELS)

# === Training setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss()

# === Training loop ===
EPOCHS = 500
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# === Evaluation (optional preview) ===
model.eval()
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask)
        probs = torch.sigmoid(outputs)
        print(probs[:2])  # Preview first few predictions
        break

def print_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("📦 模型参数信息:")
    print(f"   🔧 总参数量: {total_params:,}")
    print(f"   🧠 可训练参数: {trainable_params:,}")

def predict_error_log(log_text):
    print_model_info(model)

    model.eval()
    inputs = tokenizer(log_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # 获取概率前 3 高的索引
    top3_indices = probs.argsort()[-3:][::-1]

    # ANSI 颜色代码
    color_map = {
        top3_indices[0]: "\033[91m",  # 红
        top3_indices[1]: "\033[93m",  # 橙（黄）
        top3_indices[2]: "\033[95m",  # 紫
    }
    endc = "\033[0m"

    print(f"\n📝 输入日志：{log_text}")
    print("🔎 预测错误类型概率分布：")
    for i, (label, prob) in enumerate(zip(LABELS, probs)):
        color = color_map.get(i, "")
        print(f"{color}  {label:<20} =>  {prob:.3f}{endc}")


# 示例：手动输入一条日志进行预测
predict_error_log("Unable to reach server and token has expired, try login again")
