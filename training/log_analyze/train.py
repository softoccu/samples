import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

# === Load Data ===
df = pd.read_csv("simulated_error_logs.csv").fillna(0.0)
LABELS = list(df.columns[1:])
NUM_LABELS = len(LABELS)

# === Simple Tokenizer ===
from collections import Counter
import re

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# Build vocab
counter = Counter()
for log in df["log"]:
    counter.update(tokenize(log))
vocab = {word: i+2 for i, (word, _) in enumerate(counter.most_common(1000))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1
vocab_size = len(vocab)

def encode(text, max_len=64):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens[:max_len]]
    ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids

# === Dataset ===
class LogDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.max_len = 64

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        log_ids = encode(row["log"], self.max_len)
        labels = torch.tensor(row[LABELS].values.astype("float32"))
        return torch.tensor(log_ids), labels


class TinyTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=2, num_layers=1, num_labels=NUM_LABELS):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=128)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(self, x):
        x = self.embedding(x).transpose(0, 1)  # [seq_len, batch, dim]
        x = self.transformer(x)                # [seq_len, batch, dim]
        x = x[0]                               # [CLS token style]
        return self.classifier(x)



# === Model ===
class SimpleTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=2, num_labels=NUM_LABELS):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(self, x):
        x = self.embedding(x).transpose(0, 1)  # [seq_len, batch, dim]
        x = self.transformer(x)                # [seq_len, batch, dim]
        x = x[0]                               # Use first token (like CLS)
        return self.classifier(x)

# === Data split and loaders ===
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = LogDataset(train_df)
val_dataset = LogDataset(val_df)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# === Training ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = SimpleTransformerModel(vocab_size=vocab_size).to(device)
model = TinyTransformerModel(vocab_size=vocab_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
criterion = nn.BCEWithLogitsLoss()

EPOCHS = 1000
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")


torch.save(model.state_dict(), "tiny_transformer_model.pt")
print("✅ 模型已保存为 tiny_transformer_model.pt")


import numpy as np

def print_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("📦 模型参数信息:")
    print(f"   🔧 总参数量: {total_params:,}")
    print(f"   🧠 可训练参数: {trainable_params:,}")



def load_model_and_predict(log_text):
    # 重新初始化模型结构
    model = TinyTransformerModel(vocab_size=vocab_size).to(device)

    # 加载权重
    model.load_state_dict(torch.load("tiny_transformer_model.pt", map_location=device))
    model.eval()

    # 打印参数信息
    print_model_info(model)

    # 编码日志
    input_ids = torch.tensor([encode(log_text)]).to(device)

    # 预测
    with torch.no_grad():
        logits = model(input_ids)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # Top 3 高亮
    top3_indices = probs.argsort()[-3:][::-1]
    color_map = {
        top3_indices[0]: "\033[91m",  # 红
        top3_indices[1]: "\033[93m",  # 橙
        top3_indices[2]: "\033[95m",  # 紫
    }
    endc = "\033[0m"

    print(f"\n📝 输入日志：{log_text}")
    print("🔎 预测错误类型概率分布：")
    for i, (label, prob) in enumerate(zip(LABELS, probs)):
        color = color_map.get(i, "")
        print(f"{color}  {label:<20} =>  {prob:.3f}{endc}")



load_model_and_predict("Failed to connect to database and token has expired, try login again")
