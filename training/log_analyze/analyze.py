import torch
import torch.nn as nn
import re
import numpy as np
import pandas as pd

# === 标签列表 ===
LABELS = ['NetworkError', 'DatabaseError', 'PermissionError', 'TimeoutError',
          'LogicError', 'TokenExpired', 'CredentialExpired', 'PasswordNotMatch', 'ThisUserNotExist']
NUM_LABELS = len(LABELS)

# === Tokenizer + Vocab ===
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# 加载 CSV 来重建词表
df = pd.read_csv("simulated_error_logs.csv").fillna(0.0)
from collections import Counter
counter = Counter()
for log in df["log"]:
    counter.update(tokenize(log))
vocab = {word: i+2 for i, (word, _) in enumerate(counter.most_common(1000))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

def encode(text, max_len=64):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens[:max_len]]
    ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids

# === 模型定义 ===
class TinyTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=2, num_layers=1, num_labels=NUM_LABELS):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=128)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(self, x):
        x = self.embedding(x).transpose(0, 1)
        x = self.transformer(x)
        x = x[0]
        return self.classifier(x)

# === 参数统计 ===
def print_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📦 模型总参数量: {total_params:,}")

# === 预测函数 ===
def predict(log_text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTransformerModel(vocab_size=len(vocab)).to(device)
    model.load_state_dict(torch.load("tiny_transformer_model.pt", map_location=device))
    model.eval()

    print_model_info(model)

    input_ids = torch.tensor([encode(log_text)]).to(device)

    with torch.no_grad():
        logits = model(input_ids)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

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

# === CLI 调用 ===
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("请输入日志文本，例如：")
        print("  python predict_log.py \"Token expired while accessing server\"")
    else:
        log_input = " ".join(sys.argv[1:])
        predict(log_input)
