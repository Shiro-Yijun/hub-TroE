import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast,GradScaler


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        # 1. 字符embedding
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout)
        # 3. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,                        # 多头注意力头数，常用8
            dim_feedforward=hidden_dim,     # 本次神经网络用到的512
            dropout=dropout,
            batch_first=True,              # 我们用(T,B,D)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        # 4. 输出层
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self,x, mask=None):
        """
        x:(B, T)
        mask:(T, T)  下三角mask
        """
        # B, T = x.shape
        # Embedding + scale
        x = self.embed(x) * math.sqrt(self.embed_dim)  # (B, T, D)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, mask=mask)
        # # Transformer 默认(T, B, D)， 所以先转
        # x = x.transpose(0, 1)                          # (T, B, D)
        # # 位置编码
        # x = self.pos_encoder(x)
        # # Transformer Encoder + mask
        # x = self.transformer_encoder(x, mask=mask)     # (T, B, D)
        # # 转回(B, T, D)
        # x = x.transpose(0, 1)
        # # 预测每个位置下一个字符
        logits = self.fc(x)                            # (B, T, V)
        return logits

# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    # 生成一次mask，所有batch共用
    seq_len = loader.dataset.seq_len
    mask = generate_causal_mask(seq_len).to(device)

    # 只在训练模式开启混合精度
    scaler = GradScaler(enabled=train)

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # 前向传播：
        with autocast(device_type="cuda",enabled=train):
            logits = model(x, mask=mask)            # 把mask传进去
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            # 反向传播用scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────

def generate_causal_mask(seq_len):
    # 下三角mask: (seq_len, seq_len)
    mask = torch.triu(torch.ones(seq_len,seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model",      default="lstm", choices=["rnn", "lstm"])
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--embed_dim",  type=int,   default=512)    # 建议和hidden一致
    parser.add_argument("--hidden_dim", type=int,   default=512)    # 本次需要的神经网络数量512
    parser.add_argument("--num_layers", type=int,   default=6)      # 本次需要的transformer的层数6层
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=3e-4)   # Transformer常用小lr
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_transformer.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"device: {device}  model: {args.model.upper()}")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 模型
    model = LM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        # model_type=args.model,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")

# 文本生成函数
def generate_text(model, start_text, char2idx, idx2char, seq_len, device, max_gen_len=200):
    model.eval()
    ids = [char2idx[c] for c in start_text if c in char2idx]

    with torch.no_grad():
        for _ in range(max_gen_len):
            # 只取最后 seq_len 个字符，防止超长
            cur_seq = ids[-seq_len:]
            x = torch.tensor([cur_seq], device=device)

            # 关键：动态生成和当前输入等长的 mask，不会维度不匹配
            cur_len = x.size(1)
            causal_mask = generate_causal_mask(cur_len).to(device)

            logits = model(x, mask=causal_mask)
            pred_id = torch.argmax(logits[0, -1], dim=-1).item()
            ids.append(pred_id)

    return ''.join([idx2char[i] for i in ids])


# 加载模型并测试
def try_generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("best_transformer.pt", map_location=device)
    model = LM(
        vocab_size=len(checkpoint["char2idx"]),
        embed_dim=512,
        hidden_dim=512,
        num_layers=6,
        dropout=0.1
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    char2idx = checkpoint["char2idx"]
    idx2char = checkpoint["idx2char"]
    args = checkpoint["args"]

    # 自动续写，可改开头文字
    res = generate_text(model, start_text="今天", char2idx=char2idx, idx2char=idx2char, seq_len=args["seq_len"],
                        device=device)
    print("\n===== 文本生成结果 =====")
    print(res)


if __name__ == "__main__":
    main()
    try_generate()
