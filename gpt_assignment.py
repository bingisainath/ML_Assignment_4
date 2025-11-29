"""
CS7CS4 / CSU44061 - Week 9 Assignment
Transformer / GPT Language Model Experiments
Author: Your Name

Supports:
- 3 downsized GPT configurations (<1M parameters)
- Bias toggle in attention (--use_bias)
- Skip/residual connection toggle (--no_residual)
- Evaluation mode (--eval_only --ckpt_path)
- Generates sample text
"""

import math
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F

# -------------------------------
# Command line arguments
# -------------------------------

parser = argparse.ArgumentParser(description="Week 9 GPT Assignment")
parser.add_argument("--config", type=str, default="config1",
                    choices=["config1", "config2", "config3"],
                    help="Which downsized configuration to use")
parser.add_argument("--data_path", type=str,
                    default="input_childSpeech_trainingSet.txt",
                    help="Training data file")
parser.add_argument("--block_size", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--max_iters", type=int, default=1000)
parser.add_argument("--eval_interval", type=int, default=200)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--use_bias", action="store_true")
parser.add_argument("--no_residual", action="store_true")
parser.add_argument("--eval_only", action="store_true")
parser.add_argument("--ckpt_path", type=str, default="")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)



# -------------------------------
# Model configuration presets
# -------------------------------

def get_model_config(name: str):
    """
    Returns model hyperparameters for the 3 downsized configs.
    All < 1M parameters.
    """

    base = dict(
        block_size=args.block_size,
        dropout=args.dropout,
        use_bias=args.use_bias,
        no_residual=args.no_residual,
    )
    if name == "config1":
        base.update(dict(n_embd=128, n_layer=4, n_head=4))
    elif name == "config2":
        base.update(dict(n_embd=256, n_layer=2, n_head=4))
    elif name == "config3":
        base.update(dict(n_embd=192, n_layer=4, n_head=3))
    return base



# -------------------------------
# Data loading utilities
# -------------------------------

def load_text_file(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return p.read_text(encoding="utf-8")


def build_vocab(text: str):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode(text, stoi):
    return [stoi[c] for c in text if c in stoi]


def decode(indices, itos):
    return "".join(itos[i] for i in indices)


def create_splits(data_tensor, train_frac=0.9):
    n = int(train_frac * len(data_tensor))
    return data_tensor[:n], data_tensor[n:]



# -------------------------------
# Batching utilities
# -------------------------------

def get_batch(split, train_data, val_data, block_size, batch_size):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters=200):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data, block_size, batch_size)
            _, loss = model(X, Y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out



# -------------------------------
# Transformer Model
# -------------------------------

class Head(nn.Module):
    """One self-attention head."""

    def __init__(self, head_size, n_embd, block_size, dropout, use_bias):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=use_bias)
        self.query = nn.Linear(n_embd, head_size, bias=use_bias)
        self.value = nn.Linear(n_embd, head_size, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout, use_bias):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, n_embd, block_size, dropout, use_bias) 
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(x))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Full transformer block."""

    def __init__(self, n_embd, n_head, block_size, dropout, use_bias, no_residual):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout, use_bias)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.no_residual = no_residual

    def forward(self, x):
        if self.no_residual:   # No skip connections
            x = self.sa(self.ln1(x))
            x = self.ffwd(self.ln2(x))
        else:
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        n_embd = config["n_embd"]
        n_layer = config["n_layer"]
        n_head = config["n_head"]
        block_size = config["block_size"]
        dropout = config["dropout"]
        use_bias = config["use_bias"]
        no_residual = config["no_residual"]

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout, use_bias, no_residual)
              for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx



# -------------------------------
# Training Loop
# -------------------------------

def train_model():
    raw_text = load_text_file(args.data_path)
    stoi, itos = build_vocab(raw_text)
    vocab_size = len(stoi)

    data = torch.tensor(encode(raw_text, stoi), dtype=torch.long)
    train_data, val_data = create_splits(data)

    config = get_model_config(args.config)
    model = GPTLanguageModel(vocab_size, config).to(device)

    print("Using config:", args.config)
    print(config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for iter in range(args.max_iters + 1):
        if iter % args.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data,
                                   args.block_size, args.batch_size)
            print(f"step {iter} | train {losses['train']:.4f} | val {losses['val']:.4f}")

        xb, yb = get_batch("train", train_data, val_data,
                           args.block_size, args.batch_size)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save checkpoint
    ckpt_name = f"ckpt_{args.config}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "config": config,
    }, ckpt_name)
    print(f"Checkpoint saved to {ckpt_name}")

    # Generate some text
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(start, 200)[0].tolist()
    print("Generated Text:\n", decode(out, itos))



# -------------------------------
# Test Set Evaluation
# -------------------------------

@torch.no_grad()
def evaluate_on_file(model, stoi, itos, filename, desc):
    text = load_text_file(filename)
    encoded = encode(text, stoi)
    data = torch.tensor(encoded, dtype=torch.long, device=device)

    block_size = model.block_size
    losses = []

    for i in range(0, len(data) - block_size - 1, block_size):
        x = data[i : i + block_size].unsqueeze(0)
        y = data[i + 1 : i + block_size + 1].unsqueeze(0)
        _, loss = model(x, y)
        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    print(f"[{desc}] Loss: {avg_loss:.4f}")
    return avg_loss



def eval_only():
    ckpt = torch.load(args.ckpt_path, map_location=device)
    stoi = ckpt["stoi"]
    itos = ckpt["itos"]
    config = ckpt["config"]
    vocab_size = len(stoi)

    model = GPTLanguageModel(vocab_size, config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    evaluate_on_file(model, stoi, itos,
                     "input_childSpeech_testSet.txt",
                     "Child Speech Test Set")
    evaluate_on_file(model, stoi, itos,
                     "input_shakespeare.txt",
                     "Shakespeare Corpus")



if __name__ == "__main__":
    if args.eval_only:
        eval_only()
    else:
        train_model()
