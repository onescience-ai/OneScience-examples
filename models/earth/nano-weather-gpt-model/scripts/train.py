"""Train nanoWeatherGPT and save model.pt + meta.pkl."""

import os
import sys
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Always run from the script's own directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import GPT, GPTConfig
from data import generate_weather_corpus

# ── Hyperparameters ──────────────────────────────────────────────────────────
BLOCK_SIZE = 128
N_EMBD     = 128
N_HEAD     = 4
N_LAYER    = 4
DROPOUT    = 0.1
BATCH_SIZE = 256
EPOCHS     = 20
LR         = 3e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data       = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx : idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1 : idx + self.block_size + 1], dtype=torch.long)
        return x, y


def train():
    print("Generating training corpus...")
    corpus = generate_weather_corpus(1000)

    # Character-level tokeniser
    chars     = sorted(set(corpus))
    vocab_size = len(chars)
    stoi      = {ch: i for i, ch in enumerate(chars)}
    itos      = {i: ch for i, ch in enumerate(chars)}

    print(f"Vocab size : {vocab_size} characters")
    print(f"Corpus size: {len(corpus):,} characters")

    encoded  = [stoi[c] for c in corpus]
    split    = int(len(encoded) * 0.9)
    train_dl = DataLoader(
        CharDataset(encoded[:split], BLOCK_SIZE),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )

    config = GPTConfig(
        block_size=BLOCK_SIZE,
        vocab_size=vocab_size,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
    )
    model = GPT(config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {n_params:,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    print(f"Training on {DEVICE} for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss: {total_loss / len(train_dl):.4f}", flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save({"model_state_dict": model.state_dict(), "config": config}, "model.pt")
    with open("meta.pkl", "wb") as f:
        pickle.dump({"stoi": stoi, "itos": itos, "vocab_size": vocab_size}, f)

    print("\nSaved model.pt and meta.pkl")


if __name__ == "__main__":
    train()
