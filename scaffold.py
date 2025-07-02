#!/usr/bin/env python3
# coding: utf-8
"""
生成 UniTrader-RL 專案骨架 —— 在目前工作目錄建立所有檔案。
執行步驟（Colab 內）：
    %cd /content/0702        # 你的 repo 根目錄
    !python scaffold.py
然後：
    !git add .
    !git commit -m "init UniTrader-RL scaffold"
    !git push
"""
from pathlib import Path
import textwrap, json, os

FILES = {
    "README.md": r"""
# UniTrader-RL

> **Unified Retrieval×Transformer + QR-DQN + SAC Hierarchical RL for Quant Trading**

本專案提供一套可在 FX / Crypto / Equity 上落地的分層強化學習交易系統…（節錄，完整版見 Canvas）
""",
    "pyproject.toml": r"""
[build-system]
requires = ["setuptools>=67", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "unitrader"
version = "0.1.0"
description = "Unified Hierarchical RL Trading Library"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "UniTrader Team", email = "noreply@example.com" }]
readme = "README.md"
keywords = ["quant", "reinforcement-learning", "transformer"]

[project.optional-dependencies]
all = [
  "torch>=2.2", "pandas", "numpy", "faiss-cpu", "scikit-learn",
  "vectorbtpro<1.0", "gymnasium", "optuna", "hydra-core", "omegaconf",
  "black", "flake8", "isort", "pytest", "yarl", "rich"
]

[tool.setuptools.packages.find]
where = ["src"]
""",
    "LICENSE": r"""
MIT License

Copyright (c) 2025 UniTrader

Permission is hereby granted, free of charge, to any person obtaining a copy
...
""",
    # --- Docker ---
    "docker/Dockerfile": r"""
FROM nvidia/cuda:12.2.0-cudnn9-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git curl build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3
WORKDIR /workspace/unitrader-rl
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -e .[all]
COPY . .
CMD ["bash"]
""",
    "docker/docker-compose.yml": r"""
version: "3.9"
services:
  trader:
    build: ./
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./:/workspace/unitrader-rl
    command: >
      bash -c "pytest -q &&
               python -m unitrader.cli.train_offline
               --config configs/default.yaml
               --data-path USDJPY_M1_2020.parquet"
""",
    # --- GitHub CI ---
    ".github/workflows/ci.yml": r"""
name: CI
on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -e .[all]
      - name: Lint & Test
        run: |
          black --check src/ tests/
          flake8 src/ tests/
          pytest -q
""",
    # --- configs ---
    "configs/default.yaml": r"""
seed: 42
dataset:
  path: ''
  timeframe: '1m'
  train_ratio: 0.8
transformer:
  seq_len: 128
  d_model: 128
  n_heads: 4
  num_layers: 4
  dropout: 0.1
  memory_k: 5000
qrdqn:
  atoms: 101
  hidden_dim: 256
  lr: 3e-4
  gamma: 0.99
  cql_alpha: 0.5
sac:
  hidden_dim: 256
  lr_actor: 3e-4
  lr_critic: 3e-4
  gamma: 0.99
  tau: 0.005
training:
  batch_size: 1024
  total_steps: 2000000
  eval_interval: 10000
""",
    "configs/colab_gpu.yaml": r"""
defaults:
  - override /transformer:
      d_model: 64
      num_layers: 2
  - override /training:
      batch_size: 512
""",
    # --- package init ---
    "src/unitrader/__init__.py": r'''"""UniTrader-RL package init."""\n__all__ = ["data", "models", "rl", "backtest"]\n''',
    # --- data loader ---
    "src/unitrader/data/loader.py": r"""
from pathlib import Path
from typing import Tuple
import pandas as pd

def load_parquet(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    df.sort_index(inplace=True)
    return df

def train_test_split(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split = int(len(df) * train_ratio)
    return df.iloc[:split], df.iloc[split:]
""",
    # --- retrieval transformer (stub) ---
    "src/unitrader/models/retrieval_transformer.py": r"""
import torch
import torch.nn as nn
from typing import Tuple

class RetrievalTransformer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        feat = h[:, -1, :]
        pred = self.head(feat)
        return pred, feat
""",
    # --- QR-DQN ---
    "src/unitrader/rl/highlevel_qrdqn.py": r"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Tuple
import torch, torch.nn as nn, torch.optim as optim

@dataclass
class QRDQNConfig:
    atoms: int = 101
    hidden_dim: int = 256
    gamma: float = 0.99
    lr: float = 3e-4

class QRDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, cfg: QRDQNConfig):
        super().__init__()
        self.atoms, self.action_dim = cfg.atoms, action_dim
        self.z = torch.linspace(0, 1, cfg.atoms)
        self.net = nn.Sequential(
            nn.Linear(state_dim, cfg.hidden_dim), nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(),
            nn.Linear(cfg.hidden_dim, action_dim * cfg.atoms)
        )
        self.opt = optim.Adam(self.parameters(), lr=cfg.lr)
        self.gamma = cfg.gamma

    def forward(self, s: torch.Tensor):
        p = self.net(s).view(-1, self.action_dim, self.atoms)
        return torch.softmax(p, -1)

    @torch.no_grad()
    def act(self, s: torch.Tensor, greedy: bool = True) -> Tuple[int, float]:
        dist = self(s)
        q = (dist * self.z).sum(-1)
        idx = torch.argmax(q, -1).item() if greedy else random.randrange(self.action_dim)
        return idx, q[0, idx].item()
""",
    # --- SAC (stub) ---
    "src/unitrader/rl/lowlevel_sac.py": r"""
import torch, torch.nn as nn, torch.optim as optim
from torch.distributions import Normal

class MLP(nn.Module):
    def __init__(self, d_in, d_out, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, d_out)
        )
    def forward(self, x): return self.net(x)

class SACAgent:
    def __init__(self, s_dim, a_dim, h=256, lr=3e-4):
        self.actor = MLP(s_dim, a_dim*2, h)
        self.critic1, self.critic2 = MLP(s_dim+a_dim,1,h), MLP(s_dim+a_dim,1,h)
        self.opt_a = optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c1 = optim.Adam(self.critic1.parameters(), lr=lr)
        self.opt_c2 = optim.Adam(self.critic2.parameters(), lr=lr)
        self.log_alpha = torch.tensor(0., requires_grad=True)
        self.opt_alpha = optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -a_dim

    def policy(self, s):
        mu, log_std = self.actor(s).chunk(2,-1)
        std = log_std.clamp(-20,2).exp()
        dist = Normal(mu, std)
        a = torch.tanh(dist.rsample())
        logp = dist.log_prob(a).sum(-1, keepdim=True)
        return a, logp
""",
    # --- backtest wrapper ---
    "src/unitrader/backtest/vectorbt_runner.py": r"""
import vectorbtpro as vbt

def run_backtest(price, entry, exit):
    pf = vbt.Portfolio.from_signals(price, entries=entry, exits=exit, freq='1min')
    print(pf.stats().iloc[:10])
    pf.plot().show()
    return pf
""",
    # --- CLI stub ---
    "src/unitrader/cli/train_offline.py": r"""
import argparse
from pathlib import Path
import torch
from hydra import compose, initialize
from unitrader.data.loader import load_parquet, train_test_split
from unitrader.models.retrieval_transformer import RetrievalTransformer
from unitrader.rl.highlevel_qrdqn import QRDQN, QRDQNConfig

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/default.yaml')
    p.add_argument('--data-path', type=str, required=True)
    args = p.parse_args()

    with initialize(version_base=None, config_path='..'):
        cfg = compose(config_name=Path(args.config).name)

    df = load_parquet(args.data_path)
    train_df, _ = train_test_split(df, cfg.dataset.train_ratio)

    model = RetrievalTransformer(
        cfg.transformer.d_model, cfg.transformer.n_heads,
        cfg.transformer.num_layers, cfg.transformer.dropout)
    dummy_x = torch.randn(32, cfg.transformer.seq_len, 10)
    _, feat = model(dummy_x)
    qrdqn = QRDQN(feat.shape[-1], 3, QRDQNConfig(**cfg.qrdqn))
    print('[OK] QR-DQN initialized.')

if __name__ == '__main__':
    main()
""",
    # --- tests ---
    "tests/test_smoke.py": r"""
import importlib
def test_import():
    assert importlib.import_module('unitrader')
"""
}

def write_file(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    print("✓", path)

def main():
    for p, c in FILES.items():
        write_file(p, c)
    # 空白 Notebook 佔位
    nb_path = Path("notebooks/QuickStart.ipynb")
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    nb_path.write_text(json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
    print("✓ notebooks/QuickStart.ipynb (blank)")
    print("\n全部檔案已生成！下一步：\n  git add . && git commit -m 'init scaffold' && git push")

if __name__ == "__main__":
    main()
