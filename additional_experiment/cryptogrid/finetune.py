"""
Phase II: fine-tuning the protein language model itself.

Phase I froze ESM-2 and asked what a spatial residual adds on top of it. The obvious
reviewer question is whether that residual survives once the language model is adapted
to the task, since the published state of the art on CryptoBench fine-tunes a larger
ESM-2. This module provides the fine-tuning machinery; scripts/23 drives it.

What is here
------------
* ChainSample / build_samples : one (protein, chain) sequence with its residue indices
  and labels, built from the SAME struct cache the rest of the suite uses, so residue
  order is identical to work/cache_struct and to every existing prediction file.
* LoRALinear / inject_lora    : low-rank adapters on the attention query/value
  projections. Default mode, because it fits ESM2-650M (and ESM2-3B with gradient
  checkpointing) on a single consumer GPU and is far less prone to catastrophic
  forgetting than full fine-tuning on 885 proteins.
* FineTuneModel               : encoder + the Phase-I head (two Linear-ReLU-Dropout
  blocks), so the head architecture is unchanged and the only new variable is whether
  the encoder adapts.
* train_finetune / predict    : training loop with class-weighted BCE, discriminative
  learning rates, AMP and random-window cropping; sliding-window inference identical in
  spirit to scripts/02 so that embeddings stay comparable.

Leakage note
------------
A fine-tuned encoder has seen the labels of its training folds. Its embeddings are
therefore NOT exchangeable across folds: embeddings for the proteins of official fold f
must come from a model trained on the other three folds, and embeddings for the test
proteins from a model trained on all 885. scripts/23 enforces exactly this when it
writes the Phase-II embedding cache; do not reuse a single fine-tuned encoder for
everything.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .common import residue_chain

AA_FIX = str.maketrans({"U": "X", "O": "X", "B": "X", "Z": "X", "J": "X"})


# =====================================================================
# Samples
# =====================================================================

@dataclass
class ChainSample:
    pdb_id: str
    chain: str
    seq: str
    idx: np.ndarray          # positions of these residues inside the protein's cache arrays
    y: np.ndarray            # labels aligned to idx


def build_samples(store, pdb_ids: Sequence[str], with_labels: bool = True) -> List[ChainSample]:
    """One sample per (protein, chain), in cache order. Uses struct-cache `aa` + `residue_keys`."""
    out = []
    for pid in pdb_ids:
        z = np.load(store.ctx.struct_dir / f"{pid}.npz", allow_pickle=False)
        keys = z["residue_keys"].astype(str)
        aas = z["aa"].astype(str)
        y = z["y"].astype(np.float32) if with_labels else np.zeros(len(keys), dtype=np.float32)
        chains = list(dict.fromkeys(residue_chain(k) for k in keys))
        for ch in chains:
            idx = np.array([i for i, k in enumerate(keys) if residue_chain(k) == ch], dtype=np.int64)
            seq = "".join(aas[idx].tolist()).translate(AA_FIX)
            if len(seq) == 0:
                continue
            out.append(ChainSample(pid, ch, seq, idx, y[idx]))
    return out


def pos_weight_of_samples(samples: Sequence[ChainSample]) -> float:
    pos = float(sum(s.y.sum() for s in samples))
    tot = float(sum(len(s.y) for s in samples))
    return (tot - pos) / max(1.0, pos)


# =====================================================================
# LoRA
# =====================================================================

class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable rank-r update: y = Wx + (alpha/r) B A x."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout)
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)                      # start exactly at the pretrained model

    def forward(self, x):
        return self.base(x) + self.drop(x) @ self.A.t() @ self.B.t() * self.scaling


def inject_lora(encoder: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.05,
                targets: Sequence[str] = ("query", "value")) -> int:
    """Freeze the encoder and replace the named attention projections by LoRA wrappers."""
    for p in encoder.parameters():
        p.requires_grad_(False)
    n = 0
    for module in encoder.modules():
        for name in targets:
            child = getattr(module, name, None)
            if isinstance(child, nn.Linear) and not isinstance(child, LoRALinear):
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                n += 1
    if n == 0:
        raise RuntimeError(f"no LoRA targets {targets} found in the encoder")
    return n


def unfreeze_top_layers(encoder: nn.Module, n_layers: int) -> None:
    """Full fine-tuning of the top n transformer layers only (alternative to LoRA)."""
    for p in encoder.parameters():
        p.requires_grad_(False)
    layers = encoder.encoder.layer
    for layer in layers[len(layers) - n_layers:]:
        for p in layer.parameters():
            p.requires_grad_(True)


# =====================================================================
# Model
# =====================================================================

class FineTuneModel(nn.Module):
    """ESM-2 encoder + the Phase-I residue head. `forward` returns logits and hidden states."""

    def __init__(self, encoder, hidden: int = 256, dropout: float = 0.30):
        super().__init__()
        self.encoder = encoder
        d = int(encoder.config.hidden_size)
        self.trunk = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden, 1)
        self.d_embed = d

    def embed(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return h[:, 1:-1]                              # drop <cls> and <eos>

    def forward(self, input_ids, attention_mask):
        e = self.embed(input_ids, attention_mask)
        return self.out(self.trunk(e)).squeeze(-1), e


def trainable_report(model: nn.Module) -> str:
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    return f"{tr:,} trainable / {tot:,} total ({100 * tr / max(1, tot):.2f}%)"


# =====================================================================
# Batching
# =====================================================================

def encode_window(tokenizer, seq: str, device) -> Tuple[torch.Tensor, torch.Tensor]:
    t = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    return t["input_ids"].to(device), t["attention_mask"].to(device)


def random_window(sample: ChainSample, max_tokens: int, rng: random.Random) -> Tuple[str, np.ndarray]:
    """Random contiguous crop for training; whole chain if it already fits."""
    n = len(sample.seq)
    if n <= max_tokens:
        return sample.seq, np.arange(n)
    start = rng.randrange(0, n - max_tokens + 1)
    sl = np.arange(start, start + max_tokens)
    return sample.seq[start:start + max_tokens], sl


# =====================================================================
# Training and inference
# =====================================================================

def train_finetune(model, tokenizer, samples: Sequence[ChainSample], *, device, epochs: int,
                   lr_head: float, lr_encoder: float, weight_decay: float, max_tokens: int,
                   accum: int, seed: int, amp: bool = True, tag: str = "") -> None:
    pw = torch.tensor(pos_weight_of_samples(samples), device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    enc_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": lr_encoder}, {"params": head_params, "lr": lr_head}],
        weight_decay=weight_decay)
    steps = max(1, epochs * math.ceil(len(samples) / max(1, accum)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_encoder, lr_head], total_steps=steps,
                                                pct_start=0.1, anneal_strategy="cos")
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = random.Random(seed)
    order = list(range(len(samples)))
    done_steps = 0
    for ep in range(1, epochs + 1):
        rng.shuffle(order)
        model.train()
        tot, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for k, i in enumerate(order, 1):
            s = samples[i]
            sub, sl = random_window(s, max_tokens, rng)
            ids, mask = encode_window(tokenizer, sub, device)
            y = torch.from_numpy(s.y[sl]).to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                z, _ = model(ids, mask)
                loss = bce(z[0, :len(sub)], y) / accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"{tag}: non-finite loss on {s.pdb_id}/{s.chain}")
            scaler.scale(loss).backward()
            tot += float(loss.detach()) * accum
            nb += 1
            if k % accum == 0 or k == len(order):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if done_steps < steps - 1:
                    sched.step()
                    done_steps += 1
        print(f"  [{tag}] epoch {ep}/{epochs} loss={tot / max(1, nb):.5f}", flush=True)


@torch.no_grad()
def predict_samples(model, tokenizer, samples: Sequence[ChainSample], *, device, max_tokens: int,
                    overlap: int = 128, want_embeddings: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
    """Sliding-window inference; windows are averaged exactly as in scripts/02.

    Returns {pdb_id: {"z": (n,), "emb": (n, d) or None}} with arrays in cache order.
    """
    model.eval()
    out: Dict[str, Dict[str, np.ndarray]] = {}
    use_amp = device.type == "cuda"
    for s in samples:
        n = len(s.seq)
        zc = np.zeros(n, dtype=np.float64)
        ec = np.zeros((n, model.d_embed), dtype=np.float32) if want_embeddings else None
        cnt = np.zeros(n, dtype=np.float64)
        step = max(1, max_tokens - overlap)
        for st in range(0, n, step):
            en = min(n, st + max_tokens)
            ids, mask = encode_window(tokenizer, s.seq[st:en], device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                z, e = model(ids, mask)
            zc[st:en] += z[0, :en - st].float().cpu().numpy()
            if want_embeddings:
                ec[st:en] += e[0, :en - st].float().cpu().numpy()
            cnt[st:en] += 1.0
            if en == n:
                break
        cnt[cnt == 0] = 1.0
        zc /= cnt
        if want_embeddings:
            ec /= cnt[:, None].astype(np.float32)
        d = out.setdefault(s.pdb_id, {})
        d.setdefault("_n", None)
        d.setdefault("parts", []).append((s.idx, zc, ec))
    # reassemble per protein in cache order
    final = {}
    for pid, d in out.items():
        n_total = int(max(idx.max() for idx, _, _ in d["parts"]) + 1)
        z = np.zeros(n_total, dtype=np.float64)
        emb = np.zeros((n_total, model.d_embed), dtype=np.float32) if want_embeddings else None
        for idx, zc, ec in d["parts"]:
            z[idx] = zc
            if want_embeddings:
                emb[idx] = ec
        final[pid] = {"z": z, "emb": emb}
    return final
