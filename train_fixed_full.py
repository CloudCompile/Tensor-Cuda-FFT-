"""train_fixed_full.py

Fixed spectral mixer training on TinyStories with non-greedy sampling.

Key fix (already validated): MIX IN FREQUENCY DOMAIN.

This script:
  - trains on a large corpus via random window sampling (no 1000-seq overfit)
  - uses a JPEG/progressive frequency cutoff schedule
  - generates with temperature + top-k + repetition penalty

Run:
  python train_fixed_full.py

Notes:
  - Uses byte-level modeling (vocab=256).
  - Default seq_len=1024 (changeable).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrainConfig:
    # data
    data_path: str = "tinystories_train.txt"
    # model
    vocab_size: int = 256
    d_model: int = 512
    n_layers: int = 6
    seq_len: int = 1024  # context length
    kernel_len: int = 128  # causal conv kernel length (per block)
    ffn_mult: int = 2  # feedforward expansion factor (improves capacity)
    # training
    batch_size: int = 8
    epochs: int = 200
    steps_per_epoch: int = 250  # random-window batches per epoch
    # LR tuned for stability on full corpus (user-requested)
    lr: float = 2e-4
    weight_decay: float = 5e-4
    grad_clip: float = 1.0
    # progressive frequency schedule
    jpeg_low: int = 128
    jpeg_mid: int = 512
    jpeg_high: int = 1024
    jpeg_transition: int = 32  # soft roll-off bins to reduce Gibbs ringing
    # generation
    temperature: float = 0.8
    # Prefer nucleus sampling (top-p) over top-k for byte-level stability.
    top_p: float = 0.9
    top_k: int = 0  # optional backstop; 0 disables
    repetition_penalty: float = 1.6
    repetition_window: int = 256
    max_run_length: int = 6  # hard anti-stutter: disallow > N identical bytes
    presence_penalty: float = 0.6
    frequency_penalty: float = 0.15
    ban_cr: bool = True  # ban '\r'
    ascii_only: bool = True  # allow \n and printable ASCII only
    max_new: int = 400
    # misc
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    # checkpointing
    ckpt_path: str = "fixed_spectral_ckpt.pt"
    save_every_epochs: int = 5
    # evaluation / anti-parroting
    val_windows: int = 2048
    val_batches: int = 20
    parroting_snip_len: int = 64
    parroting_stride: int = 16
    parroting_snips: int = 64  # number of snippets to test
    log_every_steps: int = 50  # per-epoch step progress (prevents "stuck" feeling)

    # Sawtooth LR schedule (cosine annealing with stage-aligned restarts)
    stage1_epochs: int = 5
    stage2_epochs: int = 10  # epochs 5..14
    stage1_lr_mult: float = 1.0
    stage1_min_mult: float = 0.1
    stage2_lr_mult: float = 0.5
    stage2_min_mult: float = 0.05
    stage3_lr_mult: float = 0.25
    stage3_min_mult: float = 0.01


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_corpus_as_u8(path: str, *, sanitize_ascii: bool) -> torch.Tensor:
    # Read bytes (utf-8 file) and keep raw bytes; this is intentional.
    # We model bytes, so we just take file bytes modulo 256.
    with open(path, "rb") as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    if sanitize_ascii:
        # Keep \n and printable ASCII. Everything else -> space.
        keep = (arr == 10) | ((arr >= 32) & (arr <= 126))
        arr = np.where(keep, arr, 32).astype(np.uint8)
    # Keep on CPU; we sample windows and then move to GPU.
    return torch.from_numpy(arr.copy())  # make contiguous, owns memory


def make_val_starts(n_bytes: int, seq_len: int, count: int, seed: int) -> torch.Tensor:
    """Deterministic validation window start indices."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    hi = max(1, n_bytes - (seq_len + 1) - 1)
    return torch.randint(0, hi, (count,), generator=g)


@torch.no_grad()
def eval_loss(
    model: nn.Module,
    corpus_u8: torch.Tensor,
    starts: torch.Tensor,
    cfg: TrainConfig,
    cutoff: int | None,
) -> float:
    """Compute an approximate validation loss on fixed windows."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    n = int(corpus_u8.numel())
    # sample a subset of starts each call
    idx = torch.randperm(starts.numel())[: cfg.val_batches * cfg.batch_size]
    sel = starts[idx]
    losses = []
    for i in range(0, sel.numel(), cfg.batch_size):
        s = sel[i : i + cfg.batch_size]
        if s.numel() < cfg.batch_size:
            break
        bx = torch.stack([corpus_u8[j : j + cfg.seq_len] for j in s]).to(torch.long)
        by = torch.stack([corpus_u8[j + 1 : j + cfg.seq_len + 1] for j in s]).to(torch.long)
        bx = bx.to(cfg.device, non_blocking=True)
        by = by.to(cfg.device, non_blocking=True)
        logits = model(bx, cutoff=cutoff)
        loss = loss_fn(logits.reshape(-1, cfg.vocab_size), by.reshape(-1))
        losses.append(float(loss.item()))
    return float(sum(losses) / max(1, len(losses)))


def parroting_score(corpus_bytes: bytes, gen_bytes: bytes, cfg: TrainConfig) -> float:
    """Heuristic: fraction of random fixed-length snippets from the generation that occur verbatim in corpus.

    High score means the model is likely copying/memorizing; low score suggests novelty.
    """
    if len(gen_bytes) < cfg.parroting_snip_len + 1:
        return 0.0
    # ignore the prompt prefix by skipping the first 32 bytes
    start0 = min(32, len(gen_bytes) - cfg.parroting_snip_len)
    candidates = list(range(start0, len(gen_bytes) - cfg.parroting_snip_len, cfg.parroting_stride))
    if not candidates:
        return 0.0
    # deterministic sampling
    rng = np.random.default_rng(123)
    picks = rng.choice(candidates, size=min(cfg.parroting_snips, len(candidates)), replace=False)
    hits = 0
    for p in picks:
        snip = gen_bytes[p : p + cfg.parroting_snip_len]
        if corpus_bytes.find(snip) != -1:
            hits += 1
    return hits / float(len(picks))


def jpeg_cutoff(epoch: int, cfg: TrainConfig, freq_bins: int) -> int:
    # Expand horizon: low -> mid -> high -> full.
    # Values are in "frequency bins" units. For rfft(seq_len), bins = seq_len//2+1.
    if epoch < 20:
        target = cfg.jpeg_low
    elif epoch < 50:
        target = cfg.jpeg_mid
    elif epoch < 100:
        target = cfg.jpeg_high
    else:
        target = freq_bins
    return int(min(target, freq_bins))


def sawtooth_lr(global_step: int, epoch: int, cfg: TrainConfig) -> float:
    """Cosine annealing with restarts aligned to the curriculum stages.

    base LR = cfg.lr. Within each stage, LR decays from (base*stage_lr_mult)
    down to (base*stage_min_mult).
    """
    s_per = int(cfg.steps_per_epoch)
    e1 = int(cfg.stage1_epochs)
    e2 = int(cfg.stage1_epochs + cfg.stage2_epochs)

    if epoch < e1:
        stage_start = 0
        stage_epochs = max(1, e1)
        lr_mult = cfg.stage1_lr_mult
        min_mult = cfg.stage1_min_mult
    elif epoch < e2:
        stage_start = e1 * s_per
        stage_epochs = max(1, int(cfg.stage2_epochs))
        lr_mult = cfg.stage2_lr_mult
        min_mult = cfg.stage2_min_mult
    else:
        stage_start = e2 * s_per
        stage_epochs = max(1, int(cfg.epochs) - e2)
        lr_mult = cfg.stage3_lr_mult
        min_mult = cfg.stage3_min_mult

    stage_total_steps = max(1, stage_epochs * s_per)
    local_step = max(0, int(global_step) - int(stage_start))
    progress = min(1.0, local_step / float(stage_total_steps))

    # cosine from 1 -> 0
    cos01 = 0.5 * (1.0 + math.cos(math.pi * progress))
    mult = float(min_mult + (lr_mult - min_mult) * cos01)
    return float(cfg.lr * mult)


def curriculum_cutoff(epoch: int, cfg: TrainConfig, freq_bins: int) -> int:
    """Spectral Curriculum Learning (strict stages).

    Stage 1 (epoch 0-4): 128 bins  (vibe / flow)
    Stage 2 (epoch 5-14): 256 bins (structure / word forms)
    Stage 3 (epoch 15+): 512 bins  (detail / spelling)

    Values are capped to available freq_bins.
    """
    if epoch < 5:
        target = 128
    elif epoch < 15:
        target = 256
    else:
        target = 512
    return int(min(target, freq_bins))


class FixedSpectralBlock(nn.Module):
    """A single *causal* spectral mixing block.

    IMPORTANT: The earlier non-causal frequency filter leaks FUTURE tokens during training
    (because FFT mixes the whole window). That gives artificially low train loss and
    degenerates at generation time.

    Fix: implement a causal linear convolution via FFT (zero-padded), using a one-sided
    time-domain kernel.
    """

    def __init__(self, d_model: int, seq_len: int, kernel_len: int, transition_bins: int, dropout: float = 0.1):
        super().__init__()
        # Pre-norm tends to stabilize spectral ops.
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        self.seq_len = seq_len
        self.kernel_len = kernel_len
        self.transition_bins = int(max(1, transition_bins))
        # Learnable *causal* kernel in time domain: k[0..K-1]
        self.kernel = nn.Parameter(torch.zeros(kernel_len))
        nn.init.normal_(self.kernel, mean=0.0, std=0.001)  # identity-ish

        # Per-channel gain so channels aren't forced to share the exact same filter
        self.gain = nn.Parameter(torch.ones(d_model))

        # ------------------------------------------------------------
        # GATING ("Valve")
        # ------------------------------------------------------------
        # We gate the frequency-domain signal to prevent resonant attractors
        # (e.g., "888888" loops) from dominating.
        #
        # Two gates:
        #  1) Per-frequency gate: learns which spectral bands to suppress globally.
        #  2) Context gate: learns per-channel gating conditioned on current hidden state.
        #
        # Gate values are real in [0,1] and multiply complex spectra (scales real+imag).

        # Max rFFT bins given the largest FFT length this block will use.
        # n_fft is next power-of-2 >= (T + K - 1). For maximum T=seq_len.
        max_lin_conv = int(seq_len + kernel_len - 1)
        max_n_fft = 1
        while max_n_fft < max_lin_conv:
            max_n_fft *= 2
        self.max_freq_bins = max_n_fft // 2 + 1

        # Per-frequency gate logits, initialized "mostly open".
        self.gate_freq_logits = nn.Parameter(torch.ones(self.max_freq_bins) * 2.0)  # sigmoid ~0.88

        # Context-conditioned gate (per channel), initialized to "mostly open".
        self.gate_ctx = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gate_ctx.weight)
        nn.init.constant_(self.gate_ctx.bias, 2.0)  # sigmoid ~0.88

        # Pointwise feedforward (adds capacity for spelling/vocab)
        hidden = d_model * 2
        self.ffn_ln = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        # init small so residual path dominates early
        for m in self.ffn:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, cutoff: int | None = None) -> torch.Tensor:
        """x: [B, T, C] real.

        cutoff: number of rfft bins to keep (progressive training). If provided,
                bins >= cutoff are zeroed.
        """
        residual = x
        x = self.ln(x)

        B, T, C = x.shape
        # Use linear convolution via zero-padding to avoid circular wrap (future leakage).
        n_fft = 1
        while n_fft < (T + self.kernel_len - 1):
            n_fft *= 2

        # Build padded causal kernel (real)
        k = torch.zeros(n_fft, device=x.device, dtype=x.dtype)
        k[: self.kernel_len] = self.kernel
        k_freq = torch.fft.rfft(k)  # [F]

        # FFT input along time
        x_pad = F.pad(x, (0, 0, 0, n_fft - T))  # [B, n_fft, C]
        x_freq = torch.fft.rfft(x_pad, dim=1)  # [B, F, C]

        # Apply frequency response (broadcast), plus per-channel gain
        y_freq = x_freq * k_freq.unsqueeze(0).unsqueeze(-1) * self.gain.unsqueeze(0).unsqueeze(0)

        # ----------------
        # Gating (Valve)
        # ----------------
        Fbins = y_freq.size(1)

        # Per-frequency gate
        g_freq = torch.sigmoid(self.gate_freq_logits[:Fbins]).to(dtype=y_freq.real.dtype)  # [F]

        # Context gate (per channel): use pooled hidden state
        pooled = x.mean(dim=1)  # [B, C]
        g_ctx = torch.sigmoid(self.gate_ctx(pooled)).to(dtype=y_freq.real.dtype)  # [B, C]

        # Apply both
        y_freq = y_freq * g_freq.unsqueeze(0).unsqueeze(-1) * g_ctx.unsqueeze(1)

        # Progressive frequency horizon (JPEG schedule): soft roll-off to reduce Gibbs ringing
        if cutoff is not None:
            cutoff_idx = min(int(cutoff), Fbins)
            if cutoff_idx < Fbins:
                trans = min(self.transition_bins, cutoff_idx)  # can't exceed cutoff
                # 1 up to (cutoff_idx-trans), cosine down to 0 at cutoff_idx, 0 beyond.
                mask = torch.ones(Fbins, device=y_freq.device, dtype=y_freq.real.dtype)
                start = cutoff_idx - trans
                if trans > 0:
                    t = torch.linspace(0, 1, steps=trans, device=y_freq.device, dtype=mask.dtype)
                    mask[start:cutoff_idx] = 0.5 * (1.0 + torch.cos(torch.pi * t))
                mask[cutoff_idx:] = 0.0
                y_freq = y_freq * mask.unsqueeze(0).unsqueeze(-1)

        y_pad = torch.fft.irfft(y_freq, n=n_fft, dim=1)  # [B, n_fft, C]
        # Linear conv output length: T + K - 1; we take first T (causal)
        y = y_pad[:, :T, :]

        y = self.drop(y)
        x = residual + y

        # FFN residual
        ff_in = self.ffn_ln(x)
        x = x + self.drop(self.ffn(ff_in))
        return x


class FixedSpectralLM(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [
                FixedSpectralBlock(
                    cfg.d_model,
                    seq_len=cfg.seq_len,
                    kernel_len=cfg.kernel_len,
                    transition_bins=cfg.jpeg_transition,
                    dropout=0.1,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # Weight-tying via matmul with embedding weight

    def forward(self, x: torch.Tensor, cutoff: int | None = None) -> torch.Tensor:
        """x: [B, T] long -> logits [B, T, V]."""
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h, cutoff=cutoff)
        h = self.ln_f(h)
        logits = torch.matmul(h, self.embed.weight.t())
        return logits


@torch.no_grad()
def generate(
    model: FixedSpectralLM,
    prompt: str,
    cfg: TrainConfig,
    device: str,
    *,
    cutoff: int | None = None,
) -> str:
    model.eval()

    # bytes for prompt
    ctx = [b for b in prompt.encode("utf-8", errors="ignore")]
    if len(ctx) == 0:
        ctx = [32]

    def apply_top_p(logits_1d: torch.Tensor, p: float) -> torch.Tensor:
        # returns masked logits
        sorted_logits, sorted_idx = torch.sort(logits_1d, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cdf = torch.cumsum(probs, dim=-1)
        # keep at least 1 token
        keep = cdf <= p
        keep[0] = True
        cutoff_pos = keep.sum().item()
        masked = logits_1d.clone()
        masked[:] = -float("inf")
        masked[sorted_idx[:cutoff_pos]] = logits_1d[sorted_idx[:cutoff_pos]]
        return masked

    for _ in range(cfg.max_new):
        x = torch.tensor([ctx[-cfg.seq_len :]], dtype=torch.long, device=device)
        logits = model(x, cutoff=cutoff)
        next_logits = logits[0, -1].float()

        # repetition penalty (stronger, longer window)
        recent = ctx[-cfg.repetition_window :]
        for tok in set(recent):
            next_logits[tok] = next_logits[tok] / cfg.repetition_penalty

        # presence/frequency penalties (OpenAI-style) to break attractors
        if cfg.presence_penalty or cfg.frequency_penalty:
            # count occurrences in recent window
            counts = {}
            for t in recent:
                counts[t] = counts.get(t, 0) + 1
            for tok, c in counts.items():
                next_logits[tok] = next_logits[tok] - cfg.presence_penalty - cfg.frequency_penalty * float(c)

        # optionally ban control chars (esp. '\r')
        if cfg.ascii_only:
            # allow newline and printable ASCII; ban everything else
            banned = torch.ones_like(next_logits, dtype=torch.bool)
            banned[10] = False
            banned[32:127] = False
            next_logits[banned] = -float("inf")
        if cfg.ban_cr:
            next_logits[13] = -float("inf")

        # hard anti-stutter: if last N bytes are identical, ban that byte for 1 step
        if len(ctx) >= cfg.max_run_length:
            run_byte = ctx[-1]
            if all(b == run_byte for b in ctx[-cfg.max_run_length :]):
                next_logits[run_byte] = -float("inf")

        # temperature
        next_logits = next_logits / cfg.temperature

        # nucleus (top-p) sampling
        if cfg.top_p is not None and cfg.top_p < 1.0:
            next_logits = apply_top_p(next_logits, cfg.top_p)

        # optional top-k backstop
        if cfg.top_k and cfg.top_k > 0:
            k = min(cfg.top_k, next_logits.numel())
            v, _ = torch.topk(next_logits, k)
            next_logits[next_logits < v[-1]] = -float("inf")

        probs = F.softmax(next_logits, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        ctx.append(int(nxt))

    # decode bytes (may include non-ascii). Return unicode with replacement.
    return bytes(ctx).decode("utf-8", errors="replace")


def safe_console(s: str) -> str:
    """Make a string safe to print on Windows cp1252 consoles."""
    # Convert unprintable/unencodable chars into backslash escapes.
    return s.encode("unicode_escape", errors="backslashreplace").decode("ascii", errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--kernel-len", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--log-every-steps", type=int, default=None)
    parser.add_argument("--no-sawtooth", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.steps_per_epoch is not None:
        cfg.steps_per_epoch = args.steps_per_epoch
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len
    if args.kernel_len is not None:
        cfg.kernel_len = args.kernel_len
    if args.lr is not None:
        cfg.lr = args.lr
    if args.top_p is not None:
        cfg.top_p = args.top_p
    if args.top_k is not None:
        cfg.top_k = args.top_k
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.repetition_penalty is not None:
        cfg.repetition_penalty = args.repetition_penalty
    if args.log_every_steps is not None:
        cfg.log_every_steps = args.log_every_steps
    set_seed(cfg.seed)

    if cfg.device == "cpu":
        raise SystemExit("CUDA required for this run")

    if not os.path.exists(cfg.data_path):
        raise SystemExit(f"Missing dataset file: {cfg.data_path}")

    print("=" * 70)
    print("TRAIN FIXED SPECTRAL MIXER (FULL DATA, NON-GREEDY SAMPLING)")
    print("=" * 70)
    print(f"Device: {cfg.device}")
    print(f"Data:   {cfg.data_path}")
    print(f"SeqLen: {cfg.seq_len}")
    print(f"Batch:  {cfg.batch_size}")
    print(f"Epochs: {cfg.epochs} (steps/epoch={cfg.steps_per_epoch})")
    print(f"LR:     {cfg.lr} (wd={cfg.weight_decay})")
    if not args.no_sawtooth:
        print(
            f"LR sched: sawtooth cosine restarts (stage-aligned)"
            f"  s1(e0-{cfg.stage1_epochs-1}) mult {cfg.stage1_lr_mult}->{cfg.stage1_min_mult}"
            f"  s2(e{cfg.stage1_epochs}-{cfg.stage1_epochs+cfg.stage2_epochs-1}) mult {cfg.stage2_lr_mult}->{cfg.stage2_min_mult}"
            f"  s3(e{cfg.stage1_epochs+cfg.stage2_epochs}+) mult {cfg.stage3_lr_mult}->{cfg.stage3_min_mult}"
        )
    print(f"Gen:    temp={cfg.temperature} top_p={cfg.top_p} top_k={cfg.top_k} rep={cfg.repetition_penalty} win={cfg.repetition_window} maxrun={cfg.max_run_length}")
    print(f"        presence={cfg.presence_penalty} freq={cfg.frequency_penalty} ascii_only={cfg.ascii_only} ban_cr={cfg.ban_cr}")

    corpus_u8 = load_corpus_as_u8(cfg.data_path, sanitize_ascii=cfg.ascii_only)
    n = int(corpus_u8.numel())
    print(f"Corpus bytes: {n:,}")

    # precompute fixed validation windows + corpus blob for parroting test
    val_starts = make_val_starts(n, cfg.seq_len, cfg.val_windows, cfg.seed + 1)
    corpus_blob = bytes(corpus_u8.numpy().tobytes())

    model = FixedSpectralLM(cfg).to(cfg.device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,} (~{params/1e6:.2f}M)")
    print("=" * 70)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    start_epoch = 0
    if args.resume and os.path.exists(cfg.ckpt_path):
        ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        opt.load_state_dict(ckpt["opt"])
        if "scaler" in ckpt and cfg.amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"Resumed from {cfg.ckpt_path} at epoch {start_epoch}")

    freq_bins = cfg.seq_len // 2 + 1
    t0 = time.time()

    def save_ckpt(epoch_idx: int) -> None:
        torch.save(
            {
                "epoch": epoch_idx,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if cfg.amp else None,
                "cfg": cfg.__dict__,
            },
            cfg.ckpt_path,
        )

    last_epoch = start_epoch
    try:
        for epoch in range(start_epoch, cfg.epochs):
            last_epoch = epoch + 1
            model.train()
            # Spectral Curriculum Learning: cutoff is locked per stage.
            cutoff = curriculum_cutoff(epoch, cfg, freq_bins)

            losses = []
            running = 0.0
            running_lr = 0.0
            for _step in range(cfg.steps_per_epoch):
                global_step = epoch * cfg.steps_per_epoch + _step
                if not args.no_sawtooth:
                    lr_now = sawtooth_lr(global_step, epoch, cfg)
                    for pg in opt.param_groups:
                        pg["lr"] = lr_now
                else:
                    lr_now = opt.param_groups[0]["lr"]
                # random windows
                starts = torch.randint(0, n - (cfg.seq_len + 1) - 1, (cfg.batch_size,))
                batch_x = torch.stack([corpus_u8[s : s + cfg.seq_len] for s in starts]).to(torch.long)
                batch_y = torch.stack([corpus_u8[s + 1 : s + cfg.seq_len + 1] for s in starts]).to(torch.long)
                batch_x = batch_x.to(cfg.device, non_blocking=True)
                batch_y = batch_y.to(cfg.device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=cfg.amp):
                    logits = model(batch_x, cutoff=cutoff)
                    loss = loss_fn(logits.reshape(-1, cfg.vocab_size), batch_y.reshape(-1))

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()

                li = float(loss.item())
                losses.append(li)
                running += li
                running_lr += float(lr_now)

                if cfg.log_every_steps and (_step + 1) % cfg.log_every_steps == 0:
                    avg_step = running / cfg.log_every_steps
                    avg_lr = running_lr / cfg.log_every_steps
                    running = 0.0
                    running_lr = 0.0
                    print(
                        f"  step {_step+1:5d}/{cfg.steps_per_epoch}  avg_loss={avg_step:.4f}  lr={avg_lr:.6g}  cutoff={cutoff}/{freq_bins}",
                        flush=True,
                    )

            avg = sum(losses) / len(losses)
            elapsed = time.time() - t0
            vloss = eval_loss(model, corpus_u8, val_starts, cfg, cutoff=cutoff)
            gap = avg - vloss
            print(
                f"Epoch {epoch+1:3d}/{cfg.epochs}  train={avg:.4f}  val={vloss:.4f}  gap={gap:+.4f}  cutoff={cutoff}/{freq_bins}  elapsed={elapsed/60:.1f}m"
            )

            if (epoch + 1) % 25 == 0:
                print("-" * 70)
                # Use stage cutoff for stable samples.
                sample = generate(model, "Once upon a time", cfg, cfg.device, cutoff=cutoff)
                print(safe_console(sample))
                # parroting test
                score = parroting_score(corpus_blob, sample.encode("utf-8", errors="ignore"), cfg)
                print(f"[parroting_score] {score:.2f} (0=novel, 1=copied)")
                print("-" * 70)

            if (epoch + 1) % cfg.save_every_epochs == 0:
                save_ckpt(epoch + 1)

    finally:
        # always save the latest checkpoint
        if last_epoch > 0:
            save_ckpt(last_epoch)

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
