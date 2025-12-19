"""generate_from_ckpt.py

Load the latest checkpoint produced by train_fixed_full.py and run text generation.

Examples:
  python generate_from_ckpt.py --prompt "Once upon a time" --max-new 400
  python generate_from_ckpt.py --prompt "There was a little girl" --top-p 0.9 --temperature 0.9
"""

from __future__ import annotations

import argparse
import os

import torch

import train_fixed_full as tff


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="fixed_spectral_ckpt.pt")
    ap.add_argument("--prompt", nargs="+", default=["Once", "upon", "a", "time"])
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--repetition-penalty", type=float, default=1.6)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--kernel-len", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=6)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("CUDA required")

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")

    cfg = tff.TrainConfig(
        seq_len=args.seq_len,
        kernel_len=args.kernel_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        max_new=args.max_new,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )

    # Build model and load
    model = tff.FixedSpectralLM(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    prompt = " ".join(args.prompt)
    # Use schedule cutoff if checkpoint epoch is present and < full horizon,
    # otherwise generate at full horizon.
    cutoff = None
    if "epoch" in ckpt and isinstance(ckpt["epoch"], int) and ckpt["epoch"] > 0:
        freq_bins = cfg.seq_len // 2 + 1
        cutoff = tff.jpeg_cutoff(ckpt["epoch"] - 1, cfg, freq_bins)
    out = tff.generate(model, prompt, cfg, device, cutoff=cutoff)
    print(tff.safe_console(out))


if __name__ == "__main__":
    main()
