"""Behavior cloning: learn to balance the cart-pole from camera images only.

The policy sees the last N_FRAMES images (a single frame can't show velocity) and
predicts the expert's force. As an auxiliary target it also predicts the true
state - a label the simulator gives us for free - which helps the CNN learn
where the pole is.

Usage:
  python train.py                       # trains on data/cartpole_sim -> checkpoints/policy.pt
  python train.py --epochs 40 --val-episodes 5
"""
import argparse
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

N_FRAMES = 3
AUX_WEIGHT = 0.5


class Policy(nn.Module):
    def __init__(self, n_frames=N_FRAMES, img_size=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3 * n_frames, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        )
        feat = 128 * (img_size // 16) ** 2
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(feat, 256), nn.ReLU(), nn.Linear(256, 5))
        # Normalization stats, saved with the weights.
        for name, size in [("act_mean", 1), ("act_std", 1), ("state_mean", 4), ("state_std", 4)]:
            self.register_buffer(name, torch.zeros(size) if "mean" in name else torch.ones(size))

    def forward(self, frames):
        """frames: uint8 (B, N_FRAMES*3, H, W). Returns normalized (action, state) predictions."""
        out = self.head(self.encoder(frames.float() / 255.0 - 0.5))
        return out[:, :1], out[:, 1:]

    @torch.no_grad()
    def act(self, frames):
        a, _ = self(frames)
        return a * self.act_std + self.act_mean


def load_data(root):
    root = Path(root)
    df = pd.concat(pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet")))
    df = df.sort_values("index").reset_index(drop=True)
    frames = []
    for path in sorted((root / "videos" / "observation.image").rglob("*.mp4")):
        with av.open(str(path)) as c:
            frames += [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    frames = np.stack(frames)
    assert len(frames) == len(df), f"{len(frames)} video frames vs {len(df)} rows"
    return (
        torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous(),        # (N, 3, H, W) uint8
        torch.tensor(np.stack(df["observation.state"]), dtype=torch.float32),
        torch.tensor(np.stack(df["action"]), dtype=torch.float32).reshape(-1, 1),
        df["episode_index"].to_numpy(),
        df["frame_index"].to_numpy(),
    )


def stack_indices(frame_index):
    """For each frame i: [i-(N-1), ..., i-1, i], clamped to the episode start."""
    fi = torch.as_tensor(frame_index)
    i = torch.arange(len(fi))
    return torch.stack([i - torch.minimum(fi, torch.tensor(k)) for k in reversed(range(N_FRAMES))], 1)


def random_shift(x, pad=4):
    """DrQ-style augmentation: translate each image stack by up to `pad` pixels."""
    b, _, h, w = x.shape
    shift = (torch.randint(-pad, pad + 1, (b, 2), device=x.device).float() * 2 / torch.tensor([w, h], device=x.device))
    theta = torch.zeros(b, 2, 3, device=x.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1
    theta[:, :, 2] = shift
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x.float(), grid, padding_mode="border", align_corners=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cartpole_sim")
    ap.add_argument("--out", default="checkpoints/policy.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    frames, states, actions, ep, fi = load_data(args.data)
    stacks = stack_indices(fi)
    val_eps = np.unique(ep)[-args.val_episodes:]
    is_val = torch.from_numpy(np.isin(ep, val_eps))
    train_idx, val_idx = torch.where(~is_val)[0], torch.where(is_val)[0]
    print(f"Loaded {len(frames)} frames in {time.time() - t0:.1f}s | "
          f"train {len(train_idx)} / val {len(val_idx)} frames (val episodes {val_eps.tolist()}) | device {device}")

    model = Policy(img_size=frames.shape[-1]).to(device)
    model.act_mean[:] = actions[train_idx].mean(0)
    model.act_std[:] = actions[train_idx].std(0)
    model.state_mean[:] = states[train_idx].mean(0)
    model.state_std[:] = states[train_idx].std(0)
    norm_a = lambda a: (a.to(device) - model.act_mean) / model.act_std
    norm_s = lambda s: (s.to(device) - model.state_mean) / model.state_std

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(train_idx) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)

    def batch(idx):
        x = frames[stacks[idx]].flatten(1, 2).to(device)   # (B, N_FRAMES*3, H, W)
        return x, norm_a(actions[idx]), norm_s(states[idx])

    for epoch in range(args.epochs):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx))]
        tl, nb = 0.0, len(perm) // args.batch_size
        for b in range(nb):
            x, a, s = batch(perm[b * args.batch_size:(b + 1) * args.batch_size])
            pa, ps = model(random_shift(x))
            loss = F.mse_loss(pa, a) + AUX_WEIGHT * F.mse_loss(ps, s)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item()

        model.eval()
        with torch.no_grad():
            va, vs = [], []
            for b in range(0, len(val_idx), 512):
                x, a, s = batch(val_idx[b:b + 512])
                pa, ps = model(x)
                va.append(((pa - a) ** 2).sum().item())
                vs.append(((ps - s) ** 2).mean(1).sum().item())
        # Action loss is on normalized targets, so 1.0 = predicting the mean, 0.0 = perfect.
        print(f"epoch {epoch + 1:2d}  train {tl / nb:.4f}  "
              f"val action {sum(va) / len(val_idx):.4f}  val state {sum(vs) / len(val_idx):.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "img_size": frames.shape[-1], "n_frames": N_FRAMES}, args.out)
    print(f"Saved {args.out} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
