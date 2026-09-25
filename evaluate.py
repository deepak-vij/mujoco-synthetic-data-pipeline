"""Closed-loop evaluation: run the learned image policy back in the simulator.

Every policy faces the same held-out randomized worlds (same seeds), so the
comparison with the privileged expert and a do-nothing baseline is apples to apples.

Usage:
  python evaluate.py                                  # 20 episodes, writes eval_policy.mp4
  python evaluate.py --episodes 50 --seed 2000
  python evaluate.py --checkpoint checkpoints/policy_200.pt checkpoints/policy_rl.pt   # compare models
"""
import argparse
from collections import deque
from pathlib import Path

import imageio
import mujoco
import numpy as np
import torch

from datagen import IMG_SIZE, build_model, expert_policy, randomize, rollout
from train import Policy


class ImagePolicy:
    """Wraps the CNN: keeps a rolling stack of the last N frames, ignores the true state."""

    def __init__(self, net, n_frames, ctrl_range):
        self.net, self.buf, self.lo, self.hi = net, deque(maxlen=n_frames), *ctrl_range

    def __call__(self, image, state):
        frame = torch.from_numpy(image).permute(2, 0, 1)
        if not self.buf:
            self.buf.extend([frame] * self.buf.maxlen)
        self.buf.append(frame)
        x = torch.cat(list(self.buf), 0)[None]
        return float(np.clip(self.net.act(x).item(), self.lo, self.hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", default=["checkpoints/policy.pt"],
                    help="one or more models; the video shows the last one")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000, help="held out: datagen uses seed 0")
    ap.add_argument("--video", default="eval_policy.mp4")
    ap.add_argument("--video-episodes", type=int, default=4)
    args = ap.parse_args()

    model, nominal = build_model()
    ctrl_range = tuple(model.actuator_ctrlrange[0])
    policies = {"expert (true state)": lambda: expert_policy(model)}
    for path in args.checkpoint:
        ckpt = torch.load(path)
        net = Policy(ckpt["n_frames"], ckpt["img_size"])
        net.load_state_dict(ckpt["model"])
        net.eval()
        name = f"learned: {Path(path).stem}"
        policies[name] = lambda net=net, n=ckpt["n_frames"]: ImagePolicy(net, n, ctrl_range)
    policies["do nothing"] = lambda: (lambda image, state: 0.0)
    video_policy = f"learned: {Path(args.checkpoint[-1]).stem}"

    results, clips = {}, []
    with mujoco.Renderer(model, IMG_SIZE, IMG_SIZE) as renderer:
        for name, make in policies.items():
            results[name] = []
            for ep in range(args.episodes):
                rng = np.random.default_rng(args.seed + ep)
                randomize(model, nominal, rng)
                frames = []
                record = name == video_policy and ep < args.video_episodes
                info = rollout(model, renderer, rng, make(),
                               on_frame=(lambda img, s, a: frames.append(img)) if record else None)
                results[name].append(info)
                if record:
                    clips.append(frames)

    print(f"\n{args.episodes} held-out episodes (seeds {args.seed}..{args.seed + args.episodes - 1})")
    print(f"{'policy':28s} {'success':>8s} {'mean max angle':>15s}")
    for name, res in results.items():
        ok = np.mean([r["success"] for r in res])
        ang = np.degrees(np.mean([r["max_angle_after_1s"] for r in res]))
        print(f"{name:28s} {ok:8.0%} {ang:13.1f}°")
    for name, res in results.items():
        fails = [i for i, r in enumerate(res) if not r["success"]]
        if name.startswith("learned") and fails:
            print(f"{name} failed on episodes: {fails}")

    if clips:
        # 2x2 grid (or row) of the first recorded episodes, upscaled 3x; short episodes freeze on last frame.
        n = max(len(c) for c in clips)
        clips = [c + [c[-1]] * (n - len(c)) for c in clips]
        cols = 2 if len(clips) == 4 else len(clips)
        grid = []
        for t in range(n):
            tiles = [np.repeat(np.repeat(c[t], 3, 0), 3, 1) for c in clips]
            rows = [np.concatenate(tiles[r:r + cols], 1) for r in range(0, len(tiles), cols)]
            grid.append(np.concatenate(rows, 0))
        imageio.mimsave(args.video, grid, fps=50)
        print(f"Saved {args.video}")


if __name__ == "__main__":
    main()
