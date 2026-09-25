"""RL fine-tuning (PPO) of the imitation-learned pixel policy, with MuJoCo as the practice world.

Closing the loop:
  imitation model (train.py) --> practice in randomized MuJoCo worlds --> reward --> better model

- Actor  = the pretrained CNN from train.py (sees camera images only). Its output becomes
           the mean of a Gaussian, so the policy can explore around what it learned.
- Critic = a small network that estimates "how good is this situation". It sees the TRUE
           state (a simulator privilege); it's only used during training, never on the robot.
- Reward = computed from the simulator's true state each step (see `reward`).

Usage:
  python rl_finetune.py                                   # from checkpoints/policy_200.pt
  python rl_finetune.py --iters 60 --init checkpoints/policy.pt
"""
import argparse
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datagen import EPISODE_SECONDS, FALL_ANGLE, FPS, IMG_SIZE, build_model, random_camera, randomize
from train import AUX_WEIGHT, Policy


def reward(state, force):
    """+1 for every step alive, minus penalties for tilt, drifting off-center, and effort."""
    x, theta = state[0], state[1]
    return float(1.0 - theta ** 2 - 0.1 * x ** 2 - 0.0005 * force ** 2)


class CartPoleEnv:
    """One randomized cart-pole world that returns stacked camera frames, like the dataset."""

    def __init__(self, seed, n_frames):
        self.model, self.nominal = build_model()
        self.rng = np.random.default_rng(seed)
        self.renderer = mujoco.Renderer(self.model, IMG_SIZE, IMG_SIZE)
        self.frames = deque(maxlen=n_frames)
        self.substeps = round(1.0 / (FPS * self.model.opt.timestep))
        self.max_steps = int(EPISODE_SECONDS * FPS)
        self.pole = self.model.body("pole").id
        self.lo, self.hi = self.model.actuator_ctrlrange[0]

    def reset(self):
        # Same randomization and start distribution as datagen.rollout.
        randomize(self.model, self.nominal, self.rng)
        self.cam = random_camera(self.rng)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = [self.rng.uniform(-0.5, 0.5), self.rng.uniform(-0.3, 0.3)]
        self.data.qvel[:] = self.rng.normal(0, 0.1, 2)
        mujoco.mj_forward(self.model, self.data)
        self.push_t = self.rng.uniform(1.5, 3.5)
        self.push_f = self.rng.choice([-1, 1]) * self.rng.uniform(1.0, 3.0)
        self.t, self.max_angle = 0, 0.0
        self.frames.clear()
        return self._obs()

    def _obs(self):
        self.renderer.update_scene(self.data, self.cam)
        frame = torch.from_numpy(self.renderer.render()).permute(2, 0, 1)
        if not self.frames:
            self.frames.extend([frame] * self.frames.maxlen)
        self.frames.append(frame)
        state = np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)
        return torch.cat(list(self.frames), 0), state

    def step(self, force):
        force = float(np.clip(force, self.lo, self.hi))
        self.data.ctrl[0] = force
        for _ in range(self.substeps):
            self.data.xfrc_applied[self.pole, 0] = (
                self.push_f if self.push_t <= self.data.time < self.push_t + 0.05 else 0.0)
            mujoco.mj_step(self.model, self.data)
        self.t += 1
        obs, state = self._obs()
        if self.data.time > 1.0:
            self.max_angle = max(self.max_angle, abs(state[1]))
        fell = abs(state[1]) > FALL_ANGLE or abs(state[0]) > 1.7
        timeout = self.t >= self.max_steps
        return obs, state, reward(state, force), fell, timeout


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, state):
        return self.net(state).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="checkpoints/policy_200.pt", help="imitation-learned starting point")
    ap.add_argument("--out", default="checkpoints/policy_rl.pt")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=128, help="steps per env per iteration")
    ap.add_argument("--critic-warmup", type=int, default=5, help="iterations training only the critic")
    ap.add_argument("--actor-lr", type=float, default=3e-5)
    ap.add_argument("--critic-lr", type=float, default=1e-3)
    ap.add_argument("--init-std", type=float, default=0.2, help="exploration noise, in normalized action units")
    ap.add_argument("--seed", type=int, default=5000, help="RL worlds; evaluate.py uses 1000+")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    gamma, lam, clip, epochs, minibatch = 0.99, 0.95, 0.2, 4, 256

    ckpt = torch.load(args.init)
    actor = Policy(ckpt["n_frames"], ckpt["img_size"]).to(device)
    actor.load_state_dict(ckpt["model"])
    log_std = nn.Parameter(torch.tensor(np.log(args.init_std), dtype=torch.float32, device=device))
    critic = Critic().to(device)
    actor_opt = torch.optim.Adam(list(actor.parameters()) + [log_std], lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    norm_s = lambda s: (s - actor.state_mean) / actor.state_std

    envs = [CartPoleEnv(args.seed + i, ckpt["n_frames"]) for i in range(args.envs)]
    obs, states = map(list, zip(*[e.reset() for e in envs]))
    T, N = args.steps, args.envs
    ep_returns, ep_success, ep_len = [0.0] * N, deque(maxlen=100), deque(maxlen=100)
    ret_acc = [0.0] * N
    t0 = time.time()

    for it in range(1, args.iters + 1):
        # ---- 1. Rollout: the current policy practices in the simulator ----
        buf_obs = torch.zeros(T, N, *obs[0].shape, dtype=torch.uint8)
        buf_state = torch.zeros(T, N, 4)
        buf_act, buf_logp, buf_rew, buf_done, buf_val = (torch.zeros(T, N) for _ in range(5))
        actor.eval()
        for t in range(T):
            o = torch.stack(obs)
            s = torch.tensor(np.stack(states))
            with torch.no_grad():
                mean, _ = actor(o.to(device))
                mean = mean.squeeze(-1)
                dist = torch.distributions.Normal(mean, log_std.exp())
                a = dist.sample()                               # normalized action, with exploration
                logp = dist.log_prob(a)
                v = critic(norm_s(s.to(device)))
            force = (a * actor.act_std + actor.act_mean).cpu().numpy()
            buf_obs[t], buf_state[t] = o, s
            buf_act[t], buf_logp[t], buf_val[t] = a.cpu(), logp.cpu(), v.cpu()
            for i, env in enumerate(envs):
                o_i, s_i, r, fell, timeout = env.step(force[i])
                ret_acc[i] += r
                if timeout and not fell:
                    # Episode cut by time limit, not failure: credit the value of where it ended up.
                    with torch.no_grad():
                        r += gamma * critic(norm_s(torch.tensor(s_i, device=device))).item()
                buf_rew[t, i] = r
                buf_done[t, i] = float(fell or timeout)
                if fell or timeout:
                    ep_success.append(float(not fell and env.max_angle < 0.3))
                    ep_len.append(env.t)
                    ep_returns[i], ret_acc[i] = ret_acc[i], 0.0
                    o_i, s_i = env.reset()
                obs[i], states[i] = o_i, s_i

        # ---- 2. Advantages (GAE): was each action better or worse than expected? ----
        with torch.no_grad():
            last_v = critic(norm_s(torch.tensor(np.stack(states), device=device))).cpu()
        adv = torch.zeros(T, N)
        gae = torch.zeros(N)
        for t in reversed(range(T)):
            next_v = last_v if t == T - 1 else buf_val[t + 1]
            nonterm = 1.0 - buf_done[t]
            delta = buf_rew[t] + gamma * next_v * nonterm - buf_val[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
        returns = adv + buf_val

        # ---- 3. PPO update: make better-than-expected actions more likely ----
        flat = lambda x: x.reshape(T * N, *x.shape[2:])
        b_obs, b_state, b_act, b_logp, b_adv, b_ret = map(flat, (buf_obs, buf_state, buf_act, buf_logp, adv, returns))
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        train_actor = it > args.critic_warmup
        actor.train()
        stats = []
        for _ in range(epochs):
            for idx in torch.randperm(T * N).split(minibatch):
                s = norm_s(b_state[idx].to(device))
                v_loss = F.mse_loss(critic(s), b_ret[idx].to(device))
                critic_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_opt.step()
                if not train_actor:
                    continue
                mean, pred_state = actor(b_obs[idx].to(device))
                dist = torch.distributions.Normal(mean.squeeze(-1), log_std.exp())
                ratio = (dist.log_prob(b_act[idx].to(device)) - b_logp[idx].to(device)).exp()
                A = b_adv[idx].to(device)
                pg_loss = -torch.min(ratio * A, ratio.clamp(1 - clip, 1 + clip) * A).mean()
                # Keep the free "where is the pole" supervision from imitation, so vision stays grounded.
                aux_loss = F.mse_loss(pred_state, s)
                loss = pg_loss + AUX_WEIGHT * aux_loss
                actor_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + [log_std], 0.5)
                actor_opt.step()
                stats.append((pg_loss.item(), aux_loss.item(), ((ratio - 1).abs() > clip).float().mean().item()))

        pg, aux, clipfrac = np.mean(stats, 0) if stats else (0, 0, 0)
        phase = "actor+critic" if train_actor else "critic warmup"
        succ = f"{np.mean(ep_success):.0%}" if ep_success else "  -"
        mlen = f"{np.mean(ep_len):5.0f}" if ep_len else "    -"
        print(f"iter {it:3d} [{phase:13s}]  success(last 100 eps) {succ:>4s}  ep len {mlen}  "
              f"value loss {v_loss.item():7.3f}  aux {aux:.3f}  clipfrac {clipfrac:.2f}  "
              f"std {log_std.exp().item():.3f}  {time.time() - t0:5.0f}s", flush=True)

        if it % 10 == 0 or it == args.iters:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": actor.state_dict(), "img_size": ckpt["img_size"], "n_frames": ckpt["n_frames"]}, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
