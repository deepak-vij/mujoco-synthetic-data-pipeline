# MuJoCo synthetic data pipeline for embodied AI

A minimal, end-to-end version of the loop used to train robot policies:
simulate → generate synthetic data → imitation learning → evaluate → RL fine-tuning.
Everything runs on commodity hardware; no GPU cluster required. Training uses a GPU automatically when one is available (CUDA or Apple MPS) and falls back to CPU.

| Step | File | What it does |
|---|---|---|
| 1. Simulate | `cartpole.py` | MuJoCo cart-pole + LQR expert controller |
| 2. Generate data | `datagen.py` | Domain-randomized episodes saved as a LeRobot dataset |
| 3. Imitate | `train.py` | CNN policy from 3 stacked camera frames (behavior cloning) |
| 4. Evaluate | `evaluate.py` | Closed-loop tests on held-out worlds vs expert and do-nothing |
| 5. Refine | `rl_finetune.py` | PPO fine-tuning in 16 parallel randomized worlds |

## Results (50 held-out worlds)

| Policy | Success | Mean max tilt |
|---|---|---|
| Expert (true state) | 100% | 4.8° |
| Imitation, pixels only | 74% | 16.2° |
| Imitation + RL | 84% | 12.2° |
| Do nothing | 0% | 46.0° |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python cartpole.py                                   # expert demo -> cartpole.mp4
python cartpole.py --viewer                          # live 3D viewer (use mjpython on macOS)
python datagen.py --episodes 200 --root data/cartpole_sim_200
python train.py --data data/cartpole_sim_200 --val-episodes 10 --out checkpoints/policy_200.pt
python rl_finetune.py                                # starts from checkpoints/policy_200.pt
python evaluate.py --episodes 50 --checkpoint checkpoints/policy_200.pt checkpoints/policy_rl.pt
```

Approximate run times on a single machine: data 2 min, imitation training ~20 min, RL ~7 min (varies by hardware).

Optional: upload the dataset to the Hugging Face Hub with
`python datagen.py ... --push-repo <your-hf-username>/cartpole_sim` (after `hf auth login`).

## Blog post

See `blog/` for images used in the accompanying write-up.
