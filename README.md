# MuJoCo synthetic data pipeline for embodied AI

A small, end-to-end example of how robots can learn from simulated data:
simulate → generate data → learn by imitation → test → improve by practice.
Everything runs on commodity hardware; no GPU cluster required. Training uses a GPU automatically when one is available (CUDA or Apple MPS) and falls back to CPU.

| Step | File | What it does |
|---|---|---|
| 1. Simulate | `cartpole.py` | MuJoCo cart-pole simulation, plus an expert controller that balances the pole perfectly |
| 2. Generate data | `datagen.py` | Records the expert in many randomly varied worlds, saved as a LeRobot dataset |
| 3. Imitate | `train.py` | Trains a model to copy the expert using only camera images (behavior cloning) |
| 4. Test | `evaluate.py` | Tests models in new worlds they never trained on, compared with the expert and with doing nothing |
| 5. Practice | `rl_finetune.py` | Improves the model by trial and error in 16 simulated worlds at once (reinforcement learning with PPO) |

## Results (50 new test worlds, never used in training)

Each test world has its own random weights, friction, colors and camera angle.

| Approach | Success | Average worst pole tilt |
|---|---|---|
| Expert (reads exact physics) | 100% | 4.8° |
| Imitation only (camera images) | 74% | 16.2° |
| Imitation + practice (RL) | 84% | 12.2° |
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
