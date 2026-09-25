"""Generate a synthetic cart-pole dataset in LeRobot format.

Each episode randomizes physics, visuals, camera and initial state, then records
what a real robot could observe (camera image + joint state) together with the
action chosen by a privileged LQR expert that knows the true physics.

Usage:
  python datagen.py --episodes 50                  # writes ./data/cartpole_sim
  python datagen.py --episodes 50 --overwrite      # regenerate from scratch
  python datagen.py --episodes 50 --push-repo <hf_user>/cartpole_sim
"""
import argparse
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from cartpole import XML, lqr_gain

FPS = 50                 # recording rate = control rate
EPISODE_SECONDS = 5.0
IMG_SIZE = 128
TASK = "Balance the pole upright on the cart."
FALL_ANGLE = 0.8         # rad; beyond this the episode is a failure

FEATURES = {
    "observation.image": {
        "dtype": "video",
        "shape": (IMG_SIZE, IMG_SIZE, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (4,),
        "names": ["cart_pos", "pole_angle", "cart_vel", "pole_vel"],
    },
    "action": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["cart_force"],
    },
}


def build_model():
    """Cart-pole model with named geoms, plus nominal inertial params for randomization."""
    xml = XML.replace('rgba=".9 .5 .1 1" mass="1"', 'name="cart_geom" rgba=".9 .5 .1 1" mass="1"')
    xml = xml.replace('rgba=".2 .7 .9 1" mass="0.2"', 'name="pole_geom" rgba=".2 .7 .9 1" mass="0.2"')
    model = mujoco.MjModel.from_xml_string(xml)
    nominal = {"body_mass": model.body_mass.copy(), "body_inertia": model.body_inertia.copy()}
    return model, nominal


def randomize(model, nominal, rng):
    """Domain randomization: perturb physics and visuals in place. Returns the params used."""
    cart = model.body("cart").id
    pole = model.body("pole").id
    p = {
        "cart_mass_scale": rng.uniform(0.7, 1.5),
        "pole_mass_scale": rng.uniform(0.5, 2.0),
        "cart_damping": rng.uniform(0.0, 0.5),
        "pole_damping": rng.uniform(0.0, 0.05),
        "motor_gain": rng.uniform(0.8, 1.2),
    }
    for body, key in [(cart, "cart_mass_scale"), (pole, "pole_mass_scale")]:
        model.body_mass[body] = nominal["body_mass"][body] * p[key]
        model.body_inertia[body] = nominal["body_inertia"][body] * p[key]
    model.dof_damping[:] = [p["cart_damping"], p["pole_damping"]]
    model.actuator_gear[0, 0] = p["motor_gain"]

    # Visuals: colors of cart, pole, floor, and lighting.
    model.geom_rgba[model.geom("cart_geom").id, :3] = rng.uniform(0.2, 1.0, 3)
    model.geom_rgba[model.geom("pole_geom").id, :3] = rng.uniform(0.2, 1.0, 3)
    model.mat_rgba[model.material("grid").id, :3] = rng.uniform(0.4, 1.0, 3)
    model.light_diffuse[0] = rng.uniform(0.5, 1.0, 3)
    model.vis.headlight.ambient[:] = rng.uniform(0.2, 0.5, 3)
    return {k: float(v) for k, v in p.items()}


def random_camera(rng):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [rng.uniform(-0.1, 0.1), 0, rng.uniform(0.15, 0.25)]
    cam.distance = rng.uniform(2.0, 2.6)
    cam.elevation = rng.uniform(-20, -5)
    cam.azimuth = 90 + rng.uniform(-15, 15)
    return cam


def expert_policy(model):
    """Privileged LQR expert: designed with the (randomized) true physics of this episode."""
    substeps = round(1.0 / (FPS * model.opt.timestep))
    K = lqr_gain(model, substeps)
    lo, hi = model.actuator_ctrlrange[0]
    return lambda image, state: float(np.clip(-K @ state, lo, hi)[0])


def rollout(model, renderer, rng, policy, action_noise=0.0, on_frame=None):
    """Run one episode at FPS. `policy(image, state) -> force`; `on_frame` sees each step."""
    substeps = round(1.0 / (FPS * model.opt.timestep))
    cam = random_camera(rng)

    data = mujoco.MjData(model)
    data.qpos[:] = [rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3)]
    data.qvel[:] = rng.normal(0, 0.1, 2)
    mujoco.mj_forward(model, data)

    # One random shove on the pole tip mid-episode.
    push_t = rng.uniform(1.5, 3.5)
    push_f = rng.choice([-1, 1]) * rng.uniform(1.0, 3.0)

    lo, hi = model.actuator_ctrlrange[0]
    max_angle = 0.0
    for _ in range(int(EPISODE_SECONDS * FPS)):
        state = np.concatenate([data.qpos, data.qvel]).astype(np.float32)
        renderer.update_scene(data, cam)
        image = renderer.render()

        action = policy(image, state)
        if on_frame:
            on_frame(image, state, action)

        # Execute a noisy version of the action (DART-style) so the data also
        # covers states slightly off the expert's path, while labels stay clean.
        data.ctrl[0] = np.clip(action + rng.normal(0, action_noise), lo, hi)
        for _ in range(substeps):
            data.xfrc_applied[model.body("pole").id, 0] = (
                push_f if push_t <= data.time < push_t + 0.05 else 0.0)
            mujoco.mj_step(model, data)
        max_angle = max(max_angle, abs(data.qpos[1]) if data.time > 1.0 else 0.0)
        if abs(data.qpos[1]) > FALL_ANGLE:
            max_angle = max(max_angle, FALL_ANGLE)
            break

    return {
        "success": bool(max_angle < 0.3 and abs(data.qpos[0]) < 1.7),
        "max_angle_after_1s": float(max_angle),
        "push_time": float(push_t),
        "push_force": float(push_f),
        "camera": {"distance": cam.distance, "elevation": cam.elevation, "azimuth": cam.azimuth},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--root", default="data/cartpole_sim")
    ap.add_argument("--repo-id", default="local/cartpole_sim")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action-noise", type=float, default=1.0, help="std of executed-action noise (N)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--push-repo", help="HF repo id to upload to, e.g. <user>/cartpole_sim")
    args = ap.parse_args()

    root = Path(args.root)
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} exists; pass --overwrite to regenerate it.")
        shutil.rmtree(root)

    model, nominal = build_model()

    dataset = LeRobotDataset.create(
        repo_id=args.push_repo or args.repo_id, fps=FPS, features=FEATURES,
        root=root, robot_type="mujoco_cartpole", use_videos=True,
    )
    rng = np.random.default_rng(args.seed)
    meta = []
    with mujoco.Renderer(model, IMG_SIZE, IMG_SIZE) as renderer:
        for ep in range(args.episodes):
            params = randomize(model, nominal, rng)
            info = rollout(
                model, renderer, rng, expert_policy(model), args.action_noise,
                on_frame=lambda image, state, action: dataset.add_frame({
                    "observation.image": image,
                    "observation.state": state,
                    "action": np.array([action], dtype=np.float32),
                    "task": TASK,
                }))
            dataset.save_episode()
            meta.append({"episode_index": ep, **params, **info})
            print(f"episode {ep:3d}  success={info['success']}  "
                  f"pole_mass x{params['pole_mass_scale']:.2f}  max_angle={np.degrees(info['max_angle_after_1s']):5.1f}deg")
    dataset.finalize()

    # Ground-truth randomization params per episode, kept alongside the dataset.
    with open(root / "meta" / "randomization.jsonl", "w") as f:
        f.writelines(json.dumps(m) + "\n" for m in meta)
    n_ok = sum(m["success"] for m in meta)
    print(f"\n{n_ok}/{len(meta)} successful episodes, {dataset.num_frames} frames -> {root}")

    if args.push_repo:
        dataset.push_to_hub()
        print(f"Uploaded to https://huggingface.co/datasets/{args.push_repo}")


if __name__ == "__main__":
    main()
