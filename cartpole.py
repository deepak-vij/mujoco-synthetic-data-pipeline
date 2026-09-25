"""MuJoCo PoC: balance an inverted pendulum on a cart with an LQR controller.

Usage:
  python cartpole.py            # headless: simulate and save cartpole.mp4
  python cartpole.py --viewer   # interactive 3D viewer (macOS: run with `mjpython`)
"""
import sys
import numpy as np
import mujoco
import scipy.linalg

XML = """
<mujoco model="cartpole">
  <option timestep="0.002"/>
  <visual><global offwidth="960" offheight="544"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .2 .3" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="4 4" reflectance=".2"/>
  </asset>
  <worldbody>
    <light pos="0 -2 4" dir="0 0.5 -1"/>
    <geom type="plane" size="4 2 .1" material="grid" pos="0 0 -0.3"/>
    <geom name="rail" type="capsule" fromto="-2 0 0 2 0 0" size="0.02" rgba=".6 .6 .6 1" contype="0" conaffinity="0"/>
    <body name="cart">
      <joint name="slider" type="slide" axis="1 0 0" range="-1.8 1.8" limited="true" damping="0.1"/>
      <geom type="box" size="0.15 0.1 0.06" rgba=".9 .5 .1 1" mass="1"/>
      <body name="pole">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.01"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.6" size="0.03" rgba=".2 .7 .9 1" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slider" gear="1" ctrllimited="true" ctrlrange="-20 20"/>
  </actuator>
</mujoco>
"""


def lqr_gain(model, substeps=1):
    """Linearize around the upright pose and solve the discrete-time LQR.

    `substeps` > 1 designs the controller for a slower loop that holds each
    action for that many physics steps (zero-order hold).
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    nv, nu = model.nv, model.nu
    A1 = np.zeros((2 * nv, 2 * nv))
    B1 = np.zeros((2 * nv, nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, A1, B1, None, None)
    A, B = np.eye(2 * nv), np.zeros_like(B1)
    for _ in range(substeps):
        A, B = A1 @ A, A1 @ B + B1
    Q = np.diag([1.0, 10.0, 1.0, 1.0])  # [cart pos, pole angle, cart vel, pole vel]
    R = np.eye(nu) * 0.1
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def make_controller(K):
    def control(model, data):
        state = np.concatenate([data.qpos, data.qvel])
        data.ctrl[:] = -K @ state
        # Periodic push so you can watch it recover.
        data.xfrc_applied[2, 0] = 1.5 if (data.time % 3.0) < 0.05 else 0.0
    return control


def run_headless(model, control, seconds=10.0, fps=30, out="cartpole.mp4"):
    import imageio
    data = mujoco.MjData(model)
    data.qpos[1] = 0.25  # start with the pole tilted ~14 degrees
    frames = []
    with mujoco.Renderer(model, 544, 960) as renderer:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, 0, 0.2]
        cam.distance, cam.elevation, cam.azimuth = 3.5, -15, 90
        while data.time < seconds:
            control(model, data)
            mujoco.mj_step(model, data)
            if len(frames) < data.time * fps:
                renderer.update_scene(data, cam)
                frames.append(renderer.render())
    imageio.mimsave(out, frames, fps=fps)
    print(f"Final pole angle: {np.degrees(data.qpos[1]):.2f} deg, cart x: {data.qpos[0]:.3f} m")
    print(f"Saved {len(frames)} frames to {out}")


def run_viewer(model, control):
    import time
    import mujoco.viewer
    data = mujoco.MjData(model)
    data.qpos[1] = 0.25
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            t0 = time.time()
            control(model, data)
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(max(0, model.opt.timestep - (time.time() - t0)))


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_string(XML)
    K = lqr_gain(model)
    print("LQR gain:", np.round(K, 2))
    control = make_controller(K)
    if "--viewer" in sys.argv:
        run_viewer(model, control)
    else:
        run_headless(model, control)
