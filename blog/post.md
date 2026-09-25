# Scaling Robot Data with Simulation: Breaking Embodied AI's Data Bottleneck

Sep 23, 2026 · Deepak Vij

*Language models learned from the internet. Robots have no internet to learn from. Simulation may be how we build one.*

The biggest constraint on embodied AI isn't model architecture or compute. It's data. Large language models were trained on trillions of words that humanity had already written down. Nobody has been writing down, at scale, how to grasp a cup, fold a towel or keep a pole balanced on a moving cart.

My argument in this post is simple: **simulation changes robot data from something you collect into something you generate programmatically.** Once data is generated, its volume, diversity and labeling become engineering choices rather than hard limits.

To see whether that holds up in practice, I built the full loop myself: simulate a robot task, generate synthetic demonstrations, train a policy by imitation, then let it improve through reinforcement learning in the simulator.

## The embodied data gap

Robot learning has a data gap measured in orders of magnitude, not percentages.

Consider the scale on each side. Meta's Llama 3 was [pretrained on over 15 trillion tokens](https://ai.meta.com/blog/meta-llama-3/) of publicly available text. On the robotics side, [DROID](https://arxiv.org/abs/2403.12945), one of the largest real-world manipulation datasets, holds 76,000 demonstrations, or 350 hours of interaction. It took 50 data collectors across three continents 12 months to gather. [Open X-Embodiment](https://arxiv.org/abs/2310.08864), a landmark effort to pool robot data, needed a collaboration of 21 institutions across 22 different robots.

These are impressive efforts. They are also a reminder that robot data doesn't accumulate on its own the way text did. Five properties make it scarce:

- **It's physical.** Every example needs a real robot, a real scene and real time. You can't scrape it.
- **It's slow and expensive.** Most demonstrations come from people teleoperating robots, one episode at a time.
- **It's tied to a body.** Data from one robot arm doesn't transfer cleanly to a different arm, gripper or camera setup.
- **It needs actions, not just observations.** Video of people doing tasks is plentiful. Recordings of the exact motor commands that achieved them are not.
- **It rarely includes failure.** Nobody wants to crash a robot on purpose, so datasets underrepresent the mistakes a policy most needs to learn from.

The result is that embodied AI can't simply follow the language-model playbook of gathering more data. The data has to come from somewhere else.

## Simulation as a data engine

A physics simulator is a machine for producing robot experience. It moves a virtual world forward in time, a few milliseconds per step, applying gravity, friction, contact and motor forces. Record what happens, and you have training data.

What makes this strategically important is not that simulated data is free. It's that simulation gives you five levers that real-world collection doesn't:

1. **Volume on demand.** Simulations run faster than real time and in parallel. NVIDIA reported generating [780,000 synthetic trajectories in 11 hours](https://nvidianews.nvidia.com/news/nvidia-isaac-gr00t-n1-open-humanoid-robot-foundation-model-simulation-frameworks) — the equivalent of about 6,500 hours of human demonstrations.
2. **Diversity by design.** Every episode can change masses, friction, lighting, colors, camera angles and starting conditions. This is called domain randomization. The model can't memorize one world, so it has to learn what actually matters.
3. **Perfect labels.** The simulator knows the exact position, velocity and force of everything. On a real robot, much of that is expensive or impossible to measure.
4. **Safe failure.** Robots can fall, collide and fail thousands of times at no cost. That fills the gap real datasets leave.
5. **Interactive practice.** A simulator isn't only a data source; it's a practice ground. A policy can act, see the consequences and improve through reinforcement learning, generating fresh data from its own mistakes.

The first three levers address how much data you have. The last two address what kind of data you have, and they matter just as much.

The same NVIDIA announcement reported that combining synthetic data with real data improved their GR00T N1 humanoid model's performance by 40% compared with real data alone. That is the pattern to watch: synthetic data doesn't need to replace real data to be valuable. It needs to multiply it.

## Putting the thesis to the test

I wanted to see these levers work, not just read about them. So I built the whole loop at small scale, using MuJoCo (an open-source physics engine from Google DeepMind), Hugging Face's LeRobot dataset format and PyTorch.

The task: balance a pole on a moving cart using **only camera images** — no direct access to the pole's angle. It's the "hello world" of control, but learning it from pixels makes it a real perception-and-action problem.

```
Simulate  ->  Generate synthetic data  ->  Imitate  ->  Evaluate  ->  Practice with RL
(MuJoCo)      (expert demos recorded       (copy the    (unseen       (reward from the
               in randomized worlds)        expert)      worlds)       simulator)
```

### Where the demonstrations come from

A simulator only computes consequences: apply a force, and MuJoCo tells you what happens next. It never decides what force to apply. Left alone, the pole simply falls.

So each demonstration needs an **expert**: a controller that picks the right action at every step while the simulator records the result. Mine was LQR, a standard technique from control theory that computes the best balancing rule directly from the simulator's physics. It was recalculated for every randomized world and balanced the pole in 100% of test worlds.

The expert has an unfair advantage: it reads the exact physics, which no real robot can. The student model has to learn the same skill from camera images alone.

**Is the expert just human labeling by another name?** It plays the same role — its actions are the labels the model learns to copy — but it works differently:

- **It demonstrates rather than annotates.** A labeler marks up data after it exists. The expert acts inside the simulation, so it creates the data and its labels in the same pass.
- **Its closest human equivalent is teleoperation,** where a person drives the robot while it's recorded. That's how most real-world robot datasets are collected, and it's the slow, expensive step behind the scarcity described above.
- **It's code.** It never tires, costs nothing per example, and reads physics no human could see in an image.

Replacing the human demonstrator with code in a simulator is what turned weeks of collection into two minutes. The catch is that a code expert only exists for tasks we can solve mathematically. For tasks like folding laundry, teams still start from human demonstrations and use them to seed large-scale synthetic generation — the approach behind NVIDIA's pipeline mentioned earlier.

![The same task across randomized worlds](images/randomization_grid.png)

Three results speak directly to the data-scarcity argument.

### Proof point 1: More synthetic data was the biggest lever

I trained the same model, with the same code and settings, on two amounts of synthetic data:

```
Synthetic data          Time to generate    Success on unseen worlds
 50 episodes                 30 sec               35%
200 episodes                  2 min               85%
```

Nothing about the model changed. Only the data did, and producing four times more of it took two minutes. In the real world, that step would mean weeks of teleoperation. In simulation, it's a command-line flag.

### Proof point 2: The simulator doubled as a practice ground

Imitation learning has a known weakness: the student only sees situations the expert visited. When it drifts somewhere new, it has no example to copy. On 50 unseen worlds, the imitation-trained policy succeeded 74% of the time.

Then I let it practice in the simulator with reinforcement learning. It drove 16 randomized worlds in parallel, the physics scored every move, and the policy updated toward whatever worked. After about seven minutes of practice:

```
Policy                              Success on 50 unseen worlds
Do nothing                                  0%
Imitation only                             74%
Imitation + RL practice                    84%
Expert (reads exact physics)              100%
```

This is the part of simulation that's easy to underrate. RL didn't reuse the dataset at all. It generated its own data — including the mistakes the expert never made — which is exactly the kind of data real-world collection struggles to produce.

### Proof point 3: Simulation supervises what the real world can't

At three points, the pipeline used information only a simulator has:

- The **expert** used the exact physics to act perfectly.
- During imitation, the model also had to predict the **true state** from images — free labels that taught it what to look for.
- During RL, a **critic** network that saw the true state judged each move, while the policy itself saw only pixels.

None of that privileged information is needed once the policy is trained. But during training, it is a form of supervision that simply doesn't exist outside simulation.

![The RL-refined policy balancing in four unseen worlds](images/eval_rl.gif)

## Where synthetic data falls short

Simulation mitigates data scarcity; it doesn't abolish it. Being clear about the limits is what makes the strategy credible.

### The sim-to-real gap

A policy only learns the world it was shown, and no simulator shows the whole real world. My policy never saw camera glare, motor lag, sensor noise or a 50-millisecond delay between seeing and acting. For an unstable system like a balancing pole, delay alone can break a policy that looks perfect in simulation. Small gaps compound through the feedback loop.

The standard defenses are known: randomize widely, measure the real robot and match the simulator to it, simulate delays and noise, render more realistically, and fine-tune on a small amount of real data. Each narrows the gap. None closes it completely.

### Scarcity moves; it doesn't disappear

The deeper lesson from building this myself: simulation converts a data problem into three other problems.

- **Simulators.** Someone has to model the robot, the objects and the physics accurately enough. Contact, deformable objects and liquids remain hard.
- **Experts.** Imitation learning needs a demonstrator. My cart-pole had a perfect controller from textbook control theory. Most real tasks — folding laundry, assembling parts — have no such expert.
- **Rewards.** RL needs a score. "Keep the pole upright" is easy to measure in a simulator. "Fold the shirt neatly" is not.

These problems are more tractable than hand-collecting millions of demonstrations, because each one is solved once and then reused at scale. But they are where the real engineering effort goes.

### The limits of my own experiment

My results come from a single, simple task, and they're noisy. The 85% in proof point 1 came from a quick 20-episode test; the same model scored 74% on a more careful 50-episode test. Even at 50 episodes, the uncertainty is roughly ±6 points. The RL training curve also bounced around rather than climbing smoothly. The direction of every result is clear. The exact numbers should be read loosely.

## Strategic implications

If data is the bottleneck and simulation is how you generate data at scale, several things follow for anyone building embodied AI.

### 1. Invest in data engines, not just datasets

A dataset is a fixed asset that ages. A data engine — simulated worlds, randomization, experts, rewards and evaluation — keeps producing. When a new task, robot or failure mode appears, an engine generates the data for it. The durable advantage is the pipeline, not any single collection of episodes.

### 2. Treat real data as the anchor, synthetic data as the multiplier

Real data keeps a model honest about the physical world. Synthetic data supplies the volume, variety and failure cases real collection can't afford. NVIDIA's 40% gain from combining the two, rather than choosing one, is the right mental model. The practical question is not "sim or real?" but "what ratio, and where does each come from?"

### 3. Make simulation your evaluation layer

In my experiment, the most trustworthy numbers came from running every policy on the same unseen simulated worlds. Real-world testing is slow and hard to repeat exactly. Simulation offers repeatable, large-scale evaluation before anything touches hardware. For teams shipping robot software, that becomes a quality gate much like a test suite.

### 4. Expect the LLM training pattern to repeat

Language models are trained with supervised fine-tuning, then refined with reinforcement learning. My tiny pipeline followed the same arc: imitation on synthetic demonstrations, then RL in simulation. Frameworks such as [RLinf](https://github.com/RLinf/RLinf) now run this pattern for large vision-language-action models across many GPUs; its [paper](https://arxiv.org/abs/2510.06710) reports about 98% success across 130 LIBERO benchmark tasks after RL training. The playbook is converging across modalities.

### 5. Data cost becomes compute cost

When data is generated in simulation, its cost moves from people-hours to GPU-hours. That changes budgeting, hiring and scaling: simulation engineers and compute capacity start to matter as much as teleoperation teams. It also means the cost of robot data can fall as compute gets cheaper.

### The tools are already accessible

None of this requires a research lab to explore. MuJoCo runs on ordinary hardware and excels at precise physics. NVIDIA Isaac Sim adds photoreal rendering for camera-heavy tasks, on NVIDIA GPUs. LeRobot gives datasets a shared format on the Hugging Face Hub. RL frameworks like RLinf handle scale when you need it. The barrier to entry is lower than most people assume.

## What building it end to end taught me

Reading about synthetic data and making it work are different experiences. Four lessons only became clear by doing it.

**The simulator does exactly what you describe — including your mistakes.** My first run failed because a decorative rail overlapped the cart, and the physics engine faithfully computed friction that clamped the cart in place. Synthetic data is only as good as the world model behind it.

**The handoff between imitation and RL is the model, not the dataset.** I assumed RL would train on the synthetic dataset. It doesn't. The dataset trains the imitation model; RL starts from that model and generates its own data by practicing. The simulator plays two roles: data factory first, practice ground second.

**Evaluation is where synthetic-data claims live or die.** A 20-episode test told me one story and a 50-episode test told me another. Suspiciously good numbers deserved the same scrutiny as bad ones — one turned out to be a logging bug.

**The whole loop is small enough to explore yourself.** Data generation took minutes, imitation training about 20 minutes, RL practice about seven. The concepts behind industrial robot-learning pipelines are fully explorable without a GPU cluster, which makes them far easier to reason about strategically.

## The bottom line

Embodied AI won't get an internet-sized dataset by accident. It will have to build one. Simulation is the most practical way we have to do that: it turns data collection into data generation, lets robots learn from failures without breaking anything, and turns every policy into something you can test and improve before it touches hardware.

The gap between simulation and reality is real, and it moves the hard work to building good simulators, experts and rewards. But that work compounds in a way hand-collected data never does. Teams that treat simulation as core data infrastructure, rather than a research convenience, will be the ones that scale.

The code for my experiment is on GitHub: [github.com/deepak-vij/mujoco-synthetic-data-pipeline](https://github.com/deepak-vij/mujoco-synthetic-data-pipeline). It runs end to end without specialized hardware.

### Sources

- [Meta: Introducing Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/)
- [DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset](https://arxiv.org/abs/2403.12945)
- [Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://arxiv.org/abs/2310.08864)
- [NVIDIA Announces Isaac GR00T N1 (NVIDIA Newsroom, March 18, 2025)](https://nvidianews.nvidia.com/news/nvidia-isaac-gr00t-n1-open-humanoid-robot-foundation-model-simulation-frameworks)
- [RLinf on GitHub](https://github.com/RLinf/RLinf) and [RLinf-VLA paper](https://arxiv.org/abs/2510.06710)
