# ExploRLLM for Sagittarius SGR532

Reproduces the core of **ExploRLLM** (Ma et al., ICRA 2025) on the
Sagittarius SGR532 6-DOF arm using ROS Noetic + MoveIt + Gazebo.

## Project structure

```
explorllm_sagittarius/
├── envs/
│   └── pick_place_env.py     ← Gazebo gym.Env wrapper  [CORE]
├── llm/
│   └── llm_policy.py         ← LLM exploration (πH + πL)
├── agents/
│   └── custom_sac.py         ← CNN+attention encoder + ExploRLLMSAC
├── perception/
│   └── camera_perception.py  ← Real-camera sim-to-real bridge
├── train.py                  ← Training script (single run + ablation)
└── eval.py                   ← Evaluation + curve plotting
```

---

## Quick-start: training

### 1. Prerequisites

```bash
# In your sagittarius_ws (already set up from labs):
cd ~/sagittarius_ws
source devel/setup.bash

# Python packages:
pip install stable-baselines3[extra] gymnasium openai
pip install torch torchvision   # or use your existing PyTorch

# For plotting:
pip install matplotlib
```

### 2. Copy this folder into sagittarius_ws

```bash
cp -r explorllm_sagittarius ~/sagittarius_ws/src/
```

### 3. Start Gazebo

In terminal 1:
```bash
cd ~/sagittarius_ws && source devel/setup.bash
roslaunch sagittarius_gazebo demo_gazebo.launch
# Click ▷ (play) in Gazebo to start physics
```

### 4. Run training

In terminal 2:
```bash
cd ~/sagittarius_ws/src/explorllm_sagittarius
export LLM_API_KEY="your-api-key-here"

# Single run, ε=0.2, DeepSeek
python train.py --epsilon 0.2 --model deepseek-v3 --seed 0

# Pure SAC baseline (no LLM, no API key needed)
python train.py --epsilon 0.0

# Full ablation (ε={0, 0.2, 0.5} × 3 seeds)
python train.py --ablation --seeds 0 1 2
```

---

## Supported LLM providers

| Provider  | --model value   | base_url                                        |
|-----------|-----------------|-------------------------------------------------|
| OpenAI    | gpt-4o-mini     | (default)                                       |
| DeepSeek  | deepseek-v3     | https://api.deepseek.com/v1                     |
| Kimi      | kimi            | https://api.moonshot.cn/v1                      |
| Qwen      | qwen            | https://dashscope.aliyuncs.com/compatible-mode/v1 |

All use the OpenAI-compatible SDK. Just set `--api-key` and `--model`.

---

## Evaluation

```bash
# Evaluate in Gazebo
python eval.py --model-path logs/eps0.2_seed0/final_model.zip --n-episodes 30

# Plot ablation training curves
python eval.py --plot --log-dir logs/
```

---

## Real robot deployment (sim-to-real)

1. Run `roslaunch sagittarius_moveit demo_true.launch`
2. Run Lab2 HSV calibration: `roslaunch sagittarius_object_color_detector camera_calibration_hsv.launch`
3. Update calibration in `perception/camera_perception.py` (or call `load_calibration_from_yaml`)
4. Run: `python eval.py --model-path logs/eps0.2_seed0/final_model.zip --real-robot`

---

## Key differences from ExploRLLM paper

| Paper (UR5e)            | This project (SGR532)               |
|-------------------------|-------------------------------------|
| ViLD open-vocab detector| Lab2 HSV color detector             |
| Real RGB-D camera       | Gazebo GT + noise during training   |
| Suction gripper         | Parallel gripper (sagittarius_gripper)|
| 500mm workspace         | ~350mm workspace (scaled scene)     |
| 6 seeds × 9 ε values    | 3 seeds × 3 ε values (ablation)    |
| EAGERx framework        | stable-baselines3 + gym.Env wrapper |
