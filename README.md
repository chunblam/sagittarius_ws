# ExploRLLM for Sagittarius SGR532

**LLM引导的强化学习探索用于机械臂操作** — 复现 ExploRLLM (Ma et al., ICRA 2025) 核心方法，部署在香港中文大学（深圳）实验室的 Sagittarius SGR532 6-DOF 机械臂上。

---

## 项目概述

本项目将 ExploRLLM 的核心思想落地在 Sagittarius SGR532 上：使用大语言模型（LLM）为强化学习（SAC）提供语义引导的探索动作，显著加速收敛。

**最终演示效果：** 给机械臂一条自然语言指令（例如"把黄色方块放进蓝色桶"），机械臂从随机摆放的桌面上找到对应颜色的方块，抓取，然后放入指定颜色的垃圾桶。方块和垃圾桶每次测试都可以随意摆放，系统自动识别位置并执行。

**和 Lab2 的本质区别：** Lab2 是手工编程的规则系统（颜色→坐标写死），本项目是从数万次试错中学习的智能体，学会的是通用的"看到目标就去抓"策略，不依赖任何硬编码规则。

---

## 项目结构

```
sagittarius_ws/
├── config/
│   └── color_config.py          ← 颜色配置（自动从Lab2 yaml加载）
├── envs/
│   └── pick_place_env.py     ← Gazebo gym.Env 包装器（核心）
├── llm/
│   └── llm_policy.py         ← LLM 探索策略（πH + πL）
├── agents/
│   └── custom_sac.py         ← CNN+Attention+Embedding 编码器 + ExploRLLMSAC
├── perception/
│   └── camera_perception.py  ← 真机感知（面积区分方块/桶）
├── train.py                  ← 训练脚本（单次 + 消融实验）
├── eval.py                      ← 评估 + 训练曲线绘图
├── test_all.py                  ← 分步环境验证测试
├── calibration_guide.py         ← 标定值读取工具
├── pick_place_scene.world    ← Gazebo 场景（6 色 12 物体模型；训练按课程 2+2 / 3+2 激活子集）
├── docs/
│   └── SH_PLANNING.md        ← SH 策略 + 高层规划（后续方案，未实现）
└── README.md
```

---

## 核心升级（v2 相比初版）

| 功能 | 初版 | v2（当前版本） |
|------|------|----------------|
| 支持颜色数量 | 固定3种（红绿蓝） | 任意多种（从Lab2 yaml自动加载） |
| 垃圾桶位置 | 写死固定坐标 | 每个episode随机，桶的位置也是observation的一部分 |
| 颜色编码方式 | 3维one-hot（固定大小） | N维Embedding index（颜色数变化无需改网络） |
| 方块/桶区分 | 分开处理 | 同一HSV检测，面积阈值自动区分 |
| 观测向量 | crops + block_pos + gripper + lang | crops + block_pos + **bin_pos** + gripper + task |
| 动作原语 | （历史）分步 pick / place | **单次 `pick_and_place`**：一步 `env.step` 内由 **`_execute_pick_and_place`** 连续执行（移动到方块→抓取→抬起→**同一函数内**过渡到桶上方→放置），**不是**先调 `_execute_pick` 再调 `_execute_place` 两次高层入口；向桶运动前会 **`set_start_state_to_current_state()`** 以持块姿态延续规划。**动作维度 7** = `[pick_block_idx, place_bin_idx, pose_id, rpx, rpy, rbx, rby]`；子阶段间隔 **`INTER_ROBOT_STAGE_PAUSE_S=0.5s`**。旧 checkpoint **不兼容**，需重新训练。 |

**长任务（多步全配对）与 SH 策略如何组合**：见 **[`docs/SH_PLANNING.md`](docs/SH_PLANNING.md)**（规划层方案，当前未在代码中实现）。

---

## 快速开始

### 1. 环境依赖

```bash
# ROS + MoveIt（已在实验室电脑配置好）
source ~/sagittarius_ws/devel/setup.bash

# Python 包
pip install stable-baselines3[extra] gymnasium openai torch torchvision matplotlib pyyaml
```

### 2. 加入 Gazebo 场景模型

将 `pick_place_scene.world` 的所有 `<model>` 标签复制到你的 Gazebo world 文件中（`<world>` 标签内）。

包含 6 种颜色的方块（`{color}_block`）和 6 种颜色的垃圾桶（`{color}_bin`）。工作台在 world 中为 **70×70 cm**（中心约 x=0.28）。

**训练课程与观测维度**：`train.py` 使用 **`--curriculum {2+2,3+2}`**（默认 **`2+2`**：2 方块 + 2 桶；**`3+2`**：3 方块 + 2 桶，多 1 个无对应桶的干扰块）。策略网络侧 **`n_active=SLOT_COUNT=3` 固定**（`obs_dim` 不变）。物体摆放在 **`PLACE_RECT_*`**（底座前方作业矩形，见 `pick_place_env.py` 常量）内，且**永不**进入 **`ROBOT_BASE_EXCLUSION_*`** 碰撞圆（与 **`ARM_BASE_*` 可达环**是两回事：后者仅用于 IK 可达判定）。**物体中心最小间距 0.08m**（`MIN_OBJECT_CENTER_GAP`，xy 平面两两中心距离）；摆放用多轮 max-spread 贪心（`OBJECT_PLACEMENT_SPREAD_TRIALS`）使分布更均匀。网格落位微抖动 **`OBJECT_PLACE_JITTER` 默认 ±8mm**；`step` 里动作用 `OBJECT_ZONE_*` 做宽松裁剪。抓取默认 **`GRASP_ORIENTATION_MODE=horizontal`**（侧向水平对准目标）；基础四元数见 **`TCP_HORIZONTAL_BASE_QUAT`**，若 URDF 中 `ee_link` 定义不同需在 RViz/MoveIt 中微调；可调 **`TCP_GRASP_XY_BACKOFF_M`**。若仍碰底座或偏一侧，在 `pick_place_env.py` 中微调 **`ROBOT_BASE_EXCLUSION_XY` / `ROBOT_BASE_EXCLUSION_RADIUS` / `PLACE_RECT_*`**。**与旧 checkpoint 不兼容时需重新训练**。

### 3. MoveIt 命名空间（SGR532 仿真默认）

若参数在 **`/sgr532/robot_description`**、MoveIt 话题在 **`/sgr532/move_group/...`**（`rostopic list | grep move_group` 可见），本仓库默认 **`EXPLORELLM_MOVEIT_NS=sgr532`**，代码会：

- 用绝对路径 **`/sgr532/robot_description`** 加载模型；
- 用 **`ns=sgr532`** 连接 **`/sgr532/move_group`** action（仅 `ns` 或仅短路径在 Noetic 上常连不上）。

可选：连接超时（秒，默认 30）：

```bash
EXPLORELLM_MOVEIT_WAIT=45
```

单次规划时间与重试次数在 `pick_place_env.py` 中由 **`MOVEIT_PLANNING_TIME_S`**（默认 4s）、**`MOVEIT_NUM_PLANNING_ATTEMPTS`**（默认 32）控制；难姿态下若仍易超时，可略增时间或检查场景碰撞。

若 launch 使用**根**命名空间（`/robot_description` + `/move_group`），在 `.env` 中设置：

```bash
EXPLORELLM_MOVEIT_NS=root
```

（`root` / `none` / 空 含义见 `env_config.moveit_commander_ns`。）

**说明**：`/gazebo/...`、`/usb_cam/...` 等一般为全局话题，无需加 `sgr532` 前缀。

**碰撞后仍要可靠 reset（训练多轮）**：`env.reset()` 会先尝试多种 MoveIt 方式回 **`home`**，失败时若 Gazebo 可用且未关闭，会调用 **`/gazebo/reset_simulation`** 重置物理，随后仍由 **`_randomize_scene`** 重新传送方块/桶。默认开启：

```bash
EXPLORELLM_GAZEBO_RESET_SIMULATION_ON_HOME_FAIL=1   # 设为 0 可关闭
```

`reset` 返回的 `info` 中含 **`reset_home_ok`**（布尔），便于统计「本局是否成功回到 home」。

**SRDF named states**：`pick_place_env` 使用与 RViz 一致的 **`home`**（机械臂）、夹爪 **`open`**（张开）、**`middle`**（抓取夹持，约 5cm 块；勿用完全 **`close`**，否则指尖贴死易弹飞物体）。名称在 `envs/pick_place_env.py` 顶部常量 `MOVEIT_ARM_HOME_STATE`、`MOVEIT_GRIPPER_*`；若你包内不同，请改常量以匹配 SRDF。

### 4. 验证环境

```bash
# 先在另一个终端启动Gazebo
roslaunch sagittarius_gazebo demo_gazebo.launch
# 点击 ▷ 开始仿真

# 然后运行测试（共8项，逐步验证）
python test_all.py

# 只测某一项，例如只测Gazebo物体是否加载
python test_all.py --test 2
```

### 5. 开始训练

```bash
# 纯SAC（不需要API Key，用于验证训练循环能跑通）
python train.py --epsilon 0.0

# 推荐先训 2+2（2 方块 + 2 桶），再训 3+2（多 1 个干扰方块）
python train.py --epsilon 0.0 --curriculum 2+2
# python train.py --epsilon 0.0 --curriculum 3+2

# 完整 ExploRLLM（DeepSeek，推荐）
export LLM_API_KEY="sk-你的key"
python train.py --epsilon 0.2 --model deepseek-v3 --seed 0

# 只用3种颜色训练（减少训练难度，适合初次测试）
python train.py --epsilon 0.2 --colors red green blue --api-key sk-...

# 消融实验（ε∈{0, 0.2, 0.5} × 3 seeds）
python train.py --ablation --seeds 0 1 2 --api-key sk-...
```

---

## 支持的 LLM 提供商

| 提供商 | `--model` 参数 | API 地址 |
|--------|---------------|----------|
| DeepSeek（推荐，性价比高） | `deepseek-v3` | `https://api.deepseek.com/v1` |
| Kimi / Moonshot | `kimi` | `https://api.moonshot.cn/v1` |
| Qwen（阿里） | `qwen` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| OpenAI | `gpt-4o-mini` | 默认 |

所有提供商均使用 OpenAI 兼容接口，只需修改 `--model` 和 `--api-key`，其余代码不变。

---

## 颜色配置

颜色列表和 HSV 阈值从 Lab2 的 `vision_config.yaml` 自动加载：

```bash
# 加载方式1：自动搜索（会找到Lab2标定保存的yaml）
python train.py --epsilon 0.2

# 加载方式2：手动指定yaml路径
python train.py --yaml-path ~/sagittarius_ws/src/.../vision_config.yaml

# 加载方式3：手动指定颜色列表（覆盖yaml）
python train.py --colors red green blue yellow
```

新增颜色无需改代码，只需在 Lab2 的 HSV 阈值调整程序里标定好新颜色，重新保存 yaml，下次启动训练时自动加载。

---

## 真机部署（sim-to-real）

### 前置步骤

1. 运行 Lab2 颜色标定：
   ```bash
   roslaunch sagittarius_object_color_detector hsv_params.launch
   # 调整好红绿蓝黄粉橙各颜色的HSV阈值
   ```

2. 运行 Lab2 视觉标定（坐标系对齐）：
   ```bash
   roslaunch sagittarius_object_color_detector camera_calibration_hsv.launch
   # 完成5点标定，记录终端输出的 kx, bx, ky, by 值
   ```

3. 更新标定值（运行自动更新工具）：
   ```bash
   python calibration_guide.py
   # 会自动读取yaml并提示你更新 perception/camera_perception.py
   ```

### 桌面摆放规则

仿真中方块与桶均在同一作业矩形 **`PLACE_RECT_*`**（当前约 **x: 0.18~0.42 m**，**y: ±0.18 m**，以 `pick_place_env.py` 为准）内随机摆放，并满足与底座碰撞圆、物体间距等约束；真机可仍按「左方块、右桶」大致分区，与仿真不必逐厘米一致。

| 说明 | 物体 | 典型 x（真机参考） | y 范围 |
|------|------|-------------------|--------|
| 左偏 | 彩色方块 | 较小 x | -0.18 ~ 0.18 m |
| 右偏 | 彩色垃圾桶 | 较大 x | -0.18 ~ 0.18 m |

方块和垃圾桶**不需要精确测量位置**，摄像头会实时检测。

**同颜色的方块和桶不会混淆**：系统用面积区分——方块顶面小、桶开口大，面积差异在2~3倍，自动分类，不依赖人工干预。

### 运行真机评估

```bash
# 启动机械臂
roslaunch sagittarius_moveit demo_true.launch

# 评估（使用摄像头感知替代Gazebo GT）
python eval.py --model-path logs/eps0.2_seed0/final_model.zip --real-robot
```

---

## 评估与绘图

```bash
# 在Gazebo里评估30个episode
python eval.py --model-path logs/eps0.2_seed0/final_model.zip --n-episodes 30

# 只评估3种颜色（快速验证）
python eval.py --model-path ... --colors red green blue --n-episodes 15

# 绘制消融实验训练曲线（需先完成ablation训练）
python eval.py --plot --log-dir logs/
# 输出：results/training_curves.png
```

---

## 实验设计

本项目的核心实验贡献是消融实验，验证 LLM 引导探索的有效性：

| 方法 | ε 值 | 预期效果 |
|------|------|----------|
| 纯 SAC | ε = 0 | 收敛慢、不稳定，约 40k 步才能达到 50% 成功率 |
| ExploRLLM | ε = 0.2（推荐） | 约 20k 步达到 80% 成功率 |
| ExploRLLM | ε = 0.5 | 比 0.2 略慢但也优于纯 SAC |
| 纯 LLM | —— | 固定约 55% 成功率，不随步数提升 |

报告应包含：① 训练曲线对比图、② 不同 ε 的成功率表格、③ 真机 Demo 视频。

---

## 与 ExploRLLM 原论文的差异

| 维度 | 原论文（UR5e） | 本项目（SGR532） |
|------|---------------|-----------------|
| 感知方案 | ViLD 开放词汇检测 | Lab2 HSV 颜色检测 + 面积区分 |
| 训练感知 | GT位置 + 高斯噪声 | GT位置（Gazebo）+ 高斯噪声 |
| 末端执行器 | 吸盘 | 平行夹爪 |
| 工作空间 | ~500mm | ~350mm（场景按比例缩放） |
| 消融规模 | 6 seeds × 9 ε 值 | 3 seeds × 3 ε 值 |
| LLM框架 | EAGERx | stable-baselines3 + gym.Env |
| 颜色数量 | 3种字母颜色 | 最多6种（红绿蓝黄粉橙） |
| 桶的位置 | 固定 | 随机（本项目额外升级） |

---

## 常见问题

**Q: 训练时报 `rosmaster not running` 错误**
A: 先启动 `roslaunch sagittarius_gazebo demo_gazebo.launch`，再运行训练脚本。

**Q: Gazebo里看不到方块和垃圾桶**
A: 把 `pick_place_scene.world` 的 `<model>` 标签复制进 Gazebo world 文件，注意物体名是 `{color}_bin`（不是旧版的 `{color}_bowl`）。

**Q: `ColorConfig` 加载后颜色只有3种，不是6种**
A: vision_config.yaml 里只标定了3种颜色。运行 Lab2 的 HSV 阈值调整程序为其余颜色调好阈值后，yaml 会自动更新。

**Q: 真机抓取时方块和桶识别混淆**
A: 检查 `BLOCK_MAX_AREA` 和 `BIN_MIN_AREA` 阈值（在 `camera_perception.py` 第15行附近）。可以运行 `python test_all.py --test 8` 查看实际检测的面积值，根据你的摄像头高度调整阈值。

**Q: LLM API 调用慢或失败**
A: DeepSeek 在国内延迟最低，推荐使用。如果 API 调用失败，系统会自动退回到 SAC 动作，训练不会中断。

---

## 参考文献

Ma, R., Luijkx, J., Ajanović, Z., & Kober, J. (2025). ExploRLLM: Guiding Exploration in Reinforcement Learning with Large Language Models. *IEEE ICRA 2025*, 9011–9017.
