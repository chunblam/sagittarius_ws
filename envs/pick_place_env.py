#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pick_place_env.py
====================
升级版环境，相比原版的核心变化：

  变化1：支持任意多种颜色（从 color_config 动态读取）
  变化2：垃圾桶/方块在「底座前方」轴对齐矩形（PLACE_RECT_*）内随机，并禁止落入底座碰撞圆
  变化3：observation 向量用任务颜色 index（固定槽位数）
  变化4：observation 包含桶的位置（桶每回合随机）

训练课程 curriculum_mode：
  - "2+2"：2 个方块 + 2 个桶（颜色子集独立打乱），任务从方块集合与桶集合中各抽一个颜色。
  - "3+2"：3 个方块 + 2 个桶（桶颜色为方块颜色的二元子集），多 1 个无对应桶的干扰方块。

Observation 向量（SLOT_COUNT=3 固定，与策略网络维度一致；未满槽位填零）：
  [img_patches | block_positions | bin_positions | gripper | task_encoding]
   3*3*28*28      3*2             3*2             1         2

  task_encoding = [pick_local_idx, place_local_idx]
    pick_local  ∈ 0..n_blocks_ep-1（本回合 _active_block_colors）
    place_local ∈ 0..n_bins_ep-1（本回合 _active_bin_colors）

Action向量：
  [primitive(0-1), obj_index(0..n_blocks+n_bins-1), pose_id(0..K-1), res_x, res_y]
  前 n_blocks 个 index = 方块（_active_block_colors 顺序），后 n_bins 个 = 桶（_active_bin_colors 顺序）。
  当前水平抓取 + 动态 yaw（atan2）下，pose_id 不参与末端朝向，仅保留维度兼容旧策略；可将 pose_id_count 设为 1。
"""

import os
import sys
import time
import math
import itertools
from typing import Tuple, List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import env_config  # noqa: F401 — MoveIt 命名空间 EXPLORELLM_MOVEIT_NS
import rospy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetModelState, DeleteModel, SpawnModel
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose
from std_srvs.srv import Empty

import moveit_commander
from geometry_msgs.msg import PoseStamped
from moveit_commander import PlanningSceneInterface

from config.color_config import ColorConfig, get_color_config

try:
    from envs.gazebo_model_io import model_xml_for_spawn
except ImportError:
    from gazebo_model_io import model_xml_for_spawn


# ── 场景参数 ──────────────────────────────────────────────────────────────────

# 策略网络 / 观测向量使用的固定槽位数（与 ExploRLLMSAC 的 n_colors=3 一致）
SLOT_COUNT = 3
# 兼容旧脚本名：等同于 SLOT_COUNT
ACTIVE_COLORS_PER_EPISODE = SLOT_COUNT
DEFAULT_POSE_ID_COUNT = 1

# 与 pick_place_scene.world 中声明的 {color}_block / {color}_bin 一致。
# 用于 reset 时把「未参与本回合」的物体全部移出桌面：若仅用 color_cfg.colors
# 切片求 inactive，当 yaml 只配置了 3 种颜色时 inactive 为空，world 里其余 3+3
# 个模型会永远留在桌上。
WORLD_SPAWN_COLORS = ("red", "green", "blue", "yellow", "pink", "orange")

# 可达性约束（基于当前台面与机械臂安装关系的保守圆域）
ARM_BASE_X = 0.28
ARM_BASE_Y = 0.0
# 经验可达环域（保守）：过近/过远位置在该机型上更容易触发规划超时
ARM_REACH_MIN_R = 0.15
ARM_REACH_MAX_R = 0.38

# ── 物体随机摆放（重要坐标说明）────────────────────────────────────────────
# ROBOT_BASE_EXCLUSION_XY：机械臂**物理底座**在桌面上的投影中心。
# 从实验室照片可以看出底座大致在桌面左侧区域，x ≈ 0.08，y ≈ 0.0。
# 注意：这与 ARM_BASE_X(0.28) 不同，后者是 IK 可达环的参考中心。
ROBOT_BASE_EXCLUSION_XY = (0.10, 0.0)   # 底座投影中心（米）
ROBOT_BASE_EXCLUSION_RADIUS = 0.15      # 底座+连杆最近占地半径（留 5cm 安全裕量）

# 作业矩形：底座「前方」的理想抓取区域。
# x: 0.18~0.40（离底座 8~30cm，机械臂前方中段，避免过近卡奇异点）
# y: ±0.18（对称，与桌面黑框对齐）
# 方块放左半段 (x 较小)，垃圾桶放右半段 (x 较大)，整体在一个矩形内统一管理
PLACE_RECT_X = (0.18, 0.42)
PLACE_RECT_Y = (-0.18, 0.18)

# 仅用于 step() 里对动作的矩形裁剪（略宽于 PLACE_RECT）
OBJECT_ZONE_X = (0.15, 0.44)
OBJECT_ZONE_Y = (-0.20, 0.20)
# 物体中心之间最小距离（米）；放宽至 4cm 以增加可行摆放与布局多样性
MIN_OBJECT_CENTER_GAP = 0.04
# 网格落位时在 x/y 上的一次性随机偏移（米），仅在 reset 传送时加一次，**不是**物理引擎每帧抖动。
# 默认 ±8mm：略打破完全规则的网格对齐；设为 0 则与「摆好后位置固定」一致。
OBJECT_PLACE_JITTER = 0.008

# 首次 spawn 临时落点（远离桌面，避免与桌台重叠）
SPAWN_PARK_X = 2.0
SPAWN_PARK_Y0 = 0.0

# 高度参数（z轴，单位米）
# ──────────────────────────────────────────────────────────
# 坐标系约定（与 world 文件一致）：
#   桌面顶面 z = 0.00（world文件桌面顶面在z=0）
#   方块为 5cm 立方体：world 中方块底面 z≈0.001，重心 z≈0.026；逻辑上用 TABLE_Z+BLOCK_H/2
#   桶底面   z = 0.00，桶重心   z = BIN_H/2   = 0.06
#
# GRASP_H 修正：
#   方块顶面在 z = TABLE_Z + BLOCK_H（5cm 立方体时 0.05m）
#   末端目标约在方块中部附近：TABLE_Z + GRASP_H ≈ TABLE_Z + BLOCK_H/2 + 小裕量
#   当前 GRASP_H=0.085 → TABLE_Z+0.085（在旧版 0.065 上再 +2cm，减轻侧向伸入时蹭方块）
#
# APPROACH_H 修正：
#   接近高度必须明显高于最高物体（桶高0.12m）加安全裕量
#   TABLE_Z + APPROACH_H = 0.24m（高于桶顶约 12cm，便于抬起后验证与 TCP/方块 COM 偏差）
# ──────────────────────────────────────────────────────────
TABLE_Z       = 0.00   # 桌面顶面（world文件里桌面顶面在 z=0）
BLOCK_H       = 0.05   # 方块高度（与 world 中 5cm 立方体一致）
BLOCK_W       = 0.05   # 方块边长（x/y，立方体）
BIN_H         = 0.12   # 垃圾桶高度（外高）
BIN_INNER_W   = 0.065  # 垃圾桶内腔宽度（x/y）
BIN_WALL_T    = 0.0025 # 壁厚/底厚（与world一致）

APPROACH_H    = 0.24   # 接近/抬起安全高度（末端 z = TABLE_Z + APPROACH_H）
# 抓取高度：世界系 z（ee_link 原点）。方块重心约在 TABLE_Z+BLOCK_H/2；末端 TCP 与几何中心有偏差时，
# 过低的单一目标易刮桌面/蹭方块；当前 GRASP_H 已较早期值抬高（含 +2cm 裕量）。
GRASP_H       = 0.085  # 最终抓取末端 z = TABLE_Z + GRASP_H（世界系，原 0.065 上 +2cm）
# 方块顶面约 TABLE_Z+BLOCK_H，先降到顶面上方再最终下降，避免大跨度直线“扫”过方块
PRE_GRASP_CLEAR_Z = 0.04  # 在方块顶面上方预留的间隙（米），再落爪（原 0.02 上 +2cm → pre_z≈0.09）

PLACE_H       = 0.11   # 放置高度（末端 z = 0.11m，接近桶口内部）

# 抓取后「是否已离地可搬运」的验证：方块中心 z 须高于此值（米）。
# 旧版用 TABLE_Z+BIN_H+0.02≈0.14，等价于要求 COM 高于桶口；水平侧抓时 COM 常在 0.10~0.12 即已离地，
# 会误判失败并在验证前开爪。此处仅验证「已抬起」而非「高于桶口」（入桶由 place 后 _check_done 判定）。
PICK_LIFT_VERIFY_Z = TABLE_Z + 0.10
# 抬起后若方块 COM 仍低于 14cm（world z），再尝试一次「绕末端局部 X 的小 pitch」抬腕辅助。
PICK_LIFT_ASSIST_Z = TABLE_Z + 0.14
# 抬腕辅助：绕末端局部 X 轴（弧度），约 7°；若方块 z 反降可调小或改符号。
PICK_LIFT_ASSIST_PITCH_RAD = 0.12

# 抓取时笛卡尔目标修正（米）：MoveIt 的 ee_link/TCP 常与「两指中间平面」有偏差；若 URDF 把 TCP 放在
# 指尖外侧，直接以方块中心为 IK 目标会表现为「伸得太外、方块在指尖/连杆下」。沿「基座→方块」
# 方向把目标往基座收一小段，使闭合时两指中心更对准方块中心。真机可按 RViz TF 微调。
TCP_GRASP_XY_BACKOFF_M = 0.022

# 抓取末端姿态（关键参数）
# ──────────────────────────────────────────────────────────
# 理想姿态（参照实物图）：夹爪从侧面水平伸入，末端 TCP 的 Z 轴朝前，
# 平行于地面，朝向方块方向（由 yaw 决定）。
# 不使用"从上往下俯视"（topdown），原因：
#   1. SGR532 的 TCP 定义在侧面，俯视抓取需要关节进入奇异构型，规划成功率低
#   2. 水平侧向抓取与实物验证一致（图3姿态）
#   3. 姿态固定后消除了"多候选轮询→每次姿态不同→末端乱动"的问题
#
# GRASP_ORIENTATION_MODE = "horizontal"
#   在 _move_to_xy 里优先使用：绕 Y 轴 +90°（末端朝前向下水平），叠加朝向方块的 yaw
#   若规划失败，二次尝试时只放宽容差，不换姿态类型。
GRASP_ORIENTATION_MODE = "horizontal"

# 绕 Y 轴 +90° 的四元数（使末端 Z 轴从指向上方转为指向前方）
# q_y90 = (0, sin(45°), 0, cos(45°)) = (0, 0.7071, 0, 0.7071)
TCP_HORIZONTAL_BASE_QUAT = (0.0, 0.7071067811865476, 0.0, 0.7071067811865476)

# 图像crop参数
CROP_SIZE     = 28
# 仅加在「观测向量」的 block/bin 平面坐标上（模拟视觉/标定误差），不写入执行路径。
# 典型幅度：σ=0.03m → 约 95% 落在真值 ±6cm 内；真机同样有检测噪声，可改小 σ 做消融。
# step() 中运动目标使用 Gazebo GT，不受此项影响。
POSITION_NOISE_SIGMA = 0.030

# MoveIt SRDF 中的 group_state 名称（与 RViz MotionPlanning 一致，区分大小写）
MOVEIT_ARM_HOME_STATE = "home"
MOVEIT_GRIPPER_OPEN_STATE = "open"
# 抓取时用 middle：完全 close 时两指贴死，无法容纳 5cm 方块且易弹飞；与实验示例一致
MOVEIT_GRIPPER_GRASP_STATE = "middle"
# MoveIt OMPL：单次 plan 允许的最长时间（秒）与单次规划请求内的并行/重试次数。
# 单次时间过长（如 30s）在 z=APPROACH_H 等难姿态下易长时间卡在 TIMED_OUT；缩短单次上限并增加尝试次数，
# 失败更快换随机种子重试（不保证「立刻」有解，但典型上缩短用户感知的单次卡顿）。
MOVEIT_PLANNING_TIME_S = 4.0
MOVEIT_NUM_PLANNING_ATTEMPTS = 32

# Gazebo模型名称约定：{color}_block, {color}_bin
def block_name(color: str) -> str: return f"{color}_block"
def bin_name(color: str)   -> str: return f"{color}_bin"


def _quat_multiply(
    qa: Tuple[float, float, float, float], qb: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    """Hamilton 乘积 q_a * q_b（与 _pose_horizontal 中乘法一致）。"""
    ax, ay, az, aw = qa
    bx, by, bz, bw = qb
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


class SagittariusPickPlaceEnv(gym.Env):
    """
    升级版 Sagittarius pick-and-place 环境。

    支持：
      - 任意多种颜色（由ColorConfig决定）
      - 方块和垃圾桶每个episode都随机位置
      - 自然语言指令：指定pick哪种颜色、place进哪种颜色的桶
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self,
                 task: str = "short_horizon",
                 max_steps: int = 10,
                 noise_sigma: float = POSITION_NOISE_SIGMA,
                 color_config: ColorConfig = None,
                 yaml_path: str = None,
                 pose_id_count: int = DEFAULT_POSE_ID_COUNT,
                 curriculum_mode: str = "2+2"):

        super().__init__()
        self.task        = task
        self.max_steps   = max_steps
        self.noise_sigma = noise_sigma
        self.color_cfg   = color_config or get_color_config(yaml_path)

        cm = str(curriculum_mode).strip().lower().replace("-", "+")
        if cm not in ("2+2", "3+2"):
            raise ValueError(
                f"curriculum_mode 必须是 '2+2' 或 '3+2'，收到: {curriculum_mode!r}")
        self.curriculum_mode = cm
        if cm == "2+2":
            self.n_blocks_ep = 2
            self.n_bins_ep = 2
        else:
            self.n_blocks_ep = 3
            self.n_bins_ep = 2
        if self.color_cfg.n_colors < self.n_blocks_ep:
            raise ValueError(
                f"课程 {cm} 需要至少 {self.n_blocks_ep} 种颜色，"
                f"当前 color_cfg 仅 {self.color_cfg.n_colors} 种")

        self.pose_id_count = max(1, int(pose_id_count))
        self.pose_yaw_candidates = np.linspace(
            -math.pi / 2.0, math.pi / 2.0, self.pose_id_count
        ).astype(np.float32)
        # 本回合方块 / 桶颜色列表（不等长；3+2 时 len(block_colors)>len(bin_colors)）
        self._active_block_colors: List[str] = []
        self._active_bin_colors: List[str] = []

        # 固定槽位数（与 SAC / ObjectCentricExtractor 一致）
        N = SLOT_COUNT
        self.n_active = N
        self.N = N

        # 当前 episode 的任务颜色
        self.pick_color:  str = ""
        self.place_color: str = ""
        self._step_count   = 0
        self._gripper_open = True
        self._holding_color: str = None
        # reset() 中 _return_home 是否成功（供 info 与日志；碰撞后 home 失败时可能为 False）
        self._last_home_ok: bool = True

        # ── 观测空间 ──────────────────────────────────────────────────────
        # img: SLOT_COUNT * 3 * 28 * 28
        # block_positions: SLOT_COUNT * 2
        # bin_positions:   SLOT_COUNT * 2
        # gripper: 1
        # task: 2 (pick_idx, place_idx，整数但存为float)
        img_dim   = N * 3 * CROP_SIZE * CROP_SIZE
        pos_dim   = N * 2   # blocks
        bin_dim   = N * 2   # bins
        grip_dim  = 1
        task_dim  = 2
        obs_dim   = img_dim + pos_dim + bin_dim + grip_dim + task_dim

        self.img_dim  = img_dim
        self.pos_dim  = pos_dim
        self.bin_dim  = bin_dim
        self.obs_dim  = obs_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

        # ── 动作空间 ──────────────────────────────────────────────────────
        # obj_index 最大 = n_blocks_ep + n_bins_ep - 1（2+2→3，3+2→4）
        _max_obj_idx = float(self.n_blocks_ep + self.n_bins_ep - 1)
        self.action_space = spaces.Box(
            low=np.array( [0.0, 0.0, 0.0, -0.05, -0.05], dtype=np.float32),
            high=np.array(
                [1.0, _max_obj_idx, float(self.pose_id_count - 1), 0.05, 0.05],
                dtype=np.float32,
            )
        )

        # ── 内部状态 ──────────────────────────────────────────────────────
        self._ros_initialized = False
        self._moveit_arm      = None
        self._moveit_gripper  = None
        self._model_states    = None
        self._latest_image    = None
        self._bridge          = None
        self._delete_model_srv = None
        self._spawn_model_srv  = None
        self._planning_scene   = None

        # 当前episode中物体位置缓存（含噪声）
        self._block_positions: dict = {}   # color → np.array([x,y])
        self._bin_positions:   dict = {}   # color → np.array([x,y])

    @property
    def _active_colors(self) -> List[str]:
        """兼容旧接口：本回合场景中出现的颜色（去重后排序）。"""
        return sorted(
            set(self._active_block_colors) | set(self._active_bin_colors))

    # ── ROS初始化 ──────────────────────────────────────────────────────────

    def _init_ros(self):
        if self._ros_initialized:
            return

        if not rospy.core.is_initialized():
            rospy.init_node("explorllm_env", anonymous=True)

        moveit_commander.roscpp_initialize(sys.argv)
        mg_kw = env_config.moveit_move_group_commander_kwargs()
        if mg_kw:
            # 绝对参数路径 + ns：与 /sgr532/robot_description、/sgr532/move_group 一致
            self._moveit_arm = moveit_commander.MoveGroupCommander(
                "sagittarius_arm", **mg_kw)
            self._moveit_gripper = moveit_commander.MoveGroupCommander(
                "sagittarius_gripper", **mg_kw)
        else:
            self._moveit_arm = moveit_commander.MoveGroupCommander("sagittarius_arm")
            self._moveit_gripper = moveit_commander.MoveGroupCommander(
                "sagittarius_gripper")

        # 位置容差单位：米（0.012 = 12mm，非 cm）。姿态容差：弧度（约 0.12rad ≈ 6.9°）
        self._moveit_arm.set_goal_position_tolerance(0.012)
        self._moveit_arm.set_goal_orientation_tolerance(0.12)
        self._moveit_arm.set_max_velocity_scaling_factor(0.5)
        self._moveit_arm.set_max_acceleration_scaling_factor(0.5)
        self._moveit_arm.allow_replanning(True)
        self._moveit_arm.set_planning_time(float(MOVEIT_PLANNING_TIME_S))
        self._moveit_arm.set_num_planning_attempts(int(MOVEIT_NUM_PLANNING_ATTEMPTS))
        self._moveit_gripper.set_goal_joint_tolerance(0.001)
        self._moveit_gripper.set_max_velocity_scaling_factor(0.5)
        # PlanningScene 也必须走与 MoveGroup 相同的命名空间。
        # 否则会等待根命名空间的 /get_planning_scene，出现一直 waiting。
        ps_ns = mg_kw.get("ns", "") if mg_kw else ""
        try:
            self._planning_scene = PlanningSceneInterface(
                ns=f"/{ps_ns}" if ps_ns else "",
                synchronous=True,
            )
        except TypeError:
            # 兼容旧版 moveit_commander（无 synchronous 参数）
            self._planning_scene = PlanningSceneInterface(
                ns=f"/{ps_ns}" if ps_ns else ""
            )
        rospy.sleep(1.0)

        rospy.Subscriber("/gazebo/model_states", ModelStates,
                         self._model_cb, queue_size=1)

        try:
            from sensor_msgs.msg import Image
            from cv_bridge import CvBridge
            self._bridge = CvBridge()
            rospy.Subscriber("/usb_cam/image_raw", Image,
                             self._img_cb, queue_size=1)
        except ImportError:
            pass

        rospy.wait_for_service("/gazebo/set_model_state", timeout=10)
        self._set_model_state = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState)

        try:
            rospy.wait_for_service("/gazebo/delete_model", timeout=5.0)
            self._delete_model_srv = rospy.ServiceProxy(
                "/gazebo/delete_model", DeleteModel)
        except Exception as e:
            rospy.logwarn(
                f"[Env] /gazebo/delete_model 不可用，未激活物体将用远端停放代替删除: {e}")
        try:
            rospy.wait_for_service("/gazebo/spawn_sdf_model", timeout=5.0)
            self._spawn_model_srv = rospy.ServiceProxy(
                "/gazebo/spawn_sdf_model", SpawnModel)
        except Exception as e:
            rospy.logwarn(
                f"[Env] /gazebo/spawn_sdf_model 不可用，重新激活时将无法从 world 模板 spawn: {e}")

        try:
            rospy.wait_for_service("/gazebo/unpause_physics", timeout=5)
            rospy.ServiceProxy("/gazebo/unpause_physics", Empty)()
        except Exception:
            pass

        self._ros_initialized = True
        rospy.loginfo("[Env] ROS初始化完成。支持颜色: %s",
                      self.color_cfg.colors)

    def _sync_moveit_planning_scene(self, ignore_block_color: str = None):
        """
        将当前激活物体同步到 MoveIt PlanningScene：
        - 清理上轮对象（6 色方块+桶）
        - 添加桌面碰撞体
        - 仅添加本轮需要的方块与桶（位姿来自 Gazebo model_states）
        """
        if self._planning_scene is None:
            return

        # 清理旧对象
        try:
            for c in WORLD_SPAWN_COLORS:
                self._planning_scene.remove_world_object(block_name(c))
                self._planning_scene.remove_world_object(bin_name(c))
            self._planning_scene.remove_world_object("workspace_table")
        except Exception:
            pass

        # 添加桌子（与 world 一致）
        table_pose = PoseStamped()
        table_pose.header.frame_id = "world"
        table_pose.pose.position.x = 0.28
        table_pose.pose.position.y = 0.0
        table_pose.pose.position.z = -0.025
        table_pose.pose.orientation.w = 1.0
        self._planning_scene.add_box("workspace_table", table_pose, (0.70, 0.70, 0.05))

        # 激活方块 / 桶（3+2 时仅部分颜色同时有方+桶）
        for c in self._active_block_colors:
            if ignore_block_color is None or c != ignore_block_color:
                bxyz = self._get_pose(block_name(c))
                p = PoseStamped()
                p.header.frame_id = "world"
                p.pose.position.x = float(bxyz[0])
                p.pose.position.y = float(bxyz[1])
                p.pose.position.z = float(bxyz[2])
                p.pose.orientation.w = 1.0
                self._planning_scene.add_box(block_name(c), p, (0.05, 0.05, 0.05))

        for c in self._active_bin_colors:
            zyz = self._get_pose(bin_name(c))
            q = PoseStamped()
            q.header.frame_id = "world"
            q.pose.position.x = float(zyz[0])
            q.pose.position.y = float(zyz[1])
            q.pose.position.z = float(zyz[2])
            q.pose.orientation.w = 1.0
            self._planning_scene.add_box(bin_name(c), q, (0.07, 0.07, 0.12))

    def _model_cb(self, msg):
        self._model_states = msg

    def _img_cb(self, msg):
        if self._bridge:
            try:
                self._latest_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception:
                pass

    # ── Gazebo物体操作 ───────────────────────────────────────────────────────

    def _get_pose(self, name: str) -> np.ndarray:
        """从model_states获取物体GT坐标(x,y,z)。"""
        if self._model_states is None:
            return np.zeros(3, dtype=np.float32)
        try:
            idx = self._model_states.name.index(name)
            p = self._model_states.pose[idx].position
            return np.array([p.x, p.y, p.z], dtype=np.float32)
        except ValueError:
            return np.zeros(3, dtype=np.float32)

    def _teleport(self, name: str, x: float, y: float, z: float) -> bool:
        state = ModelState()
        state.model_name = name
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        state.pose.orientation.w = 1.0
        state.reference_frame = "world"
        try:
            resp = self._set_model_state(state)
            if resp is not None and hasattr(resp, "success") and not resp.success:
                msg = getattr(resp, "status_message", "")
                rospy.logwarn(f"[Env] SetModelState 拒绝 {name}: {msg}")
                return False
            return True
        except Exception as e:
            rospy.logwarn(f"[Env] Teleport {name} 失败: {e}")
            return False

    def _model_in_world(self, model_name: str) -> bool:
        if self._model_states is None:
            return False
        try:
            return model_name in self._model_states.name
        except Exception:
            return False

    def _delete_model_safe(self, model_name: str) -> bool:
        if self._delete_model_srv is None:
            return False
        try:
            self._delete_model_srv(model_name)
            return True
        except Exception as e:
            rospy.logwarn(f"[Env] delete_model {model_name}: {e}")
            return False

    def _spawn_model_safe(self, model_name: str, model_xml: str,
                          pose: Pose) -> bool:
        if self._spawn_model_srv is None:
            return False
        try:
            self._spawn_model_srv(
                model_name=model_name,
                model_xml=model_xml,
                robot_namespace="",
                initial_pose=pose,
                reference_frame="world",
            )
            return True
        except Exception as e:
            rospy.logwarn(f"[Env] spawn_sdf_model {model_name}: {e}")
            return False

    def _sync_gazebo_models_delete_or_spawn(self) -> None:
        """
        未激活颜色：从仿真中删除模型（减少实体与碰撞）。
        激活但缺失（上轮被删）：从 pick_place_scene.world 提取 SDF 并 spawn。
        若 Gazebo 服务不可用，回退为远端 _teleport 停放（与旧版兼容）。
        """
        # ── 每轮先清空所有颜色模型，再只生成本轮激活 3+3 ───────────────────
        if self._delete_model_srv is not None:
            for c in WORLD_SPAWN_COLORS:
                self._delete_model_safe(block_name(c))
                self._delete_model_safe(bin_name(c))
            # 等待一次 model_states 刷新，避免后续“刚删完又判定存在”
            rospy.sleep(0.1)
        else:
            # 无删除服务时回退到远端停放
            park_i = 0
            for c in WORLD_SPAWN_COLORS:
                for nm, zc in ((block_name(c), BLOCK_H / 2.0),
                               (bin_name(c), BIN_H / 2.0)):
                    self._teleport(nm, SPAWN_PARK_X, SPAWN_PARK_Y0 + park_i * 0.2, zc)
                    park_i += 1

        # ── 仅为本回合需要的方块/桶补齐模型（3+2 时某色可能仅有方块无桶） ──
        spawn_k = 0
        for c in self._active_block_colors:
            nm, zc = block_name(c), BLOCK_H / 2.0
            xml = model_xml_for_spawn(nm)
            if xml is None:
                rospy.logerr(f"[Env] 无法为 {nm} 生成 spawn XML，请检查 world 文件")
                continue
            pose = Pose()
            pose.position.x = SPAWN_PARK_X + spawn_k * 0.06
            pose.position.y = SPAWN_PARK_Y0 + spawn_k * 0.14
            pose.position.z = zc
            pose.orientation.w = 1.0
            if self._spawn_model_srv is not None:
                if self._spawn_model_safe(nm, xml, pose):
                    spawn_k += 1
                else:
                    self._teleport(nm, pose.position.x, pose.position.y, zc)
                    spawn_k += 1
            else:
                self._teleport(nm, pose.position.x, pose.position.y, zc)
                spawn_k += 1
        for c in self._active_bin_colors:
            nm, zc = bin_name(c), BIN_H / 2.0
            xml = model_xml_for_spawn(nm)
            if xml is None:
                rospy.logerr(f"[Env] 无法为 {nm} 生成 spawn XML，请检查 world 文件")
                continue
            pose = Pose()
            pose.position.x = SPAWN_PARK_X + spawn_k * 0.06
            pose.position.y = SPAWN_PARK_Y0 + spawn_k * 0.14
            pose.position.z = zc
            pose.orientation.w = 1.0
            if self._spawn_model_srv is not None:
                if self._spawn_model_safe(nm, xml, pose):
                    spawn_k += 1
                else:
                    self._teleport(nm, pose.position.x, pose.position.y, zc)
                    spawn_k += 1
            else:
                self._teleport(nm, pose.position.x, pose.position.y, zc)
                spawn_k += 1
        rospy.sleep(0.1)

    def _deterministic_grid(self, n: int, zone_x: tuple,
                            zone_y: tuple, gap: float,
                            placement_valid: bool = False) -> list:
        cols = max(1, int((zone_x[1] - zone_x[0]) / gap))
        rows = max(1, int((zone_y[1] - zone_y[0]) / gap))
        pts  = []
        def _cell_ok(cx: float, cy: float) -> bool:
            if placement_valid:
                return self._is_valid_placement_xy(cx, cy)
            return self._is_reachable_xy(cx, cy)
        for row in range(rows):
            for col in range(cols):
                cx = zone_x[0] + gap * (col + 0.5)
                cy = zone_y[0] + gap * (row + 0.5)
                if (
                    cx <= zone_x[1]
                    and cy <= zone_y[1]
                    and _cell_ok(cx, cy)
                ):
                    pts.append((cx, cy))
                if len(pts) >= n:
                    return pts
        return pts[:n]

    def _is_reachable_xy(self, x: float, y: float) -> bool:
        r = float(np.hypot(x - ARM_BASE_X, y - ARM_BASE_Y))
        return (ARM_REACH_MIN_R <= r <= ARM_REACH_MAX_R)

    def _project_to_reachable_xy(self, x: float, y: float) -> np.ndarray:
        dx = float(x - ARM_BASE_X)
        dy = float(y - ARM_BASE_Y)
        r = float(np.hypot(dx, dy))
        if r < 1e-9:
            return np.array([ARM_BASE_X + ARM_REACH_MIN_R, ARM_BASE_Y], dtype=np.float32)
        rr = min(max(r, ARM_REACH_MIN_R), ARM_REACH_MAX_R)
        s = rr / r
        return np.array([ARM_BASE_X + dx * s, ARM_BASE_Y + dy * s], dtype=np.float32)

    def _is_clear_of_robot_base(self, x: float, y: float) -> bool:
        """物体中心不得进入机械臂底座碰撞圆（与 ARM_BASE 可达环是两套约束）。"""
        bx, by = ROBOT_BASE_EXCLUSION_XY
        r = float(np.hypot(x - bx, y - by))
        return r >= float(ROBOT_BASE_EXCLUSION_RADIUS) - 1e-9

    def _is_valid_placement_xy(self, x: float, y: float) -> bool:
        """可达环内 + 不在底座碰撞圆内 + 在作业矩形内（双保险）。"""
        if not self._is_reachable_xy(x, y):
            return False
        if not self._is_clear_of_robot_base(x, y):
            return False
        if not (PLACE_RECT_X[0] <= x <= PLACE_RECT_X[1]
                and PLACE_RECT_Y[0] <= y <= PLACE_RECT_Y[1]):
            return False
        return True

    def _build_rect_placement_candidates(self) -> List[Tuple[float, float]]:
        """在 PLACE_RECT 内细网格上生成候选点，仅保留 _is_valid_placement_xy。"""
        pts: List[Tuple[float, float]] = []
        cell = max(0.035, min(MIN_OBJECT_CENTER_GAP * 0.42,
                               (PLACE_RECT_X[1] - PLACE_RECT_X[0]) / 8.0,
                               (PLACE_RECT_Y[1] - PLACE_RECT_Y[0]) / 6.0))
        rx, ry = PLACE_RECT_X, PLACE_RECT_Y
        cols = max(1, int((rx[1] - rx[0]) / cell))
        rows = max(1, int((ry[1] - ry[0]) / cell))
        for row in range(rows):
            for col in range(cols):
                cx = rx[0] + cell * (col + 0.5)
                cy = ry[0] + cell * (row + 0.5)
                if cx <= rx[1] and cy <= ry[1]:
                    if self._is_valid_placement_xy(float(cx), float(cy)):
                        pts.append((float(cx), float(cy)))
        return pts

    def _sample_rect_xy(self, rng: np.random.Generator) -> Tuple[float, float]:
        """在 PLACE_RECT 内均匀随机，直到满足有效放置约束。"""
        for _ in range(120):
            x = float(rng.uniform(*PLACE_RECT_X))
            y = float(rng.uniform(*PLACE_RECT_Y))
            if self._is_valid_placement_xy(x, y):
                return x, y
        # 退化：矩形几何中心附近（仍做校验）
        cx = 0.5 * (PLACE_RECT_X[0] + PLACE_RECT_X[1])
        cy = 0.5 * (PLACE_RECT_Y[0] + PLACE_RECT_Y[1])
        if self._is_valid_placement_xy(cx, cy):
            return cx, cy
        return cx, cy

    def _greedy_grid_centers(self, n_needed: int, gap: float,
                             rng: np.random.Generator) -> List[Tuple[float, float]]:
        """在 PLACE_RECT 栅格上贪心选取 n 个点，两两中心距 ≥ gap。"""
        candidates = self._build_rect_placement_candidates()
        if len(candidates) < max(8, n_needed * 3):
            rospy.logwarn(
                "[Env] 作业矩形内有效候选点偏少，请放宽 PLACE_RECT_* 或 "
                "ROBOT_BASE_EXCLUSION_* / 可达环")
        rng.shuffle(candidates)
        selected: List[Tuple[float, float]] = []
        for pt in candidates:
            if len(selected) >= n_needed:
                break
            x, y = float(pt[0]), float(pt[1])
            if all(
                np.hypot(x - sx, y - sy) >= gap - 1e-9
                for sx, sy in selected
            ):
                selected.append((x, y))
        return selected

    def _place_active_objects_unified(self, rng: np.random.Generator) -> None:
        """在 PLACE_RECT 作业矩形内摆放；优先矩形栅格+微抖动；永不进入底座碰撞圆。"""
        targets: List[Tuple[str, float]] = []
        for c in self._active_block_colors:
            targets.append((block_name(c), BLOCK_H / 2.0))
        for c in self._active_bin_colors:
            targets.append((bin_name(c), BIN_H / 2.0))

        n = len(targets)
        gap = float(MIN_OBJECT_CENTER_GAP)

        def _try_assign(centers: List[Tuple[float, float]]) -> bool:
            if len(centers) < n:
                return False
            centers = centers[:n]
            perm = list(range(n))
            rng.shuffle(perm)
            jitter = float(OBJECT_PLACE_JITTER)
            for k in range(n):
                t_idx = perm[k]
                nm, zc = targets[t_idx]
                cx, cy = centers[k]
                x, y = float(cx), float(cy)
                if jitter > 0:
                    for _ in range(24):
                        tx = cx + float(rng.uniform(-jitter, jitter))
                        ty = cy + float(rng.uniform(-jitter, jitter))
                        if self._is_valid_placement_xy(tx, ty):
                            x, y = tx, ty
                            break
                if not self._is_valid_placement_xy(x, y):
                    x, y = cx, cy
                self._teleport(nm, x, y, zc)
            return True

        centers = self._greedy_grid_centers(n, gap, rng)
        if _try_assign(centers):
            return

        rospy.logwarn(
            "[Env] 矩形网格摆放失败，回退到矩形内随机采样")

        placed_xy: List[Tuple[float, float]] = []
        for idx, (nm, zc) in enumerate(targets):
            ok = False
            for _ in range(400):
                x, y = self._sample_rect_xy(rng)
                if all(
                    np.hypot(x - px, y - py) >= gap
                    for px, py in placed_xy
                ):
                    placed_xy.append((x, y))
                    self._teleport(nm, x, y, zc)
                    ok = True
                    break
            if ok:
                continue
            grid = self._deterministic_grid(
                n, PLACE_RECT_X, PLACE_RECT_Y, gap, placement_valid=True)
            if idx < len(grid):
                x, y = grid[idx]
                placed_xy.append((x, y))
                self._teleport(nm, x, y, zc)
                rospy.logwarn(
                    f"[Env] {nm} 随机摆放失败，使用 fallback 网格 ({x:.3f},{y:.3f})")
            else:
                rospy.logerr(
                    "[Env] 物体无法摆放：请放宽 PLACE_RECT_* 或 "
                    "MIN_OBJECT_CENTER_GAP / 物体数量")

    def _randomize_scene(self):
        """
        随机化场景：
          1. 按 curriculum_mode 抽取方块/桶颜色子集与任务；
          2. 未激活模型从 Gazebo 删除；缺失模型按 world 模板 spawn；
          3. 在 PLACE_RECT_* 矩形内摆放（网格优先 + 微抖动），中心距 ≥ MIN_OBJECT_CENTER_GAP，
             且永不进入 ROBOT_BASE_EXCLUSION 圆。
        """
        rng = np.random.default_rng()

        all_colors = list(self.color_cfg.colors)
        rng.shuffle(all_colors)

        if self.curriculum_mode == "2+2":
            self._active_block_colors = all_colors[:2]
            self._active_bin_colors = list(self._active_block_colors)
            rng.shuffle(self._active_block_colors)
            rng.shuffle(self._active_bin_colors)
        else:
            # 3+2：3 个方块，桶为其中 2 色（第 3 色为干扰块，无对应桶）
            self._active_block_colors = all_colors[:3]
            rng.shuffle(self._active_block_colors)
            pair = rng.choice(
                list(itertools.combinations(self._active_block_colors, 2)))
            self._active_bin_colors = list(pair)
            rng.shuffle(self._active_bin_colors)

        self.pick_color = rng.choice(self._active_block_colors)
        self.place_color = rng.choice(self._active_bin_colors)

        self._sync_gazebo_models_delete_or_spawn()
        self._place_active_objects_unified(rng)
        # 让 MoveIt 仅看到本轮激活物体，且位姿与 Gazebo 一致
        self._sync_moveit_planning_scene()

    # ── 带噪声的位置获取 ─────────────────────────────────────────────────────

    def _refresh_positions(self):
        """从 Gazebo 刷新方块/桶的平面位置缓存（无噪声，与 model_states 一致）。

        观测中的位置噪声仅在 _build_observation 构造向量时叠加；step/奖励/运动执行
        必须使用此处 GT，否则会与 Gazebo 中物体中心不一致导致对不准。
        """
        for color in self._active_block_colors:
            xyz = self._get_pose(block_name(color))
            self._block_positions[color] = np.array(xyz[:2], dtype=np.float32)
        for color in self._active_bin_colors:
            xyz = self._get_pose(bin_name(color))
            self._bin_positions[color] = np.array(xyz[:2], dtype=np.float32)

    def _get_block_pos(self, color: str) -> np.ndarray:
        return self._block_positions.get(
            color, np.zeros(2, dtype=np.float32))

    def _get_bin_pos(self, color: str) -> np.ndarray:
        return self._bin_positions.get(
            color, np.zeros(2, dtype=np.float32))

    # ── Observation构建 ──────────────────────────────────────────────────────

    def _build_crops(self, block_xy_slots: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        提取方块槽位对应的图像 crop（SLOT_COUNT 行；空槽为黑图）。
        block_xy_slots：每槽用于取图的平面坐标（通常为 GT+噪声，与 obs 中 block_pos 一致）；
        为 None 时退回 GT+噪声（与旧行为一致）。
        返回形状 (SLOT_COUNT, 3, 28, 28)
        """
        import cv2
        crops = []
        img = self._latest_image

        for slot in range(SLOT_COUNT):
            if slot < len(self._active_block_colors):
                color = self._active_block_colors[slot]
                if block_xy_slots is not None:
                    block_pos = block_xy_slots[slot]
                else:
                    block_pos = self._get_block_pos(color) + np.random.normal(
                        0, self.noise_sigma, 2).astype(np.float32)
                if img is not None:
                    u, v = self._robot_to_pixel(block_pos)
                    h, w = img.shape[:2]
                    u, v = int(np.clip(u, 14, w-14)), int(np.clip(v, 14, h-14))
                    crop = img[v-14:v+14, u-14:u+14]
                    if crop.shape[:2] != (28, 28):
                        crop = np.zeros((28, 28, 3), dtype=np.uint8)
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop = crop.astype(np.float32) / 255.0
                else:
                    crop = np.zeros((28, 28, 3), dtype=np.float32)
            else:
                crop = np.zeros((28, 28, 3), dtype=np.float32)
            crops.append(crop.transpose(2, 0, 1))

        return np.array(crops, dtype=np.float32)  # (SLOT_COUNT, 3, 28, 28)

    def _robot_to_pixel(self, xy: np.ndarray) -> Tuple[float, float]:
        """机械臂坐标 → 像素坐标（需要标定值）。"""
        from perception.camera_perception import CameraPerception
        # 用默认标定值，真机部署时会被替换
        kx, bx = -0.00029, 0.31084
        ky, by  =  0.00030, 0.09080
        v = (xy[0] - bx) / kx if abs(kx) > 1e-9 else 240
        u = (xy[1] - by) / ky if abs(ky) > 1e-9 else 320
        return float(u), float(v)

    def _build_observation(self) -> np.ndarray:
        """
        构建完整 observation 向量（固定 SLOT_COUNT 槽位，未满填零）。
        布局：[crops | block_pos | bin_pos | gripper | task]
        """
        self._refresh_positions()

        # 观测：GT + 高斯噪声（仅用于策略输入；crops 与 block_pos 向量共用同一组噪声）
        noise2 = lambda: np.random.normal(0, self.noise_sigma, 2).astype(np.float32)
        bp_list: List[np.ndarray] = []
        for slot in range(SLOT_COUNT):
            if slot < len(self._active_block_colors):
                c = self._active_block_colors[slot]
                bp_list.append(self._get_block_pos(c) + noise2())
            else:
                bp_list.append(np.zeros(2, dtype=np.float32))

        bn_list: List[np.ndarray] = []
        for slot in range(SLOT_COUNT):
            if slot < len(self._active_bin_colors):
                c = self._active_bin_colors[slot]
                bn_list.append(self._get_bin_pos(c) + noise2())
            else:
                bn_list.append(np.zeros(2, dtype=np.float32))

        crops = self._build_crops(block_xy_slots=bp_list)   # (SLOT_COUNT, 3, 28, 28)
        block_pos = np.array(bp_list, dtype=np.float32).flatten()
        bin_pos = np.array(bn_list, dtype=np.float32).flatten()

        gripper = np.array(
            [0.0 if self._gripper_open else 1.0], dtype=np.float32)

        pick_local  = float(self._active_block_colors.index(self.pick_color))
        place_local = float(self._active_bin_colors.index(self.place_color))
        task = np.array([pick_local, place_local], dtype=np.float32)

        obs = np.concatenate([
            crops.flatten(), block_pos, bin_pos, gripper, task,
        ]).astype(np.float32)

        assert obs.shape == (self.obs_dim,), \
            f"obs shape {obs.shape} != expected {(self.obs_dim,)}"

        return obs

    # ── MoveIt动作原语 ──────────────────────────────────────────────────────

    def _open_gripper(self):
        self._moveit_gripper.set_named_target(MOVEIT_GRIPPER_OPEN_STATE)
        self._moveit_gripper.go(wait=True)
        self._gripper_open = True
        time.sleep(0.3)

    def _close_gripper(self):
        """夹持方块：使用 SRDF `middle`，非 `close`（完全闭合无法容纳方块宽度）。"""
        self._moveit_gripper.set_named_target(MOVEIT_GRIPPER_GRASP_STATE)
        self._moveit_gripper.go(wait=True)
        self._gripper_open = False
        time.sleep(0.3)

    def _pose_for_yaw(self, yaw: float) -> tuple:
        """纯绕世界 Z 轴旋转的四元数（侧向夹，末端仍朝上）。"""
        half = float(yaw) * 0.5
        return (0.0, 0.0, math.sin(half), math.cos(half))

    def _pose_horizontal(self, yaw: float) -> tuple:
        """
        水平侧向抓取四元数：末端 TCP 平行于地面，朝向方块方向（由 yaw 决定）。

        构造方式（四元数乘法）：
          q_total = q_yaw(绕Z) * q_y90(绕Y+90°)
          先绕 Y 转 90°（末端从朝上变为朝前），再绕 Z 转 yaw（对准方块方向）。

        这与图3所示的理想抓取姿态一致：夹爪从侧面水平插入方块。
        """
        # q_y90: 绕 Y 轴 +90°
        qy_x, qy_y, qy_z, qy_w = TCP_HORIZONTAL_BASE_QUAT  # (0, 0.7071, 0, 0.7071)

        # q_yaw: 绕 Z 轴 yaw
        half_yaw = float(yaw) * 0.5
        qz_x, qz_y, qz_z, qz_w = 0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)

        # q_total = q_yaw * q_y90（先 y90 再 yaw，右乘顺序）
        # 四元数乘法：(a1×a2)，其中 a=(x,y,z,w)
        rx = qz_w*qy_x + qz_x*qy_w + qz_y*qy_z - qz_z*qy_y
        ry = qz_w*qy_y - qz_x*qy_z + qz_y*qy_w + qz_z*qy_x
        rz = qz_w*qy_z + qz_x*qy_y - qz_y*qy_x + qz_z*qy_w
        rw = qz_w*qy_w - qz_x*qy_x - qz_y*qy_y - qz_z*qy_z
        return (rx, ry, rz, rw)

    def _pose_horizontal_lift_assist(self, yaw: float) -> tuple:
        """水平侧向姿态上叠加绕末端局部 X 轴的小 pitch（q_pitch * q_horizontal），利于抬高方块 COM z。"""
        half = float(PICK_LIFT_ASSIST_PITCH_RAD) * 0.5
        q_p = (math.sin(half), 0.0, 0.0, math.cos(half))
        q_h = self._pose_horizontal(yaw)
        return _quat_multiply(q_p, q_h)

    def _move_to_xy(self, x: float, y: float, z: float, yaw: float = 0.0,
                    ignore_block_color: str = None,
                    orientation_mode: str = "horizontal") -> bool:
        """
        规划并执行到目标位姿（末端笛卡尔）。

        orientation_mode:
          - "horizontal"（默认）：水平侧向抓取，末端平行桌面，朝向由 yaw 决定。
            这是与实物验证一致的稳定姿态（参照图3），消除了多候选轮询导致的乱动。
            规划失败时只做一次容差放宽重试，不换姿态。
          - "horizontal_lift_assist"：在 horizontal 基础上叠加小 pitch（抓取后抬腕辅助）。
          - "yaw"：纯绕世界 Z 的 yaw（末端仍朝上，用于放置时的垂直下放）。

        避障：通过 MoveIt PlanningScene 中的桌面 + 方块 + 桶碰撞体自动避障。
        """
        self._sync_moveit_planning_scene(ignore_block_color=ignore_block_color)
        self._moveit_arm.set_start_state_to_current_state()
        ee_link = self._moveit_arm.get_end_effector_link()

        # ── 计算目标姿态四元数 ────────────────────────────────────────────
        if orientation_mode == "horizontal":
            qx, qy, qz, qw = self._pose_horizontal(yaw)
        elif orientation_mode == "horizontal_lift_assist":
            qx, qy, qz, qw = self._pose_horizontal_lift_assist(yaw)
        else:
            qx, qy, qz, qw = self._pose_for_yaw(yaw)

        def _try_plan_execute(tx: float, ty: float, tz: float,
                               tqx: float, tqy: float, tqz: float, tqw: float,
                               label: str) -> bool:
            target = PoseStamped()
            target.header.frame_id = "world"
            target.pose.position.x = float(tx)
            target.pose.position.y = float(ty)
            target.pose.position.z = float(tz)
            target.pose.orientation.x = tqx
            target.pose.orientation.y = tqy
            target.pose.orientation.z = tqz
            target.pose.orientation.w = tqw
            self._moveit_arm.clear_pose_targets()
            self._moveit_arm.set_pose_target(target, ee_link)
            plan_ok, traj, plan_time, err = self._moveit_arm.plan()
            if not plan_ok:
                rospy.logwarn(
                    f"[Env] plan 失败 [{label}] pos=({tx:.3f},{ty:.3f},{tz:.3f}) "
                    f"time={plan_time:.2f}s err={err}")
                self._moveit_arm.clear_pose_targets()
                return False
            try:
                exec_result = self._moveit_arm.execute(traj, wait=True)
                self._moveit_arm.clear_pose_targets()
                time.sleep(0.2)
                ok_exec = exec_result if isinstance(exec_result, bool) else True
                if not ok_exec:
                    rospy.logwarn(f"[Env] execute 返回 False [{label}]")
                return ok_exec
            except Exception as e:
                rospy.logwarn(f"[Env] execute 异常 [{label}]: {e}")
                try:
                    self._moveit_arm.stop()
                    self._moveit_arm.clear_pose_targets()
                except Exception:
                    pass
                return False

        # ── 第一次尝试：精确目标 z ────────────────────────────────────────
        if _try_plan_execute(x, y, z, qx, qy, qz, qw, "primary"):
            return True

        # ── 第二次尝试：放宽姿态容差后重试，位置不变 ──────────────────────
        # 只放宽方向容差，不换姿态类型，保持末端动作一致性
        orig_pos_tol  = self._moveit_arm.get_goal_position_tolerance()
        orig_ort_tol  = self._moveit_arm.get_goal_orientation_tolerance()
        self._moveit_arm.set_goal_position_tolerance(0.022)
        self._moveit_arm.set_goal_orientation_tolerance(0.20)
        result = _try_plan_execute(x, y, z, qx, qy, qz, qw, "relaxed_tol")
        self._moveit_arm.set_goal_position_tolerance(orig_pos_tol)
        self._moveit_arm.set_goal_orientation_tolerance(orig_ort_tol)

        if not result:
            rospy.logwarn(
                f"[Env] _move_to_xy 两次尝试均失败 "
                f"pos=({x:.3f},{y:.3f},{z:.3f}) mode={orientation_mode}")
        return result

    def _attach_held_block_to_scene(self, color: str):
        if self._planning_scene is None or self._moveit_arm is None:
            return
        try:
            ee = self._moveit_arm.get_end_effector_link()
            p = PoseStamped()
            p.header.frame_id = ee
            p.pose.position.x = 0.0
            p.pose.position.y = 0.0
            p.pose.position.z = 0.0
            p.pose.orientation.w = 1.0
            self._planning_scene.attach_box(ee, block_name(color), p, (0.05, 0.05, 0.05))
        except Exception as e:
            rospy.logwarn(f"[Env] attach_box 失败 {color}: {e}")

    def _detach_held_block_from_scene(self, color: str):
        if self._planning_scene is None or self._moveit_arm is None:
            return
        try:
            ee = self._moveit_arm.get_end_effector_link()
            self._planning_scene.remove_attached_object(ee, block_name(color))
        except Exception as e:
            rospy.logwarn(f"[Env] detach_box 失败 {color}: {e}")

    def _grasp_tcp_xy_from_block_center(self, bx: float, by: float) -> Tuple[float, float]:
        """将 IK 目标从方块中心沿「基座 → 方块」方向收回 TCP_GRASP_XY_BACKOFF_M，使两指中间更对准方块中心。

        Gazebo 中方块 pose 一般为几何中心；MoveIt 的 ee_link 常在指尖侧，直接以中心为目标易出现
        「伸得过深、方块卡在指尖/连杆下」。若仍偏，请调 TCP_GRASP_XY_BACKOFF_M 或检查 URDF。"""
        b = float(TCP_GRASP_XY_BACKOFF_M)
        if b <= 0:
            return bx, by
        dx = float(bx - ARM_BASE_X)
        dy = float(by - ARM_BASE_Y)
        r = float(np.hypot(dx, dy))
        if r < 1e-6:
            return bx, by
        ux, uy = dx / r, dy / r
        return float(bx - b * ux), float(by - b * uy)

    def _execute_pick(self, x: float, y: float, color: str, pose_id: int) -> bool:
        """
        完整 pick 原语：开爪 → 接近 → 预抓取位 → 落爪 → 关爪 → 抬起 → 物理验证。

        抓取姿态（水平侧向，与图3一致）：
          yaw = atan2(block_y - base_y, block_x - base_x)
          末端从底座方向水平朝向方块，平行地面插入，稳定不乱动。

        三步下降（避免从高空直线扫过方块）：
          1. APPROACH_H：安全高度接近（高于桶与方块，当前 TABLE_Z+APPROACH_H）
          2. pre_z：方块顶面上方 PRE_GRASP_CLEAR_Z
          3. GRASP_H：抓取位（方块中部）
        """
        # TCP 反向收一点，使两指中心对准方块中心
        gx, gy = self._grasp_tcp_xy_from_block_center(x, y)

        # yaw = 从底座指向方块的水平方向角
        # 这使夹爪始终朝着方块方向水平伸入，姿态稳定
        yaw = float(math.atan2(gy - ARM_BASE_Y, gx - ARM_BASE_X))

        self._open_gripper()

        # 1. 安全高度接近（水平侧向姿态）
        if not self._move_to_xy(
            gx, gy, TABLE_Z + APPROACH_H, yaw,
            ignore_block_color=color, orientation_mode="horizontal",
        ):
            return False

        # 2. 降到方块顶面上方（避免从高处直线扫入横推方块）
        pre_z = TABLE_Z + BLOCK_H + PRE_GRASP_CLEAR_Z
        if not self._move_to_xy(
            gx, gy, pre_z, yaw,
            ignore_block_color=color, orientation_mode="horizontal",
        ):
            return False

        # 3. 落到抓取高度（方块中部）
        if not self._move_to_xy(
            gx, gy, TABLE_Z + GRASP_H, yaw,
            ignore_block_color=color, orientation_mode="horizontal",
        ):
            return False

        self._close_gripper()

        # 4. 抬起到安全高度（持 middle 夹持，不在此处开爪）
        lift_ok = self._move_to_xy(
            gx, gy, TABLE_Z + APPROACH_H, yaw,
            ignore_block_color=color, orientation_mode="horizontal",
        )
        block_z = float(self._get_pose(block_name(color))[2])

        # 方块 COM 仍低于 14cm（world z）时：同高度下做一次抬腕辅助，抬高方块 z 以利通过 PICK_LIFT_VERIFY_Z
        if block_z < PICK_LIFT_ASSIST_Z:
            rospy.loginfo(
                f"[Env] 抓取后方块 z={block_z:.4f} < PICK_LIFT_ASSIST_Z={PICK_LIFT_ASSIST_Z:.4f}，"
                f"尝试抬腕辅助 (pitch={PICK_LIFT_ASSIST_PITCH_RAD:.3f} rad)")
            assist_ok = self._move_to_xy(
                gx, gy, TABLE_Z + APPROACH_H, yaw,
                ignore_block_color=color, orientation_mode="horizontal_lift_assist",
            )
            if not assist_ok:
                rospy.logwarn("[Env] 抬腕辅助规划/执行失败，沿用抬升后姿态继续验证")
            block_z = float(self._get_pose(block_name(color))[2])

        # MoveIt execute 可能误报 False，但仿真里臂已到位；若高度已达标则继续持块→attach→后续 place 再开爪
        if block_z < PICK_LIFT_VERIFY_Z:
            rospy.logwarn(
                f"[Env] 抓取验证失败 {color}_block: z={block_z:.4f} "
                f"< PICK_LIFT_VERIFY_Z={PICK_LIFT_VERIFY_Z:.4f}（未离地），开爪放弃")
            self._open_gripper()
            self._holding_color = None
            return False

        if not lift_ok:
            rospy.logwarn(
                "[Env] 抬升路径曾返回 False，但方块高度已达标，保持夹持并继续（middle）")

        self._holding_color = color
        self._attach_held_block_to_scene(color)
        return True

    def _execute_place(self, x: float, y: float, pose_id: int) -> bool:
        """
        完整 place 原语：接近桶上方 → 下放到桶口内 → 开爪 → 抬起。
        放置时 yaw 对准桶方向，保持水平侧向姿态与 pick 一致。
        """
        held = self._holding_color
        # 放置时 yaw 对准桶（从底座方向朝向桶）
        yaw = float(math.atan2(y - ARM_BASE_Y, x - ARM_BASE_X))

        if not self._move_to_xy(
            x, y, TABLE_Z + APPROACH_H, yaw, orientation_mode="horizontal"
        ):
            return False
        if not self._move_to_xy(
            x, y, TABLE_Z + PLACE_H, yaw, orientation_mode="horizontal"
        ):
            return False
        self._open_gripper()
        if held:
            self._detach_held_block_from_scene(held)
        self._holding_color = None
        # 放完后抬起（失败不影响放置结果，忽略返回值）
        self._move_to_xy(x, y, TABLE_Z + APPROACH_H, yaw, orientation_mode="horizontal")
        return True

    def _detach_all_attached_blocks_from_scene(self):
        """reset/碰撞恢复前移除 PlanningScene 中可能挂在本体上的方块附着，避免规划器状态异常。"""
        if self._planning_scene is None or self._moveit_arm is None:
            return
        try:
            ee = self._moveit_arm.get_end_effector_link()
        except Exception:
            return
        for c in WORLD_SPAWN_COLORS:
            try:
                self._planning_scene.remove_attached_object(ee, block_name(c))
            except Exception:
                pass

    def _gazebo_reset_simulation(self) -> bool:
        """调用 Gazebo reset_simulation：重置物理与时间，机械臂回到 spawn 姿态（随后 _randomize_scene 仍会摆物）。"""
        try:
            rospy.wait_for_service("/gazebo/reset_simulation", timeout=3.0)
            rospy.ServiceProxy("/gazebo/reset_simulation", Empty)()
            rospy.loginfo("[Env] 已调用 /gazebo/reset_simulation")
            time.sleep(0.4)
            return True
        except Exception as e:
            rospy.logwarn(f"[Env] /gazebo/reset_simulation 不可用或失败: {e}")
            return False

    def _try_moveit_arm_home(self) -> bool:
        """多种方式尝试回到 SRDF named state `home`（碰撞后单次 go() 常失败）。"""
        arm = self._moveit_arm
        if arm is None:
            return False
        try:
            arm.set_start_state_to_current_state()
        except Exception:
            pass
        try:
            arm.clear_pose_targets()
            arm.set_named_target(MOVEIT_ARM_HOME_STATE)
            if arm.go(wait=True):
                return True
        except Exception as e:
            rospy.logwarn(f"[Env] MoveIt set_named_target+go(home) 失败: {e}")
        try:
            arm.clear_pose_targets()
            jv = arm.get_named_target_values(MOVEIT_ARM_HOME_STATE)
            arm.set_joint_value_target(jv)
            if arm.go(wait=True):
                return True
        except Exception as e:
            rospy.logwarn(f"[Env] MoveIt get_named_target_values+go(home) 失败: {e}")
        try:
            arm.clear_pose_targets()
            orig_pt = arm.get_goal_position_tolerance()
            orig_ot = arm.get_goal_orientation_tolerance()
            arm.set_goal_position_tolerance(0.08)
            arm.set_goal_orientation_tolerance(0.8)
            arm.set_named_target(MOVEIT_ARM_HOME_STATE)
            plan_ok, traj, _, err = arm.plan()
            if plan_ok and traj is not None:
                ex = arm.execute(traj, wait=True)
                ok_exec = ex if isinstance(ex, bool) else True
                arm.set_goal_position_tolerance(orig_pt)
                arm.set_goal_orientation_tolerance(orig_ot)
                arm.clear_pose_targets()
                if ok_exec:
                    return True
            arm.set_goal_position_tolerance(orig_pt)
            arm.set_goal_orientation_tolerance(orig_ot)
        except Exception as e:
            rospy.logwarn(f"[Env] MoveIt plan+execute(home) 失败: {e}")
        try:
            arm.clear_pose_targets()
        except Exception:
            pass
        return False

    def _return_home(self):
        """
        episode reset / close 时回 home：先清附着、停轨迹、再 MoveIt；
        仍失败且允许时调用 Gazebo reset_simulation，避免长时间训练因一次碰撞无法继续。
        """
        self._last_home_ok = True
        if not self._ros_initialized:
            self._init_ros()
        if self._moveit_arm is None:
            self._last_home_ok = False
            return

        self._detach_all_attached_blocks_from_scene()
        try:
            self._moveit_arm.stop()
            self._moveit_arm.clear_pose_targets()
        except Exception:
            pass
        try:
            if self._moveit_gripper is not None:
                self._moveit_gripper.stop()
                self._moveit_gripper.clear_pose_targets()
        except Exception:
            pass
        try:
            if self._moveit_gripper is not None:
                self._moveit_gripper.set_named_target(MOVEIT_GRIPPER_OPEN_STATE)
                self._moveit_gripper.go(wait=True)
                self._gripper_open = True
        except Exception as e:
            rospy.logwarn(f"[Env] reset 开爪失败（可忽略）: {e}")

        ok = self._try_moveit_arm_home()
        if not ok and env_config.gazebo_reset_simulation_on_home_fail():
            rospy.logwarn(
                "[Env] MoveIt 回 home 失败，尝试 Gazebo reset_simulation 后重试（"
                "物体位置将在 _randomize_scene 中重新传送）")
            if self._gazebo_reset_simulation():
                try:
                    rospy.ServiceProxy("/gazebo/unpause_physics", Empty)()
                except Exception:
                    pass
                time.sleep(0.35)
                ok = self._try_moveit_arm_home()

        if not ok:
            rospy.logwarn(
                "[Env] 回 home 仍失败；本 episode 可能仍异常。可检查碰撞或重启 Gazebo；"
                "或设置 EXPLORELLM_GAZEBO_RESET_SIMULATION_ON_HOME_FAIL=1（默认已开启）")
            self._last_home_ok = False
        else:
            try:
                self._open_gripper()
            except Exception as e:
                rospy.logwarn(f"[Env] home 后开爪失败: {e}")

    # ── 奖励函数 ──────────────────────────────────────────────────────────────

    def _compute_reward(self, primitive: int, color: str,
                        action_xy: np.ndarray, success: bool) -> float:
        """
        奖励函数（重写版）。

        完整成功链 = 选对颜色方块 → 执行+抓取验证通过 → 放入正确桶 → 任务完成。
        每个环节都有对应的奖励/惩罚，让 policy 清楚"差在哪一步"。

        奖励结构
        ──────────────────────────────────────────────────────────────────────
        PICK 原语 (primitive == 0):
          -dist(action_xy, pick_color 方块位置)   密集距离 shaping（始终对目标颜色算）
          -0.5   color != pick_color              选了错误颜色方块，明确惩罚
          +0.5   success and color==pick_color    正确颜色且物理验证通过
          -0.2   not success                      运动/抓取失败

        PLACE 原语 (primitive == 1):
          -dist(action_xy, place_color 桶位置)    密集距离 shaping（始终对目标桶算）
          -0.5   color != place_color             朝错误桶方向运动，惩罚
          +2.0   success and 方块进入目标桶        放置成功
          -0.2   not success                      运动失败
          +3.0   _check_done()                    任务完成 terminal bonus（独立叠加）
        ──────────────────────────────────────────────────────────────────────
        """
        r = 0.0

        if primitive == 0:  # ── PICK ──────────────────────────────────────
            # 距离 shaping：action 相对目标方块（始终用任务指定颜色）
            target_block_pos = self._get_block_pos(self.pick_color)
            dist = float(np.linalg.norm(action_xy - target_block_pos))
            r -= dist

            if color != self.pick_color:
                # 选了错误颜色的方块：明确惩罚（原来只是少拿 0.2，现在主动扣）
                r -= 0.5
            else:
                if success:
                    # 正确颜色 + 物理验证通过（方块真的被抬起）
                    r += 0.5
                else:
                    # 正确颜色但运动/抓取失败
                    r -= 0.2

        elif primitive == 1:  # ── PLACE ────────────────────────────────────
            # 距离 shaping：action 相对目标桶（始终用任务指定颜色）
            target_bin_pos = self._get_bin_pos(self.place_color)
            dist = float(np.linalg.norm(action_xy - target_bin_pos))
            r -= dist

            if color != self.place_color:
                # 朝错误颜色的桶方向运动：惩罚
                r -= 0.5

            if not success:
                r -= 0.2
            else:
                # 运动成功：检查方块是否真的进入目标桶内腔
                if self._is_block_in_target_bin():
                    r += 2.0   # 放置成功

            # Terminal bonus：任务完全完成时单独叠加（与中间 shaping 量级分开）
            if self._check_done():
                r += 3.0

        return float(r)

    def _is_block_in_target_bin(self) -> bool:
        """
        判断目标方块是否进入目标桶内腔。
        条件：
          1) 方块中心在桶内腔投影范围内（并扣除方块半宽，避免贴壁误判）；
          2) 方块中心 z 在桶口以下。
        """
        block_xyz = self._get_pose(block_name(self.pick_color))
        bin_xyz   = self._get_pose(bin_name(self.place_color))

        dx = float(abs(block_xyz[0] - bin_xyz[0]))
        dy = float(abs(block_xyz[1] - bin_xyz[1]))

        inner_half = BIN_INNER_W / 2.0
        block_half = BLOCK_W / 2.0
        margin_xy  = max(inner_half - block_half, 0.0)

        inside_xy = (dx <= margin_xy) and (dy <= margin_xy)
        below_rim = float(block_xyz[2]) < (TABLE_Z + BIN_H)
        return bool(inside_xy and below_rim)

    def _check_done(self) -> bool:
        """检查当前任务是否完成（目标方块确实在目标桶内腔）。"""
        return self._is_block_in_target_bin()

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_ros()
        self._step_count    = 0
        self._gripper_open  = True
        self._holding_color = None

        # 先回 home 再随机摆物：避免上一局结束时臂仍伸在桌面上方时，物体先被 teleport 进连杆包络导致弹飞/穿模
        self._return_home()
        time.sleep(1.0)
        # _randomize_scene 内部会重新抽取 _active_colors 和任务颜色
        self._randomize_scene()

        obs  = self._build_observation()
        info = {
            "pick_color":          self.pick_color,
            "place_color":         self.place_color,
            "active_colors":       list(self._active_colors),
            "active_block_colors": list(self._active_block_colors),
            "active_bin_colors":   list(self._active_bin_colors),
            "curriculum_mode":     self.curriculum_mode,
            "n_blocks":            self.n_blocks_ep,
            "n_bins":              self.n_bins_ep,
            "n_colors":            self.n_active,
            "reset_home_ok":       bool(self._last_home_ok),
        }
        return obs, info

    def step(self, action: np.ndarray):
        self._step_count += 1
        # 执行前用最新 Gazebo 位姿更新 GT，避免与观测/上步不同步导致对不准
        self._refresh_positions()

        # 解码动作
        primitive = int(np.round(np.clip(action[0], 0, 1)))
        max_obj = self.n_blocks_ep + self.n_bins_ep - 1
        obj_idx   = int(np.round(np.clip(action[1], 0, max_obj)))
        if len(action) >= 5:
            pose_id = int(np.round(np.clip(action[2], 0, self.pose_id_count - 1)))
            res_xy  = np.clip(action[3:5], -0.05, 0.05)
        else:
            # 兼容旧模型的4维动作：默认 pose_id=0
            pose_id = 0
            res_xy  = np.clip(action[2:4], -0.05, 0.05)

        # obj_idx 0..n_blocks-1 = 方块；n_blocks.. = 桶
        nb = self.n_blocks_ep
        if obj_idx < nb:
            color   = self._active_block_colors[obj_idx]
            base_xy = self._get_block_pos(color)
        else:
            color   = self._active_bin_colors[obj_idx - nb]
            base_xy = self._get_bin_pos(color)

        # 放置动作默认对准桶中心，保持“桶中心上方垂直下放”；
        # pick 才使用 residual 做微调搜索。
        if primitive == 1:
            target_xy = np.clip(
                base_xy,
                [OBJECT_ZONE_X[0], OBJECT_ZONE_Y[0]],
                [OBJECT_ZONE_X[1], OBJECT_ZONE_Y[1]],
            )
        else:
            target_xy = np.clip(
                base_xy + res_xy,
                [OBJECT_ZONE_X[0], OBJECT_ZONE_Y[0]],
                [OBJECT_ZONE_X[1], OBJECT_ZONE_Y[1]],
            )
        # 双保险：动作目标若越出可达域，投影回可达边界
        if not self._is_reachable_xy(float(target_xy[0]), float(target_xy[1])):
            target_xy = self._project_to_reachable_xy(
                float(target_xy[0]), float(target_xy[1]))

        # 执行动作原语
        if primitive == 0:
            success = self._execute_pick(
                target_xy[0], target_xy[1], color, pose_id)
        else:
            success = self._execute_place(
                target_xy[0], target_xy[1], pose_id)

        reward     = self._compute_reward(primitive, color, target_xy, success)
        terminated = self._check_done()
        truncated  = self._step_count >= self.max_steps

        obs = self._build_observation()
        info = {
            "primitive":     primitive,
            "color":         color,
            "obj_idx":       obj_idx,
            "pose_id":       pose_id,
            "target_xy":     target_xy.tolist(),
            "success":       success,
            "step":          self._step_count,
            "pick_color":    self.pick_color,
            "place_color":   self.place_color,
            "active_colors": list(self._active_colors),
            "active_block_colors": list(self._active_block_colors),
            "active_bin_colors":   list(self._active_bin_colors),
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._ros_initialized:
            try:
                self._return_home()
                moveit_commander.roscpp_shutdown()
            except Exception:
                pass
