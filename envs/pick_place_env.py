#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pick_place_env.py
====================
升级版环境，相比原版的核心变化：

  变化1：支持任意多种颜色（从 color_config 动态读取）
  变化2：垃圾桶/方块位置在 OBJECT_ZONE_* 内随机（贴近机械臂基座前方「前景区」，避免落在桌面最远端）
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
"""

import os
import sys
import time
import math
import itertools
from typing import Tuple, List

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

# 激活物体随机区域：桌面「前景区」——相对旧版明显左移（避免物体落在桌面最右侧远端），
# 与实物黑框矩形对齐：在机械臂可达环内、靠近基座前方，便于 IK / MoveIt 求解。
# 世界系：桌面中心约 (0.28,0)；矩形落在 [ARM_REACH_MIN_R, ARM_REACH_MAX_R] 环域内。
OBJECT_ZONE_X = (0.38, 0.53)
OBJECT_ZONE_Y = (-0.18, 0.18)
# 物体中心之间最小距离（米）；与实物「至少 10cm」一致，减少拥挤导致的规划失败
MIN_OBJECT_CENTER_GAP = 0.10
# 网格落位时在 x/y 上的一次性随机偏移（米），仅在 reset 传送时加一次，**不是**物理引擎每帧抖动。
# 默认 ±5mm：略打破完全规则的网格对齐；设为 0 则与「摆好后位置固定」一致。
OBJECT_PLACE_JITTER = 0.005

# 可达性约束（基于当前台面与机械臂安装关系的保守圆域）
ARM_BASE_X = 0.28
ARM_BASE_Y = 0.0
# 经验可达环域（保守）：过近/过远位置在该机型上更容易触发规划超时
ARM_REACH_MIN_R = 0.18
ARM_REACH_MAX_R = 0.33

# 首次 spawn 临时落点（远离桌面，避免与桌台重叠）
SPAWN_PARK_X = 2.0
SPAWN_PARK_Y0 = 0.0

# 高度参数（z轴，单位米）
# ──────────────────────────────────────────────────────────
# 坐标系约定（与 world 文件一致）：
#   桌面顶面 z = 0.00（world文件桌面顶面在z=0）
#   方块底面 z = 0.00，方块重心 z = BLOCK_H/2 = 0.02
#   桶底面   z = 0.00，桶重心   z = BIN_H/2   = 0.06
#
# GRASP_H 修正：
#   原来 TABLE_Z + GRASP_H = 0.00 + 0.005 = 0.005m
#   方块顶面在 z = TABLE_Z + BLOCK_H = 0.00 + 0.04 = 0.04m
#   末端只到 0.025m，在桌面里，触发碰撞检测 → CONTROL_FAILED
#   正确值：末端应到达方块中部稍上方
#   TABLE_Z + GRASP_H = TABLE_Z + BLOCK_H/2 + 0.005 = 0.025m
#   → GRASP_H = BLOCK_H/2 + 0.005 = 0.025m
#
# APPROACH_H 修正：
#   接近高度必须明显高于最高物体（桶高0.12m）加安全裕量
#   TABLE_Z + APPROACH_H = 0.18m（高于桶顶2cm，便于抬起后验证）
# ──────────────────────────────────────────────────────────
TABLE_Z       = 0.00   # 桌面顶面（world文件里桌面顶面在 z=0）
BLOCK_H       = 0.04   # 方块高度
BLOCK_W       = 0.05   # 方块边长（x/y）
BIN_H         = 0.12   # 垃圾桶高度（外高）
BIN_INNER_W   = 0.065  # 垃圾桶内腔宽度（x/y）
BIN_WALL_T    = 0.0025 # 壁厚/底厚（与world一致）

APPROACH_H    = 0.18   # 接近高度（末端 z = 0.18m）
# 抓取高度：世界系 z。方块重心约在 TABLE_Z+BLOCK_H/2；末端 TCP 与几何中心有偏差时，
# 过低的单一目标易在下降过程中挤压方块/刮桌面。采用略高于几何中心 + 分步下降。
GRASP_H       = 0.030  # 最终抓取末端 z（TABLE_Z + 该值），略高于原 0.025
# 方块顶面约 TABLE_Z+BLOCK_H，先降到顶面上方再最终下降，避免大跨度直线“扫”过方块
PRE_GRASP_CLEAR_Z = 0.02  # 在方块顶面上方预留的间隙（米），再落爪

PLACE_H       = 0.11   # 放置高度（末端 z = 0.11m，接近桶口内部）

# 图像crop参数
CROP_SIZE     = 28
POSITION_NOISE_SIGMA = 0.030   # 稍微降低噪声，6→3颜色后精度可以提高

# MoveIt SRDF 中的 group_state 名称（与 RViz MotionPlanning 一致，区分大小写）
MOVEIT_ARM_HOME_STATE = "home"
MOVEIT_GRIPPER_OPEN_STATE = "open"
MOVEIT_GRIPPER_CLOSE_STATE = "close"

# Gazebo模型名称约定：{color}_block, {color}_bin
def block_name(color: str) -> str: return f"{color}_block"
def bin_name(color: str)   -> str: return f"{color}_bin"


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

        self._moveit_arm.set_goal_position_tolerance(0.008)
        self._moveit_arm.set_goal_orientation_tolerance(0.08)
        self._moveit_arm.set_max_velocity_scaling_factor(0.4)
        self._moveit_arm.set_max_acceleration_scaling_factor(0.4)
        self._moveit_arm.allow_replanning(True)
        # 单次规划时间不宜过长；失败时由 _move_to_xy 内多策略重试（见下）
        self._moveit_arm.set_planning_time(5.0)
        self._moveit_arm.set_num_planning_attempts(12)
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
                self._planning_scene.add_box(block_name(c), p, (0.05, 0.05, 0.04))

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
                            zone_y: tuple, gap: float) -> list:
        cols = max(1, int((zone_x[1] - zone_x[0]) / gap))
        rows = max(1, int((zone_y[1] - zone_y[0]) / gap))
        pts  = []
        for row in range(rows):
            for col in range(cols):
                cx = zone_x[0] + gap * (col + 0.5)
                cy = zone_y[0] + gap * (row + 0.5)
                if (
                    cx <= zone_x[1]
                    and cy <= zone_y[1]
                    and self._is_reachable_xy(cx, cy)
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

    def _greedy_grid_centers(self, n_needed: int, gap: float,
                             rng: np.random.Generator) -> List[Tuple[float, float]]:
        """从 OBJECT_ZONE 内细网格中心点中贪心选取 n 个，两两距离 ≥ gap。

        注意：候选网格步长必须 **小于** gap，否则窄矩形内格点过少，无法摆下 4~5 个物体。
        """
        span_x = OBJECT_ZONE_X[1] - OBJECT_ZONE_X[0]
        span_y = OBJECT_ZONE_Y[1] - OBJECT_ZONE_Y[0]
        cell = max(0.04, min(gap * 0.52, span_x / 5.0, span_y / 7.0))
        candidates = self._deterministic_grid(
            max(80, n_needed * 20), OBJECT_ZONE_X, OBJECT_ZONE_Y, cell)
        if len(candidates) < n_needed:
            cell2 = max(0.035, cell * 0.85)
            candidates = self._deterministic_grid(
                max(80, n_needed * 20), OBJECT_ZONE_X, OBJECT_ZONE_Y, cell2)
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
        """在 OBJECT_ZONE 内摆放本回合所有方块与桶；优先网格+微抖动，再随机重试。"""
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
                x, y = centers[k]
                x += float(rng.uniform(-jitter, jitter))
                y += float(rng.uniform(-jitter, jitter))
                if not self._is_reachable_xy(x, y):
                    xy = self._project_to_reachable_xy(x, y)
                    x, y = float(xy[0]), float(xy[1])
                self._teleport(nm, x, y, zc)
            return True

        centers = self._greedy_grid_centers(n, gap, rng)
        if _try_assign(centers):
            return

        rospy.logwarn(
            "[Env] 网格摆放失败，回退到随机采样（仍保证间距与可达域）")

        placed_xy: List[Tuple[float, float]] = []
        for idx, (nm, zc) in enumerate(targets):
            ok = False
            for _ in range(400):
                x = float(rng.uniform(*OBJECT_ZONE_X))
                y = float(rng.uniform(*OBJECT_ZONE_Y))
                if not self._is_reachable_xy(x, y):
                    continue
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
                n, OBJECT_ZONE_X, OBJECT_ZONE_Y, gap)
            if idx < len(grid):
                x, y = grid[idx]
                placed_xy.append((x, y))
                self._teleport(nm, x, y, zc)
                rospy.logwarn(
                    f"[Env] {nm} 随机摆放失败，使用 fallback 网格 ({x:.3f},{y:.3f})")
            else:
                rospy.logerr(
                    "[Env] 物体无法摆放：请扩大 OBJECT_ZONE_* 或减小 "
                    "MIN_OBJECT_CENTER_GAP / 物体数量")

    def _randomize_scene(self):
        """
        随机化场景：
          1. 按 curriculum_mode 抽取方块/桶颜色子集与任务；
          2. 未激活模型从 Gazebo 删除；缺失模型按 world 模板 spawn；
          3. 在 OBJECT_ZONE_* 内摆放（网格优先 + 微抖动），中心距 ≥ MIN_OBJECT_CENTER_GAP。
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
        """刷新激活颜色物体的带噪声位置缓存（每次step调用）。"""
        noise = lambda: np.random.normal(0, self.noise_sigma, 2)
        for color in self._active_block_colors:
            xyz = self._get_pose(block_name(color))
            self._block_positions[color] = xyz[:2] + noise()
        for color in self._active_bin_colors:
            xyz = self._get_pose(bin_name(color))
            self._bin_positions[color]   = xyz[:2] + noise()

    def _get_block_pos(self, color: str) -> np.ndarray:
        return self._block_positions.get(
            color, np.zeros(2, dtype=np.float32))

    def _get_bin_pos(self, color: str) -> np.ndarray:
        return self._bin_positions.get(
            color, np.zeros(2, dtype=np.float32))

    # ── Observation构建 ──────────────────────────────────────────────────────

    def _build_crops(self) -> np.ndarray:
        """
        提取方块槽位对应的图像 crop（SLOT_COUNT 行；空槽为黑图）。
        返回形状 (SLOT_COUNT, 3, 28, 28)
        """
        import cv2
        crops = []
        img = self._latest_image

        for slot in range(SLOT_COUNT):
            if slot < len(self._active_block_colors):
                color = self._active_block_colors[slot]
                block_pos = self._get_block_pos(color)
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

        crops = self._build_crops()   # (SLOT_COUNT, 3, 28, 28)

        bp_list = []
        for slot in range(SLOT_COUNT):
            if slot < len(self._active_block_colors):
                bp_list.append(self._get_block_pos(self._active_block_colors[slot]))
            else:
                bp_list.append(np.zeros(2, dtype=np.float32))
        block_pos = np.array(bp_list, dtype=np.float32).flatten()

        bn_list = []
        for slot in range(SLOT_COUNT):
            if slot < len(self._active_bin_colors):
                bn_list.append(self._get_bin_pos(self._active_bin_colors[slot]))
            else:
                bn_list.append(np.zeros(2, dtype=np.float32))
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
        self._moveit_gripper.set_named_target(MOVEIT_GRIPPER_CLOSE_STATE)
        self._moveit_gripper.go(wait=True)
        self._gripper_open = False
        time.sleep(0.3)

    def _pose_for_yaw(self, yaw: float) -> tuple:
        half = float(yaw) * 0.5
        return (0.0, 0.0, math.sin(half), math.cos(half))

    def _move_to_xy(self, x: float, y: float, z: float, yaw: float = 0.0,
                    ignore_block_color: str = None) -> bool:
        """
        规划并执行到目标位姿（末端笛卡尔）。

        避障说明：此处仅调用 MoveIt MoveGroup.plan()/execute()，在 MoveIt
        规划场景（URDF/SRDF 碰撞几何、规划组）内做无碰撞轨迹搜索；**未**
        单独调用 sagittarius_ws 里的「额外避障包」。若未配置 Octomap /
        动态障碍更新，则主要避开机器人自身与环境静态模型；路径质量取决于
        OMPL 规划器与容差设置。

        若某姿态下 IK/规划不可行（常见于仅绕 Z 的 yaw 约束），会依次尝试：
          1) 给定 yaw 四元数
          2) 中性姿态 (w=1)，让 planner 自选末端朝向
          3) 略抬高 z（+0.05m）再试 1) 与 2)

        plan 失败直接返回 False。
        execute 失败（异常或返回 False）也返回 False，
        避免将"规划成功但执行失败"误报为 success=True 污染奖励信号。
        """
        # 规划前同步一次场景（避免每步重试重复同步）
        self._sync_moveit_planning_scene(ignore_block_color=ignore_block_color)
        self._moveit_arm.set_start_state_to_current_state()

        ee_link = self._moveit_arm.get_end_effector_link()
        attempts = []

        def _add_attempt(label: str, qx: float, qy: float, qz: float, qw: float,
                         zz: float):
            attempts.append((label, qx, qy, qz, qw, zz))

        qx_y, qy_y, qz_y, qw_y = self._pose_for_yaw(yaw)
        _add_attempt("yaw", qx_y, qy_y, qz_y, qw_y, z)
        _add_attempt("neutral_w1", 0.0, 0.0, 0.0, 1.0, z)
        # 略抬高：减少奇异/关节限位导致的无解
        z_up = float(z) + 0.05
        _add_attempt("yaw_z+up", qx_y, qy_y, qz_y, qw_y, z_up)
        _add_attempt("neutral_z+up", 0.0, 0.0, 0.0, 1.0, z_up)

        last_err = None
        for label, qx, qy, qz, qw, zz in attempts:
            target = PoseStamped()
            target.header.frame_id = "world"
            target.pose.position.x = float(x)
            target.pose.position.y = float(y)
            target.pose.position.z = float(zz)
            target.pose.orientation.x = qx
            target.pose.orientation.y = qy
            target.pose.orientation.z = qz
            target.pose.orientation.w = qw
            self._moveit_arm.clear_pose_targets()
            self._moveit_arm.set_pose_target(target, ee_link)
            plan_ok, traj, plan_time, err = self._moveit_arm.plan()
            last_err = (label, plan_ok, plan_time, err)
            if not plan_ok:
                rospy.logwarn(
                    f"[Env] plan 失败 [{label}] pos=({x:.3f},{y:.3f},{zz:.3f}) "
                    f"time={plan_time} err={err}")
                continue
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

        self._moveit_arm.clear_pose_targets()
        if last_err:
            rospy.logwarn(
                f"[Env] _move_to_xy 全部尝试失败，最后: {last_err}")
        return False

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
            self._planning_scene.attach_box(ee, block_name(color), p, (0.05, 0.05, 0.04))
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

    def _execute_pick(self, x: float, y: float, color: str, pose_id: int) -> bool:
        """
        完整 pick 原语：开爪 → 接近 → 下降 → 关爪 → 物理验证 → 抬起。

        抓取物理验证：
          关爪后等 Gazebo 物理引擎更新，检查目标方块 z 坐标。
          被夹起时方块随末端抬升，z 应明显高于 TABLE_Z + BLOCK_H。
          验证失败（方块仍在桌面）则开爪放弃，返回 False。
          这确保 success=True 时方块确实被夹住，奖励信号可信。
        """
        # 抬起验证阈值：必须高于桶口，再额外留 2cm 裕量
        LIFT_VERIFY_Z = TABLE_Z + BIN_H + 0.02

        yaw = float(self.pose_yaw_candidates[int(np.clip(pose_id, 0, self.pose_id_count - 1))])
        self._open_gripper()
        if not self._move_to_xy(
            x, y, TABLE_Z + APPROACH_H, yaw, ignore_block_color=color
        ):
            return False
        # 先降到方块顶面上方，再垂直落爪，减少从高空直线插向桌面时横扫方块/刮桌沿
        pre_z = TABLE_Z + BLOCK_H + PRE_GRASP_CLEAR_Z
        if not self._move_to_xy(
            x, y, pre_z, yaw, ignore_block_color=color
        ):
            return False
        if not self._move_to_xy(
            x, y, TABLE_Z + GRASP_H, yaw, ignore_block_color=color
        ):
            return False
        self._close_gripper()
        if not self._move_to_xy(
            x, y, TABLE_Z + APPROACH_H, yaw, ignore_block_color=color
        ):
            self._open_gripper()
            self._holding_color = None
            return False

        # ── 物理验证：抬起动作结束后再检查 z（不依赖固定 sleep） ─────────
        block_z = self._get_pose(block_name(color))[2]
        if block_z < LIFT_VERIFY_Z:
            rospy.logwarn(
                f"[Env] 抓取验证失败 {color}_block: z={block_z:.4f} "
                f"< 阈值 {LIFT_VERIFY_Z:.4f}，开爪放弃")
            self._open_gripper()
            self._holding_color = None
            return False

        self._holding_color = color
        self._attach_held_block_to_scene(color)
        return True

    def _execute_place(self, x: float, y: float, pose_id: int) -> bool:
        held = self._holding_color
        yaw = float(self.pose_yaw_candidates[int(np.clip(pose_id, 0, self.pose_id_count - 1))])
        if not self._move_to_xy(x, y, TABLE_Z + APPROACH_H, yaw): return False
        if not self._move_to_xy(x, y, TABLE_Z + PLACE_H, yaw):    return False
        self._open_gripper()
        if held:
            self._detach_held_block_from_scene(held)
        self._holding_color = None
        self._move_to_xy(x, y, TABLE_Z + APPROACH_H, yaw)
        return True

    def _return_home(self):
        try:
            self._moveit_arm.set_named_target(MOVEIT_ARM_HOME_STATE)
            self._moveit_arm.go(wait=True)
            self._open_gripper()
        except Exception:
            pass

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

        # _randomize_scene 内部会重新抽取 _active_colors 和任务颜色
        self._randomize_scene()
        time.sleep(0.5)
        self._return_home()

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
        }
        return obs, info

    def step(self, action: np.ndarray):
        self._step_count += 1

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
