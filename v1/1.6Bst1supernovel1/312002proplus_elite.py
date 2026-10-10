# ELITE 单文件版：复值 SSM 实值化（2×2 块同构）+ 原生 AMP，其余与 ultra 系列严格对照。
# ============================================================
# 212002proplus_maxpppp2_elite.py — 实值化精英版 — 流式 JSONL 数据加载版本
# 读取方式：全局摘要器（写入 cat([I_gated_norm, g])，读取 input_mod_linear(I) + global_mod_linear(g)）
# 内核改造：ParallelComplexWaveLayer 的全部复值算子（Ω/A_disc/ZOH/扫描/C·h）换成实值 2×2 块，
#           中间不再出现 complex 张量 → 可安全开 bf16 AMP；参数空间与 182 ultra 完全一致（可直接对照/迁移权重）。
# 对应源文件：182002proplus_maxpppp2_ultra.py（复值母本）
# 保留功能：显式冻结层控制（FREEZE_LAYERS / SEED_FREEZE）
#           嵌入层冻结控制（FREEZE_EMBEDDING / SEED_EMBEDDING）
# ============================================================
import json
import math
import unicodedata
import os
import random
import time
import contextlib
from datetime import datetime
from typing import List, Optional, Dict, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.checkpoint import checkpoint

# 必须安装 tokenizers 库
try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    HAS_TOKENIZERS = True
except ImportError:
    raise ImportError("请安装 tokenizers 库: pip install tokenizers")

# ------------------------------------------------------------
# 配置参数（仅保留核心训练与推理）
# ------------------------------------------------------------
VOCAB_SIZE = 40000
HIDDEN_SIZE =512
NUM_LAYERS = 5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 5.0
BATCH_SIZE = 1
EPOCHS = 1
LOG_INTERVAL = 1
TEMPERATURE = 0.7
TOP_P = 0.9
GRADIENT_ACCUMULATION_STEPS = 1
USE_GRADIENT_CHECKPOINT = True   # True=省内存但慢30%，False=快但费内存
WINDOW_OVERLAP = 0.0               # 0.0=不重叠, 0.5=50%%重叠
USE_AMP = True                   # 实值化后可安全开启 bf16 AMP（线性层/embedding 走 bf16，SSM 扫描与矩阵指数锁 fp32）
TIE_EMBEDDING_OUTPUT = True      # 共享 embedding 与输出层权重，省参数/梯度/优化器状态
LOSS_CHUNK_SIZE = 512            # 训练时按 token 分块算输出层loss；0=关闭分块

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEMORY = DEVICE.type == 'cuda'  # CPU 上自动关闭

MAX_SEQ_LEN = 1024
EARLY_STOP_PATIENCE = 3


# ============================================================
# Ultra 规则奖励训练参数（集中管理）
# ============================================================
# 拒绝采样：每道题生成多少个候选回答，只把最终答案正确的回答写入 SFT JSONL。
ULTRA_RS_SAMPLES_PER_QUESTION = 8
# 拒绝采样：每个候选回答最多生成多少 token。
ULTRA_RS_MAX_NEW_TOKENS = 512

# 基础 RL：单回答规则奖励强化学习。每题生成一次，答对奖励 1，答错奖励 0。
ULTRA_RL_EPOCHS = 1
ULTRA_RL_MAX_NEW_TOKENS = 512
ULTRA_RL_LEARNING_RATE = 1e-6
# 基础 RL 的移动平均基线。越接近 1 越平稳，但适应奖励变化越慢。
ULTRA_RL_BASELINE_MOMENTUM = 0.9

# GRPO：同一道题每组生成多少个回答，组内相对比较后更新。
ULTRA_GRPO_EPOCHS = 1
ULTRA_GRPO_GROUP_SIZE = 8
ULTRA_GRPO_MAX_NEW_TOKENS = 512
ULTRA_GRPO_LEARNING_RATE = 1e-6
# 每组 rollout 重复更新次数。建议先用 1，过大容易过拟合当前组。
ULTRA_GRPO_INNER_STEPS = 1
# 限制一次更新偏离旧策略的幅度。
ULTRA_GRPO_CLIP_EPSILON = 0.2
# 对偏离本组采样时旧策略的额外惩罚，减少训练突然跑飞。
ULTRA_GRPO_OLD_POLICY_PENALTY = 0.01


# 复值层超参数（实值化后仍沿用原语义：ν 漏水速率 / θ 相位 / M 子通道间耦合）
COMPLEX_MAX_FREQ = 2.0 * math.pi

# 稳定性惩罚超参数
STABILITY_AMP_Q0 = 1000.0        # amp 的参考值，超过才开始惩罚
STABILITY_DELTA_Q0 = 50.0        # delta 的参考值
STABILITY_LAMBDA = 1e-6          # 惩罚系数

# 复状态维度（关键可调旋钮）
#   STATE_DIM 必须为偶数；K = STATE_DIM // 2 为复值状态块数（即原复值通道数）。
#   实值化后内部状态向量长度为 S = STATE_DIM = 2K（实虚交替），K 可任意调整：
#   K=2 (STATE_DIM=4)  ← 默认（与 182 ultra 严格同构）
#   K=4 (STATE_DIM=8)  ← 更高动力学容量（原 MIMO 思路的轻量替代，不引入 block 结构，仅加通道）
#   注意：增大 K 会让 coupling_log 维度 = K*(K-1)//2 增长，参数量随之增加。
STATE_DIM = 6
# ============================================================
# 全局摘要器配置（写入端 + 读取端统一挂载）
# ============================================================
# SUMMARY_TYPE: 摘要器类型
#   'mean'            : 全局平均（无参数基线）
#   'linear_attention': 可学习内容检索
#   'mini_wave'       : 迷你版主模型（复用 ParallelComplexWaveLayer，独立规模）
SUMMARY_TYPE = 'linear_attention'

# 线性注意力摘要器参数
SUMMARY_ATTN_DK = 128          # key/query 投影维度

# mini_wave 摘要器参数
MINI_HIDDEN = 64              # 迷你隐藏维度
MINI_NUM_LAYERS = 1           # 迷你层数
MINI_STATE_DIM = 2            # 迷你状态维度，必须偶数，K = MINI_STATE_DIM // 2


# ============================================================
# 余弦退火重启调度器参数
# ============================================================
COSINE_T_0 = 1000            # 第一个余弦周期的步数（步数到达后重启学习率）
COSINE_T_MULT = 1            # 每次重启后周期倍数（1=固定周期，2=周期翻倍）
COSINE_ETA_MIN = 1e-6        # 学习率下限（余弦退火的最低点，防止学习率归零）

# ============================================================
# 冻结控制参数
# ============================================================
# FREEZE_LAYERS: 冻结前N层（0=不冻结，全部可训练）
#   设为1则冻结第0层，设为2则冻结第0-1层，以此类推
#   冻结层使用 SEED_FREEZE 独立初始化，不参与训练
FREEZE_LAYERS = 0

# SEED_FREEZE: 冻结层随机初始化种子（确保冻结区初始化可复现）
SEED_FREEZE = 42

# FREEZE_EMBEDDING: 是否冻结嵌入层和位置编码（False=可训练）
#   设为True时嵌入层使用 SEED_EMBEDDING 独立初始化并冻结
FREEZE_EMBEDDING = False

# SEED_EMBEDDING: 嵌入层独立随机种子（与 SEED_FREEZE 分开）
SEED_EMBEDDING = 42

# ============================================================
# 耦合控制参数
# ============================================================
# DISABLE_COUPLING: False keeps coupling enabled by default.
#   False: learnable coupling matrix is active.
#   True:  coupling matrix is zeroed and frozen.
DISABLE_COUPLING = False

# ============================================================
# 残差连接控制
# ============================================================
# USE_RESIDUAL: 是否使用残差连接（False=无残差模式，nr）
#   True:  每层输出 = wave_out + layer_input（标准配置）
#   False: 每层输出 = wave_out（无残差，信号衰减严重）
USE_RESIDUAL = True

# ============================================================
# MoE-FFN 配置（默认关闭，关闭时与 ultra 严格等价）
# ============================================================
MOE_ENABLED = True            # True=每层 SSM 后接 MoE-FFN（新增容量，非对照基线）
MOE_NUM_EXPERTS = 20            # 专家数 E
MOE_TOP_K = 2                  # 每个 token 激活的专家数
MOE_EXPERT_HIDDEN = 10240       # SwiGLU 专家中间维度
MOE_AUX_LOSS_COEF = 0.01       # 路由负载均衡辅助损失系数
MOE_RESIDUAL = True            # MoE 输出加残差（并接后置 RMSNorm）
# 专家卸载：'off'=全驻 GPU / 'on'=全驻 CPU 按需搬移 / 'auto'=按显存估算自动决定
MOE_OFFLOAD = 'on'
MOE_OFFLOAD_AUTO_RATIO = 0.25  # auto：专家权重估算字节数 > GPU 总显存×此比例则卸载
# 专家权重量化：None | 'fp8_e4m3'（E4M3 STE 训练量化） | 'nvfp4'（E2M1 4bit 推理量化）
MOE_EXPERT_QUANT = None
# 专家权重归一化（weight_norm）：把每个专家的权重矩阵拆成「方向 / ||方向|| · 幅度标量」，
#   权重范数恒等于可学习幅度（幅度进 no_decay），使 SwiGLU gating 乘积不再随范数二次方发散。
#   这是结构性稳定（球面上优化方向），非阈值/调参方案；True=开，False=原始裸权重（回退对照）。
MOE_EXPERT_WNORM = True

# ============================================================
# AMP 细粒度开关（True=锁 fp32；False=随 bf16 量化）
# ============================================================
AMP_FP32_GATE_CONV = True      # 门控因果卷积
AMP_FP32_SSM_SCAN = True       # 并行扫描 + step 串行递推（矩阵指数由 Omega 显式 fp32 天然锁定）
AMP_FP32_SUMMARIZER = True     # 摘要器状态累积（LinearAttention/MiniWave）

# ============================================================
# 独立量化开关（FP8 训练 / NVFP4 量化）—— 供 FP8Linear 与工具函数使用
# ============================================================
FP8_TRAIN = False              # True=FP8Linear 在训练时对权重做 E4M3 STE 量化
NVFP4_QUANTIZE = False         # True=权重打包为 E2M1 4bit（推理/微调，逐块缩放）
NVFP4_BLOCK_SIZE = 32          # NVFP4 量化块大小（scale 粒度）

# 安全有界范围
COUPLING_SEED = 42
_coupling_gen = torch.Generator().manual_seed(COUPLING_SEED)

# B/C 投影幅度安全上限
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------
# 日志辅助函数
# ------------------------------------------------------------
_LOG_FILE = None

def init_log_file():
    global _LOG_FILE
    if _LOG_FILE is None:
        _LOG_FILE = open("training_log.txt", "a", encoding='utf-8')
        _LOG_FILE.write(f"\n--- 新会话开始于 {datetime.now()} ---\n")
        _LOG_FILE.flush()

_LOG_FLUSH_COUNTER = 0
_LOG_FLUSH_INTERVAL = 50

def log_and_print(*args, **kwargs):
    import sys
    global _LOG_FLUSH_COUNTER
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    msg = sep.join(str(arg) for arg in args) + end
    sys.stdout.write(msg)
    sys.stdout.flush()
    if _LOG_FILE is not None:
        _LOG_FILE.write(msg)
        _LOG_FLUSH_COUNTER += 1
        if _LOG_FLUSH_COUNTER >= _LOG_FLUSH_INTERVAL:
            _LOG_FILE.flush()
            _LOG_FLUSH_COUNTER = 0

def close_log_file():
    global _LOG_FILE
    if _LOG_FILE is not None:
        _LOG_FILE.flush()
        _LOG_FILE.close()
        _LOG_FILE = None

# ------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------
def normalize_text(text: str) -> str:
    return unicodedata.normalize('NFKC', text).lower()

# ------------------------------------------------------------
# 高性能 BPE 分词器
# ------------------------------------------------------------
SPECIAL_TOKENS_LIST = ['<|bos|>', '<|eos|>', '<|pad|>', '<|unk|>', '<|user|>', '<|assistant|>', '<|end|>']

class FastBPE:
    def __init__(self, initial_vocab_size: int = VOCAB_SIZE):
        self.initial_vocab_size = initial_vocab_size
        self.special_tokens_list = SPECIAL_TOKENS_LIST.copy()
        self._tokenizer = None
        self.pad_token_id = None
        self.unk_token_id = None
        self.bos_token_id = None
        self.eos_token_id = None
        self.user_token_id = None
        self.assistant_token_id = None
        self.end_token_id = None

    def train(self, texts: List[str], verbose: bool = False):
        """
        训练 BPE 分词器。建议调用前先检查数据集字符规模。
        """
        normalized = [normalize_text(t) for t in texts]
        tokenizer = Tokenizer(BPE(unk_token='<|unk|>'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=self.initial_vocab_size,
            special_tokens=self.special_tokens_list,
            min_frequency=2,
            show_progress=verbose,
        )
        tokenizer.train_from_iterator(normalized, trainer=trainer)
        self._tokenizer = tokenizer
        self._update_special_ids()

    def _update_special_ids(self):
        if self._tokenizer is None:
            return
        vocab = self._tokenizer.get_vocab()
        self.bos_token_id = vocab.get('<|bos|>', 0)
        self.eos_token_id = vocab.get('<|eos|>', 1)
        self.pad_token_id = vocab.get('<|pad|>', 2)
        self.unk_token_id = vocab.get('<|unk|>', 3)
        self.user_token_id = vocab.get('<|user|>', 4)
        self.assistant_token_id = vocab.get('<|assistant|>', 5)
        self.end_token_id = vocab.get('<|end|>', 6)

    def encode(self, text: str) -> List[int]:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer 未训练或未加载")
        norm_text = normalize_text(text)
        ids = self._tokenizer.encode(norm_text).ids
        return [tid if tid < self._tokenizer.get_vocab_size() else self.unk_token_id for tid in ids]

    def decode(self, ids: List[int]) -> str:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer 未训练或未加载")
        # 修复 Bug 5：仅使用 decode 的 skip_special_tokens，移除手动过滤及 replace
        text = self._tokenizer.decode(ids, skip_special_tokens=True)
        return text

    def save(self, path: str):
        if self._tokenizer is None:
            log_and_print("警告：tokenizer未初始化，无法保存")
            raise RuntimeError("Tokenizer 未初始化，保存失败")
        try:
            dir_name, file_name = os.path.split(path)
            file_base = os.path.splitext(file_name)[0]
            tokenizer_path = os.path.join(dir_name, f"{file_base}_tokenizer.json")
            self._tokenizer.save(tokenizer_path)
            state = {
                'initial_vocab_size': self.initial_vocab_size,
                'special_tokens_list': self.special_tokens_list}
            torch.save(state, path)
        except Exception as e:
            # 修复 Bug 7：抛出异常，不再静默
            raise RuntimeError(f"保存 tokenizer 失败: {e}") from e

    def load(self, path: str):
        if not os.path.exists(path):
            log_and_print(f"警告: 文件 {path} 不存在，无法加载 tokenizer")
            return False
        try:
            data = torch.load(path, map_location='cpu', weights_only=False)
            self.initial_vocab_size = data.get('initial_vocab_size', VOCAB_SIZE)
            self.special_tokens_list = data.get('special_tokens_list', SPECIAL_TOKENS_LIST.copy())
        except Exception as e:
            # 修复 Bug 6：捕获具体异常，记录并返回 False
            log_and_print(f"加载 tokenizer 元数据失败 ({path}): {e}")
            return False

        dir_name, file_name = os.path.split(path)
        file_base = os.path.splitext(file_name)[0]
        tokenizer_path = os.path.join(dir_name, f"{file_base}_tokenizer.json")
        if os.path.exists(tokenizer_path):
            try:
                self._tokenizer = Tokenizer.from_file(tokenizer_path)
                self._tokenizer.pre_tokenizer = Whitespace()
                self._tokenizer.decoder = decoders.BPEDecoder()
                self._update_special_ids()
                log_and_print(f"成功从 {tokenizer_path} 加载 tokenizer")
                return True
            except Exception as e:
                log_and_print(f"加载 tokenizer.json 失败: {e}")
                return False
        else:
            log_and_print(f"未找到 {tokenizer_path}，无法加载 tokenizer")
            return False

    @property
    def vocab_size(self) -> int:
        if self._tokenizer is None:
            return len(self.special_tokens_list)
        return self._tokenizer.get_vocab_size()

# ------------------------------------------------------------
# 学习率调度器
# ------------------------------------------------------------
class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps: int, decay_steps=None, eta_min: float = 0.1, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            factor = (step + 1) / (self.warmup_steps + 1e-8)
        elif self.decay_steps is not None:
            t = step - self.warmup_steps
            progress = min(t / self.decay_steps, 1.0)
            factor = self.eta_min + (1.0 - self.eta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            t = step - self.warmup_steps
            factor = self.eta_min + (1.0 - self.eta_min) * (self.warmup_steps / (self.warmup_steps + t)) ** 0.5
        return [base_lr * factor for base_lr in self.base_lrs]



# ------------------------------------------------------------
# 余弦退火重启调度器（Cosine Annealing with Warm Restarts）
# ------------------------------------------------------------
class CosineAnnealingWarmRestartScheduler(_LRScheduler):
    """
    余弦退火重启调度器：
    - 学习率按余弦曲线从初始值降到 ETA_MIN
    - 每 T_0 步重启一次，学习率跳回初始值
    - T_mult 控制每次重启后周期长度的倍增系数
    - 适用于需要反复探索的训练场景，帮助逃离局部最优
    """
    def __init__(self, optimizer, T_0: int, T_mult: int = 1, eta_min: float = 0.0,
                 last_epoch: int = -1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_i = T_0       # 当前周期长度
        self.t_cur = 0        # 当前周期内的步数
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.t_cur >= self.T_i:
            self.t_cur = 0
            self.T_i = self.T_i * self.T_mult
        factor = self.eta_min + (1 - self.eta_min) *                  (1 + math.cos(math.pi * self.t_cur / self.T_i)) / 2
        self.t_cur += 1
        return [base_lr * max(0.0, min(1.0, factor)) for base_lr in self.base_lrs]
# ------------------------------------------------------------
# RMSNorm
# ------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

# ------------------------------------------------------------
# 无除法并行前缀扫描（Blelloch 扫描，完全可微版本）
# ------------------------------------------------------------
def _blelloch_scan(A_flat: torch.Tensor, b_flat: torch.Tensor) -> torch.Tensor:
    """关联前缀扫描（纯函数，无 autograd 记录）：h_t = A_t @ h_{t-1} + b_t，h_{-1}=0。
    A_flat: (B, T, K, K)   b_flat: (B, T, K)  ->  h: (B, T, K)
    实值/复值通用；Blelloch up/down sweep，O(log T) 次 kernel launch。"""
    B_flat, T, K, _ = A_flat.shape
    device = A_flat.device
    dtype = A_flat.dtype
    real_dtype = torch.float32 if dtype in (torch.float32, torch.complex64) else torch.float64

    # 填充到严格大于 T 的 2 次幂
    next_pow2 = 1 << (T - 1).bit_length()
    if next_pow2 <= T:
        next_pow2 = T * 2
    pad_len = next_pow2 - T
    eye = torch.eye(K, dtype=real_dtype, device=device).unsqueeze(0).unsqueeze(0).to(dtype)
    A_pad = torch.cat([A_flat, eye.expand(B_flat, pad_len, K, K)], dim=1)
    b_pad = torch.cat([b_flat, torch.zeros(B_flat, pad_len, K, dtype=dtype, device=device)], dim=1)
    T_padded = next_pow2
    max_level = T_padded.bit_length() - 1

    # Up-sweep
    for level in range(max_level):
        step = 1 << level
        left = torch.arange(step - 1, T_padded, 2 * step, device=device)
        if left.numel() == 0:
            continue
        right = left + step
        A_new = torch.matmul(A_pad[:, right, :, :], A_pad[:, left, :, :]).to(dtype)
        b_new = (torch.matmul(A_pad[:, right, :, :], b_pad[:, left, :].unsqueeze(-1)).squeeze(-1) + b_pad[:, right, :]).to(dtype)
        A_pad[:, right, :, :] = A_new
        b_pad[:, right, :] = b_new

    A_pad[:, -1, :, :] = torch.eye(K, dtype=real_dtype, device=device).unsqueeze(0).to(dtype)
    b_pad[:, -1, :] = 0.0

    # Down-sweep
    for level in range(max_level - 1, -1, -1):
        step = 1 << level
        left = torch.arange(step - 1, T_padded, 2 * step, device=device)
        if left.numel() == 0:
            continue
        right = left + step
        temp_A = A_pad[:, left, :, :].clone()
        temp_b = b_pad[:, left, :].clone()
        A_pad[:, left, :, :] = A_pad[:, right, :, :]
        b_pad[:, left, :] = b_pad[:, right, :]
        A_new = torch.matmul(temp_A, A_pad[:, right, :, :]).to(dtype)
        b_new = (torch.matmul(temp_A, b_pad[:, right, :].unsqueeze(-1)).squeeze(-1) + temp_b).to(dtype)
        A_pad[:, right, :, :] = A_new
        b_pad[:, right, :] = b_new

    b_excl = b_pad[:, :T, :].contiguous()   # 保存给 backward 用（= h_{t-1}），连续化避免保留 padded b_pad
    h = torch.matmul(A_flat, b_excl.unsqueeze(-1)).squeeze(-1) + b_flat
    return h, b_excl


class BlellochScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, b):
        """
        A: (..., T, K, K)   b: (..., T, K)
        返回 h: (..., T, K) 满足 h_t = A_t @ h_{t-1} + b_t
        """
        orig_shape = A.shape
        *batch_dims, T, K, _ = orig_shape
        dtype = A.dtype

        A_flat = A.reshape(-1, T, K, K).clone()
        b_flat = b.reshape(-1, T, K).clone()
        B_flat = A_flat.shape[0]

        A_orig = A_flat[:, :T, :, :].clone()

        h_flat, b_excl = _blelloch_scan(A_flat, b_flat)
        h = h_flat.view(*batch_dims, T, K)

        ctx.save_for_backward(A_orig, b_excl)
        ctx._dtype = dtype
        ctx._T = T
        ctx._K = K
        ctx._batch_dims = batch_dims
        ctx._B_flat = B_flat
        return h

    @staticmethod
    def backward(ctx, grad_h):
        A_orig, b_excl = ctx.saved_tensors
        T, K = ctx._T, ctx._K
        B_flat = ctx._B_flat
        device = A_orig.device
        dtype = ctx._dtype
        grad_h_flat = grad_h.reshape(B_flat, T, K)

        # ---- 反向并行化：h_{t-1} 直接取前向保存的 b_excl（= h_{t-1}），co-state 用 O(log T) 反向关联扫描 ----
        with torch.no_grad():
            h_prev = b_excl                                             # (B_flat, T, K)，h_prev[t] = h_{t-1}
            # 反向递推 dh_t = grad_h_t + A_{t+1}^H dh_{t+1}（dh_{T-1}=grad_h_{T-1}）
            # 时间轴反转后仍是线性关联扫描：D_r = A_rev[r-1] D_{r-1} + grad_rev[r]
            A_rev = torch.flip(A_orig, dims=[1]).transpose(-1, -2)
            if dtype.is_complex:
                A_rev = A_rev.conj()
            grad_rev = torch.flip(grad_h_flat, dims=[1])                # (B_flat, T, K)
            real_dtype = torch.float32 if dtype in (torch.float32, torch.complex64) else torch.float64
            eye = torch.eye(K, dtype=real_dtype, device=device).to(dtype).unsqueeze(0).expand(B_flat, 1, K, K)
            A_scan = torch.cat([eye, A_rev[:, :-1, :, :]], dim=1)       # (B_flat, T, K, K)
            D_rev, _ = _blelloch_scan(A_scan, grad_rev)                 # (B_flat, T, K)
            dh_total = torch.flip(D_rev, dims=[1]).contiguous()         # (B_flat, T, K)

        grad_b = dh_total
        # grad_A_t = dh_total_t ⊗ h_{t-1}^H（复值取共轭，实值恒等）
        grad_A = torch.einsum('btk,btj->btkj', dh_total, h_prev.conj())

        grad_A = grad_A.view(*ctx._batch_dims, T, K, K)
        grad_b = grad_b.view(*ctx._batch_dims, T, K)
        return grad_A, grad_b

def parallel_block_scan(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return BlellochScanFn.apply(A, b)

# ------------------------------------------------------------
# 对角复数波神经元层（实值化：2×2 块同构，无 complex 中间量；子通道间耦合 KxK 斜埃尔米特）
# 读取方式：全局摘要器（写入 cat([I_gated_norm, g])，读取 input_mod_linear(I) + global_mod_linear(g)）
# ------------------------------------------------------------
class ParallelComplexWaveLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, state_dim: int = 2, coupling_scale: float = 1.0,
                 disable_coupling: bool = False, disable_summarizer: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.state_dim = state_dim
        K = state_dim // 2
        self.disable_coupling = disable_coupling
        self.disable_summarizer = disable_summarizer
        self.layer_idx = -1

        self.input_linear = nn.Linear(input_dim, output_dim)
        self.input_gain = nn.Parameter(torch.ones(1))
        self.gate_linear = nn.Linear(output_dim, output_dim)

        # 修复 Bug 2：因果卷积，kernel_size=5，不带padding，手动左填充4
        self.gate_conv = nn.Conv1d(1, 1, kernel_size=5, padding=0, bias=False)
        nn.init.xavier_uniform_(self.gate_conv.weight)

        theta_min, theta_max = 0.01, math.pi
        log_theta = torch.rand(output_dim, K) * (math.log(theta_max) - math.log(theta_min)) + math.log(theta_min)
        self.theta_raw = nn.Parameter(log_theta)  # theta=exp(theta_raw)>0 恒正（结构性正约束，杜绝变负漂移）
        nu_min, nu_max = 0.01, 1.0
        log_nu = torch.rand(output_dim, K) * (math.log(nu_max) - math.log(nu_min)) + math.log(nu_min)
        self.nu_log = nn.Parameter(log_nu)

        # ---- 耦合参数，根据 disable_coupling 冻结为零 ----
        if disable_coupling:
            # 耦合强度因子置零且冻结
            # 耦合矩阵权重、偏置全部置零且冻结
            self.coupling_log = nn.Parameter(torch.full((output_dim, K * (K - 1) // 2), -20.0))
            self.coupling_log.requires_grad = False
        else:
            self.coupling_log = nn.Parameter(torch.full((output_dim, K * (K - 1) // 2), -2.0))

        from torch.nn.utils.parametrizations import weight_norm

        # ---- 读取端基底（两种模式共享） ----
        self.C_base_real = nn.Parameter(torch.randn(output_dim, K) * 0.1)
        self.C_base_imag = nn.Parameter(torch.randn(output_dim, K) * 0.1)

        if disable_summarizer:
            # ---- 朴素 170 模式（被 mini_wave 复用）：写入只看当前 token，读取 EMA + 当前 ----
            self.delta_linear = weight_norm(nn.Linear(output_dim, output_dim * K))
            self.B_linear = weight_norm(nn.Linear(output_dim, output_dim * state_dim))

            self.summary_linear = weight_norm(nn.Linear(output_dim * K * 2, output_dim * K))

            # P4 修改：EMA 衰减系数
            self.h_ema_beta = nn.Parameter(torch.tensor(0.38))

            self.input_mod_linear = weight_norm(nn.Linear(output_dim, output_dim * K))
        else:
            # ---- 摘要模式（180/181/182 外层）：写入/读取都吃全局摘要 g ----
            if SUMMARY_TYPE == 'mean':
                self.summarizer = MeanSummarizer(output_dim)
            elif SUMMARY_TYPE == 'linear_attention':
                self.summarizer = LinearAttentionSummarizer(output_dim, SUMMARY_ATTN_DK)
            elif SUMMARY_TYPE == 'mini_wave':
                self.summarizer = MiniWaveSummarizer(
                    output_dim, MINI_HIDDEN, MINI_NUM_LAYERS, MINI_STATE_DIM,
                    disable_coupling=disable_coupling, use_residual=USE_RESIDUAL,
                )
            else:
                raise ValueError(f"Unknown SUMMARY_TYPE: {SUMMARY_TYPE}")

            # 写入端：输入维度 D -> 2D（拼接当前 token 与全局摘要 g）
            self.delta_linear = weight_norm(nn.Linear(2 * output_dim, output_dim * K))
            self.B_linear = weight_norm(nn.Linear(2 * output_dim, output_dim * state_dim))

            # 读取端：当前 token 直接路径 + 全局摘要路径
            self.input_mod_linear = weight_norm(nn.Linear(output_dim, output_dim * K))
            self.global_mod_linear = weight_norm(nn.Linear(output_dim, output_dim * K))

        # 幅度调制参数（仅用于输出门控，与读取端无关）
        self.max_freq = COMPLEX_MAX_FREQ
        self.delta_scale = nn.Parameter(torch.tensor(-2.5))   # softplus(-2.5)≈0.075，delta 初始∈[0.001,0.1]
        self.B_scale = nn.Parameter(torch.tensor(0.0))       # softplus(0)≈0.693

        self.output_proj = nn.Linear(3 * K * output_dim, output_dim)
        self.output_norm = RMSNorm(output_dim)
        self._init_weights()

    def _init_weights(self):
        # 普通线性层：幅度被 RMS 归一化或 RMSNorm 兜底，无需 weight_norm
        plain = [self.input_linear, self.gate_linear, self.output_proj]
        for lin in plain:
            nn.init.xavier_uniform_(lin.weight, gain=1.0)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        # weight_norm 层：方向 xavier、幅度=方向范数（首次前向与裸初始化严格等价）
        wn_layers = [self.delta_linear, self.B_linear, self.input_mod_linear]
        if hasattr(self, 'summary_linear'):
            wn_layers.append(self.summary_linear)
        if hasattr(self, 'global_mod_linear'):
            wn_layers.append(self.global_mod_linear)
        for lin in wn_layers:
            gain = 0.1 if lin is self.delta_linear or lin is self.B_linear else 1.0
            v = lin.parametrizations.weight.original1
            nn.init.xavier_uniform_(v, gain=gain)
            lin.parametrizations.weight.original0.data = v.norm(dim=1, keepdim=True)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        # delta_linear 幅度可学习（进 no_decay）：指数耦合是"衰减耦合"（Ω 实部 -ν<0），
        #   delta 大 → exp(-δν) 更小 → 更快遗忘，不会爆炸，无需冻结

    def forward(self, x: torch.Tensor,
                time_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        K = self.state_dim // 2
        S = self.state_dim                            # 实值状态维度 = 2K（实虚交替）
        device = x.device

        I = self.input_linear(x) * self.input_gain
        I_conv = I.permute(0, 2, 1).reshape(B * D, 1, T)

        # 因果卷积：左侧填充 kernel_size-1（fp32，与 step 的手动点积严格一致）
        I_padded = F.pad(I_conv, (self.gate_conv.kernel_size[0] - 1, 0))
        with torch.amp.autocast(device_type=device.type, enabled=(not AMP_FP32_GATE_CONV)):
            local_bias = self.gate_conv(I_padded)          # (B*D, 1, T)
        local_bias = local_bias.view(B, D, T).permute(0, 2, 1)  # (B, T, D)

        gate = (self.gate_linear(I) + local_bias)
        I_gated = I * gate

        # === 幅度解耦：RMS 归一化输入驱动扫描，幅度旁路恢复 ===
        I_rms = torch.sqrt(torch.mean(I_gated**2, dim=-1, keepdim=True) + 1e-6)
        I_gated_norm = I_gated / I_rms
        # =========================================================

        if self.disable_summarizer:
            delta_raw = self.delta_linear(I_gated_norm)
            B_raw = self.B_linear(I_gated_norm)
        else:
            g = self.summarizer(I_gated_norm)                        # (B, T, D)
            x_with_summary = torch.cat([I_gated_norm, g], dim=-1)    # (B, T, 2D)
            delta_raw = self.delta_linear(x_with_summary)
            B_raw = self.B_linear(x_with_summary)

        delta = F.softplus(delta_raw) * F.softplus(self.delta_scale)
        delta = delta.view(B, T, D, K)                   # (B,T,D,K)

        # 实值 2×2 块 Omega：严格同构于复数 Ω = M + diag(-ν + iθ)，无复数中间量。
        # 复矩阵元 a+bi → [[a,-b],[b,a]]：实-实/虚-虚块 = M - diag(ν)，实-虚/虚-实块 = ∓diag(θ)。
        coupling = torch.exp(self.coupling_log)
        M = torch.zeros(D, K, K, device=device, dtype=coupling.dtype)
        idx = torch.triu_indices(K, K, offset=1, device=coupling.device)
        M[:, idx[0], idx[1]] = coupling
        M = M - M.transpose(-1, -2)                      # 实反对称

        nu = torch.exp(self.nu_log)                      # (D, K) 漏水速率，恒 > 0
        theta = torch.exp(self.theta_raw)                               # (D, K) 相位（RoPE 等价）

        Md = M - torch.diag_embed(nu)                    # (D, K, K) 实-实 & 虚-虚 对角块
        Omega = torch.zeros(D, S, S, device=device, dtype=torch.float32)
        Omega[:, 0::2, 0::2] = Md
        Omega[:, 1::2, 1::2] = Md
        thd = torch.diag_embed(theta)
        Omega[:, 0::2, 1::2] = -thd                      # 实-虚 = -diag(θ)
        Omega[:, 1::2, 0::2] = thd                       # 虚-实 = +diag(θ)
        Omega = Omega.unsqueeze(0).unsqueeze(0)          # (1, 1, D, S, S)

        delta_rep = delta.repeat_interleave(2, dim=-1)   # (B,T,D,S) 每通道复制到实虚两行
        A_disc = torch.linalg.matrix_exp(delta_rep.unsqueeze(-1) * Omega)   # (B,T,D,S,S) 实值

        # 精确 ZOH 注入（纯实值闭式）：w = (e^{δγ}-1)/γ，γ = -ν + iθ
        #   Re(w) = (ν(1-e·c) + θ·e·s) / (ν²+θ²)，Im(w) = (θ(1-e·c) - ν·e·s) / (ν²+θ²)
        #   e = e^{-δν}, c = cos(δθ), s = sin(δθ)。大 δ 饱和、小 δ 退化为欧拉 δ·B。
        B_raw = B_raw.view(B, T, D, K, 2)                # 末维 (re, im)
        B_real = B_raw[..., 0].float()                   # (B,T,D,K)
        B_imag = B_raw[..., 1].float()
        B_scale = F.softplus(self.B_scale)

        e = torch.exp(-delta * nu)                       # (B,T,D,K)
        c = torch.cos(delta * theta)
        s = torch.sin(delta * theta)
        denom = nu * nu + theta * theta                  # (D,K) |γ|²，恒 > 0（nu>0）
        a = (nu * (1 - e * c) + theta * e * s) / denom   # Re(w)
        b = (theta * (1 - e * c) - nu * e * s) / denom   # Im(w)
        B_disc_real = (B_real * a - B_imag * b) * B_scale
        B_disc_imag = (B_real * b + B_imag * a) * B_scale

        c_in = I_gated_norm.unsqueeze(-1)                # (B,T,D,1)
        b_input = torch.zeros(B, T, D, S, device=device, dtype=torch.float32)
        b_input[..., 0::2] = B_disc_real * c_in
        b_input[..., 1::2] = B_disc_imag * c_in

        if time_mask is not None:
            mask_exp = time_mask.float().to(A_disc.dtype).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            eye = torch.eye(S, dtype=A_disc.dtype, device=device).view(1,1,1,S,S)
            A_disc = A_disc * mask_exp + (1.0 - mask_exp) * eye
            b_input = b_input * mask_exp.squeeze(-1)

        # ---- 合并 B 和 D 维度，调用并行扫描（fp32 保持 SSM 递推稳定） ----
        B_orig, T_len, D_orig, S_ = A_disc.shape[0], A_disc.shape[1], A_disc.shape[2], A_disc.shape[-1]
        A_flat = A_disc.permute(0, 2, 1, 3, 4).reshape(B_orig * D_orig, T_len, S_, S_)
        b_flat = b_input.permute(0, 2, 1, 3).reshape(B_orig * D_orig, T_len, S_)
        with torch.amp.autocast(device_type=device.type, enabled=(not AMP_FP32_SSM_SCAN)):
            h_flat = parallel_block_scan(A_flat, b_flat)
        h = h_flat.view(B_orig, D_orig, T_len, S_).permute(0, 2, 1, 3)   # (B, T, D, S) 实值（实虚交替）
        h = torch.nan_to_num(h, nan=0.0)

        if self.disable_summarizer:
            # ========== EMA 替代因果累积均值 ==========
            beta = F.softplus(self.h_ema_beta)   # 数值稳定
            I_S = torch.eye(S, dtype=torch.float32, device=device)
            A_ema = (beta * I_S).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, D, S, S)
            b_ema = (1 - beta) * h                            # (B, T, D, S) 实值
            B_ema = B; T_ema = T; D_ema = D
            A_ema_flat = A_ema.permute(0, 2, 1, 3, 4).contiguous().reshape(B_ema * D_ema, T_ema, S, S)
            b_ema_flat = b_ema.permute(0, 2, 1, 3).reshape(B_ema * D_ema, T_ema, S)
            with torch.amp.autocast(device_type=device.type, enabled=(not AMP_FP32_SSM_SCAN)):
                h_ema_flat = parallel_block_scan(A_ema_flat, b_ema_flat)
            h_ema = h_ema_flat.view(B_ema, D_ema, T_ema, S).permute(0, 2, 1, 3)  # (B, T, D, S)
            h_ema = torch.nan_to_num(h_ema, nan=0.0)
            mod = self.summary_linear(h_ema.flatten(2)) + self.input_mod_linear(I_gated_norm)
        else:
            # 全局摘要读取：当前 token 直接路径 + 摘要器输出
            mod = self.input_mod_linear(I_gated_norm) + self.global_mod_linear(g)

        mod = mod.view(B, T_len, D_orig, K)              # (B, T, D, K)

        # 读取：C 作用于 h，复数乘 (C_re + i C_im)(h_re + i h_im) 的实/虚闭式
        C_real = self.C_base_real.unsqueeze(0).unsqueeze(0)       # (1, 1, D, K)
        C_imag = self.C_base_imag.unsqueeze(0).unsqueeze(0)
        h_re = h[..., 0::2]                              # (B, T, D, K)
        h_im = h[..., 1::2]
        y_re = (C_real * h_re - C_imag * h_im) * mod
        y_im = (C_real * h_im + C_imag * h_re) * mod

        # ========== 幅度输出（310 系列：已拆除调和缩放 A，y 保持原始扫描输出）==========
        amp = torch.sqrt(y_re**2 + y_im**2 + 1e-12)
        feat = torch.cat([y_re, y_im, amp], dim=-1)
        feat_flat = feat.flatten(2)
        wave_raw = self.output_proj(feat_flat)

        wave_out = self.output_norm(wave_raw)

        if time_mask is not None:
            wave_out = wave_out * time_mask.unsqueeze(-1).float()

        if self.training and T > 0:
            self._last_mon = (I.std().detach(), amp.std().detach(), wave_raw.mean().detach(), wave_raw.std().detach(),
                              delta.mean().detach(), delta.min().detach(), delta.max().detach())

        # 稳定性惩罚（仅训练时计算）
        stab_penalty = torch.zeros(1, device=device)

        return wave_out, stab_penalty

    def init_state(self, device):
        """初始化递归推理状态：主递推 h（实值 2K 维，实虚交替）；摘要模式额外持有摘要器状态，朴素 170 模式持有 EMA 状态。"""
        D = self.output_dim
        S = self.state_dim
        ks = self.gate_conv.kernel_size[0]
        state = {
            'h': torch.zeros(D, S, dtype=torch.float32, device=device),
            'conv': torch.zeros(ks - 1, D, dtype=torch.float32, device=device),
        }
        if self.disable_summarizer:
            state['h_ema'] = torch.zeros(D, S, dtype=torch.float32, device=device)
        else:
            state['summarizer'] = self.summarizer.init_state(device, D=D)
        return state

    def step(self, x_t, state):
        """单 token 递归步：x_t (D,) -> y_t (D,)，原地更新 state。与 forward 逐时间步严格等价。"""
        D = self.output_dim
        K = self.state_dim // 2
        S = self.state_dim
        device = x_t.device

        I_t = self.input_linear(x_t) * self.input_gain

        w = self.gate_conv.weight[0, 0]
        hist = torch.cat([state['conv'], I_t.unsqueeze(0)], dim=0)  # [I_{t-4},...,I_t]
        local_bias = (w.unsqueeze(1) * hist).sum(0)                  # w0*I_{t-4}+...+w4*I_t
        state['conv'] = torch.cat([state['conv'][1:], I_t.unsqueeze(0)], dim=0)

        gate = self.gate_linear(I_t) + local_bias
        I_gated = I_t * gate

        I_rms = torch.sqrt(torch.mean(I_gated ** 2) + 1e-6)
        I_gated_norm = I_gated / I_rms

        if self.disable_summarizer:
            delta = F.softplus(self.delta_linear(I_gated_norm)) * F.softplus(self.delta_scale)
            B_raw = self.B_linear(I_gated_norm).view(D, K, 2)
        else:
            g_t = self.summarizer.step(I_gated_norm, state['summarizer'])   # (D,)
            x_with_summary = torch.cat([I_gated_norm, g_t], dim=-1)         # (2D,)
            delta = F.softplus(self.delta_linear(x_with_summary)) * F.softplus(self.delta_scale)
            B_raw = self.B_linear(x_with_summary).view(D, K, 2)

        delta = delta.view(D, K)

        nu = torch.exp(self.nu_log)                       # (D, K)
        theta = torch.exp(self.theta_raw)                                # (D, K)

        M = torch.zeros(D, K, K, device=device, dtype=torch.float32)
        idx = torch.triu_indices(K, K, offset=1, device=device)
        M[:, idx[0], idx[1]] = torch.exp(self.coupling_log)
        M = M - M.transpose(-1, -2)

        Md = M - torch.diag_embed(nu)                     # (D, K, K)
        Omega = torch.zeros(D, S, S, device=device, dtype=torch.float32)
        Omega[:, 0::2, 0::2] = Md
        Omega[:, 1::2, 1::2] = Md
        thd = torch.diag_embed(theta)
        Omega[:, 0::2, 1::2] = -thd
        Omega[:, 1::2, 0::2] = thd

        delta_rep = delta.repeat_interleave(2, dim=-1)    # (D, S)
        A_disc = torch.linalg.matrix_exp(delta_rep.unsqueeze(-1) * Omega)   # (D, S, S)

        B_real = B_raw[..., 0].float()                    # (D, K)
        B_imag = B_raw[..., 1].float()
        B_scale = F.softplus(self.B_scale)

        e = torch.exp(-delta * nu)
        c = torch.cos(delta * theta)
        s = torch.sin(delta * theta)
        denom = nu * nu + theta * theta
        a = (nu * (1 - e * c) + theta * e * s) / denom
        b = (theta * (1 - e * c) - nu * e * s) / denom
        B_disc_real = (B_real * a - B_imag * b) * B_scale
        B_disc_imag = (B_real * b + B_imag * a) * B_scale

        b_t = torch.zeros(D, S, device=device, dtype=torch.float32)
        b_t[:, 0::2] = B_disc_real * I_gated_norm.unsqueeze(-1)
        b_t[:, 1::2] = B_disc_imag * I_gated_norm.unsqueeze(-1)

        with torch.amp.autocast(device_type=device.type, enabled=(not AMP_FP32_SSM_SCAN)):
            h_t = torch.matmul(A_disc, state['h'].unsqueeze(-1)).squeeze(-1) + b_t
        h_t = torch.nan_to_num(h_t, nan=0.0)
        state['h'] = h_t

        if self.disable_summarizer:
            beta = F.softplus(self.h_ema_beta)
            h_ema_t = beta * state['h_ema'] + (1.0 - beta) * h_t
            state['h_ema'] = h_ema_t
            mod = (self.summary_linear(h_ema_t.flatten()) + self.input_mod_linear(I_gated_norm)).view(D, K)
        else:
            mod = (self.input_mod_linear(I_gated_norm) + self.global_mod_linear(g_t)).view(D, K)

        C_real = self.C_base_real                       # (D, K)
        C_imag = self.C_base_imag
        h_re = h_t[:, 0::2]                             # (D, K)
        h_im = h_t[:, 1::2]
        y_re = (C_real * h_re - C_imag * h_im) * mod
        y_im = (C_real * h_im + C_imag * h_re) * mod

        # 幅度输出（310 系列：已拆除调和缩放 A，y 保持原始扫描输出）
        amp = torch.sqrt(y_re ** 2 + y_im ** 2 + 1e-12)
        feat = torch.cat([y_re, y_im, amp], dim=-1).flatten()
        wave_raw = self.output_proj(feat)
        wave_out = self.output_norm(wave_raw)
        return wave_out


# ------------------------------------------------------------
# 全局摘要器：三种实现（统一 init_state(device, D)/step(x_t, state) 接口）
# ------------------------------------------------------------
class MeanSummarizer(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.D = D

    def forward(self, x):
        B, T, D = x.shape
        denom = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, T, 1)
        return torch.cumsum(x, dim=1) / denom

    def init_state(self, device, D=None):
        D = D or self.D
        return {'sum': torch.zeros(D, device=device, dtype=torch.float32), 'count': 0}

    def step(self, x_t, state):
        state['sum'] = state['sum'] + x_t
        state['count'] += 1
        return state['sum'] / state['count']


class LinearAttentionSummarizer(nn.Module):
    def __init__(self, D, dk):
        super().__init__()
        self.D = D
        self.dk = dk
        from torch.nn.utils.parametrizations import weight_norm
        self.W_K = nn.Linear(D, dk)
        self.W_Q = nn.Linear(D, dk)
        self.W_V = weight_norm(nn.Linear(D, D))
        for lin in [self.W_K, self.W_Q]:
            nn.init.xavier_uniform_(lin.weight, gain=1.0)
            nn.init.zeros_(lin.bias)
        _v = self.W_V.parametrizations.weight.original1
        nn.init.xavier_uniform_(_v, gain=1.0)
        self.W_V.parametrizations.weight.original0.data = _v.norm(dim=1, keepdim=True)
        nn.init.zeros_(self.W_V.bias)

    def forward(self, x):
        B, T, D = x.shape
        k = F.normalize(self.W_K(x), p=2, dim=-1)   # (B,T,dk) 余弦归一化，q·k ∈ [-1,1] 有界
        v = self.W_V(x)       # (B,T,D)
        q = F.normalize(self.W_Q(x), p=2, dim=-1)   # (B,T,dk) 余弦归一化
        # 状态累积与读数锁 fp32（AMP 只作用于投影层），与 step 严格一致
        with torch.amp.autocast(device_type=x.device.type, enabled=(not AMP_FP32_SUMMARIZER)):
            # cumsum 向量化：S_t = Σ_{s≤t} k_s ⊗ v_s，替代逐 t Python 循环（数学等价、纯 PyTorch、零新依赖）
            incr = torch.einsum('btd,bte->btde', k, v)
            if incr.dtype in (torch.float16, torch.bfloat16):
                incr = incr.float()   # 与 step 的 fp32 累积地板一致（AMP 下 k 经 F.normalize 后恒 fp32）
            S = torch.cumsum(incr, dim=1)   # (B,T,dk,D) 锁 fp32 累积
            g = torch.einsum('btd,btde->bte', q, S) / torch.arange(1, T + 1, device=x.device, dtype=S.dtype).view(1, T, 1)
        return g   # (B,T,D)

    def init_state(self, device, D=None):
        return {'S': torch.zeros(self.dk, self.D, device=device, dtype=torch.float32), 'count': 0}

    def step(self, x_t, state):
        k_t = F.normalize(self.W_K(x_t), p=2, dim=-1)   # (dk) 余弦归一化
        v_t = self.W_V(x_t)   # (D)
        q_t = F.normalize(self.W_Q(x_t), p=2, dim=-1)   # (dk) 余弦归一化
        # 状态累积与读数锁 fp32，与 forward 严格一致
        with torch.amp.autocast(device_type=x_t.device.type, enabled=(not AMP_FP32_SUMMARIZER)):
            state['S'] = state['S'] + torch.outer(k_t, v_t)
            state['count'] += 1
            g_t = torch.matmul(q_t, state['S']) / state['count']  # (D)
        return g_t


class MiniWaveSummarizer(nn.Module):
    def __init__(self, D, mini_hidden, mini_num_layers, mini_state_dim,
                 disable_coupling, use_residual):
        super().__init__()
        self.D = D
        self.mini_hidden = mini_hidden
        self.use_residual = use_residual
        from torch.nn.utils.parametrizations import weight_norm
        self.input_proj = nn.Linear(D, mini_hidden)
        self.output_proj = weight_norm(nn.Linear(mini_hidden, D))
        self.norms = nn.ModuleList([RMSNorm(mini_hidden) for _ in range(mini_num_layers)])
        self.layers = nn.ModuleList()
        for _ in range(mini_num_layers):
            self.layers.append(
                ParallelComplexWaveLayer(
                    input_dim=mini_hidden,
                    output_dim=mini_hidden,
                    state_dim=mini_state_dim,
                    disable_coupling=disable_coupling,
                    disable_summarizer=True,
                )
            )
        nn.init.xavier_uniform_(self.input_proj.weight, gain=1.0)
        nn.init.zeros_(self.input_proj.bias)
        _v = self.output_proj.parametrizations.weight.original1
        nn.init.xavier_uniform_(_v, gain=1.0)
        self.output_proj.parametrizations.weight.original0.data = _v.norm(dim=1, keepdim=True)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        x = self.input_proj(x)
        layer_input = x
        for i, layer in enumerate(self.layers):
            out, _ = layer(layer_input)
            if self.use_residual:
                out = out + layer_input
            out = self.norms[i](out)
            layer_input = out
        return self.output_proj(layer_input)

    def init_state(self, device, D=None):
        return [layer.init_state(device) for layer in self.layers]

    def step(self, x_t, states):
        x_t = self.input_proj(x_t)
        layer_input = x_t
        for i, layer in enumerate(self.layers):
            out = layer.step(layer_input, states[i])
            if self.use_residual:
                out = out + layer_input
            out = self.norms[i](out)
            layer_input = out
        return self.output_proj(layer_input)


# ------------------------------------------------------------
# 对话数据集

# ------------------------------------------------------------
# 流式 JSONL 数据集（逐行读取，不加载全部到内存）
# ------------------------------------------------------------
class StreamingJSONLDataset(IterableDataset):
    """shuffle buffer 流式 JSONL：顺序读取，buffer 打乱，高性能 tokenize。"""

    def __init__(self, file_path, tokenizer, max_seq_len=MAX_SEQ_LEN,
                 epoch=0, is_val=False, assigned_indices=None):
        super().__init__()
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.epoch = epoch
        self.is_val = is_val
        self.assigned_indices = assigned_indices
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.user_id = tokenizer.user_token_id
        self.assistant_id = tokenizer.assistant_token_id
        self.end_id = tokenizer.end_token_id
        self.pad_id = tokenizer.pad_token_id
        if any(x is None for x in [self.bos_id, self.eos_id, self.user_id, self.assistant_id, self.end_id]):
            raise ValueError("Tokenizer 缺少必需的特殊 token")

    def _encode(self, text):
        """??? FastBPE.encode????????????/UNK ?????"""
        return self.tokenizer.encode(text)

    def _tokenize_conversation(self, convs):
        kept_pairs = []
        token_count = 0
        id_cache = {}   # 缓存 content -> ids，避免同一段 content 被重复 encode（原来算长度 + 实际使用各 encode 一次）
        i = len(convs) - 1
        while i >= 1:
            assistant_turn = convs[i]
            user_turn = convs[i - 1]
            if assistant_turn.get('role') == 'assistant' and user_turn.get('role') == 'user':
                pair = [user_turn, assistant_turn]
                pair_tokens = 0
                for turn in pair:
                    content = turn.get('content', '')
                    ids = id_cache.get(content)
                    if ids is None:
                        ids = self._encode(content)
                        id_cache[content] = ids
                    pair_tokens += len(ids) + 2
                kept_pairs.insert(0, pair)
                token_count += pair_tokens
                if token_count + 2 >= self.max_seq_len:
                    break
                i -= 2
            else:
                i -= 1
        convs = [turn for pair in kept_pairs for turn in pair]
        tokens = []
        assistant_mask = []
        for turn in convs:
            role = turn['role']
            content = turn['content']
            content_ids = id_cache.get(content)
            if content_ids is None:
                content_ids = self._encode(content)
                id_cache[content] = content_ids
            if role == 'user':
                tokens.append(self.user_id)
                assistant_mask.append(False)
                tokens.extend(content_ids)
                assistant_mask.extend([False] * len(content_ids))
                tokens.append(self.end_id)
                assistant_mask.append(False)
            elif role == 'assistant':
                tokens.append(self.assistant_id)
                assistant_mask.append(False)
                tokens.extend(content_ids)
                assistant_mask.extend([True] * len(content_ids))
                tokens.append(self.end_id)
                assistant_mask.append(True)
        tokens = [self.bos_id] + tokens
        assistant_mask = [False] + assistant_mask
        tokens.append(self.eos_id)
        assistant_mask.append(True)
        input_ids = tokens[:-1]
        labels = tokens[1:]
        labels = [lbl if mask else -100 for lbl, mask in zip(labels, assistant_mask[1:])]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        rng = random.Random(42 + self.epoch * 1000 + worker_id)
        buffer = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if line_idx % num_workers != worker_id:
                    continue
                if self.assigned_indices is not None and line_idx not in self.assigned_indices:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    convs = data.get('conversations', [])
                    if not convs:
                        continue
                    sample = self._tokenize_conversation(convs)
                except (json.JSONDecodeError, KeyError):
                    continue
                if self.is_val:
                    yield sample
                else:
                    buffer.append(sample)
                    if len(buffer) >= 1000:
                        rng.shuffle(buffer)
                        while buffer:
                            yield buffer.pop()
        if not self.is_val and buffer:
            rng.shuffle(buffer)
            while buffer:
                yield buffer.pop()

class TextFileDataset(IterableDataset):
    """随机无放回抽样 txt：首次 tokenize 写 memmap 缓存，之后随机抽窗口序号。"""

    def __init__(self, file_path, tokenizer, max_seq_len=MAX_SEQ_LEN,
                 epoch=0, is_val=False, window_overlap=0.0):
        super().__init__()
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.epoch = epoch
        self.is_val = is_val
        self.window_overlap = window_overlap  # 0.0=不重叠, 0.5=50%重叠
        self._token_ids = None
        self._num_windows = 0

    def _ensure_tokenized(self):
        if self._token_ids is not None:
            return
        cache_file = self.file_path + f".len{self.max_seq_len}.norm.ids.bin"
        if os.path.exists(cache_file) and os.path.getmtime(cache_file) >= os.path.getmtime(self.file_path):
            total_ids = os.path.getsize(cache_file) // 4
            self._token_ids = np.memmap(cache_file, dtype=np.int32, mode='r', shape=(total_ids,))
            step = max(1, int(self.max_seq_len * (1 - self.window_overlap)))
            window_size = self.max_seq_len + 1
            self._num_windows = max(0, (total_ids - window_size) // step + 1)
            return
        log_and_print(f"首次 tokenize 缓存生成中（流式批量 Rust 编码）...")
        t0 = time.time()
        vocab_size = self.tokenizer.vocab_size
        unk_id = self.tokenizer.unk_token_id
        chunk_size = 4096
        total_ids = 0
        buf = []
        with open(cache_file, 'wb') as out, open(self.file_path, 'r', encoding='utf-8') as f:
            def flush():
                nonlocal total_ids
                if not buf:
                    return
                normalized = [normalize_text(t) for t in buf]
                encodings = self.tokenizer._tokenizer.encode_batch(normalized)
                chunk_ids = []
                for enc in encodings:
                    chunk_ids.extend(tid if tid < vocab_size else unk_id for tid in enc.ids)
                ids = np.array(chunk_ids, dtype=np.int32)
                ids.tofile(out)
                total_ids += ids.size
                buf.clear()
            for line in f:
                if not line.strip():
                    continue
                buf.append(line.rstrip('\n\r') + '\n')
                if len(buf) >= chunk_size:
                    flush()
            flush()
        elapsed = time.time() - t0
        if total_ids == 0:
            if os.path.exists(cache_file):
                os.remove(cache_file)
            self._token_ids = None
            self._num_windows = 0
            return
        log_and_print(f"tokenize 完成：{total_ids:,} tokens，耗时 {elapsed:.1f}s")
        self._token_ids = np.memmap(cache_file, dtype=np.int32, mode='r', shape=(total_ids,))
        step = max(1, int(self.max_seq_len * (1 - self.window_overlap)))
        window_size = self.max_seq_len + 1
        self._num_windows = max(0, (total_ids - window_size) // step + 1)

    def __iter__(self):
        self._ensure_tokenized()
        if self._token_ids is None or self._num_windows == 0:
            return
        rng = random.Random(42 + self.epoch * 1000)
        step = max(1, int(self.max_seq_len * (1 - self.window_overlap)))
        indices = list(range(self._num_windows))
        if not self.is_val:
            rng.shuffle(indices)
        for idx in indices:
            start = idx * step
            end = start + self.max_seq_len + 1
            window = self._token_ids[start:end]
            input_ids = torch.tensor(np.array(window[:self.max_seq_len]), dtype=torch.long)
            labels = torch.tensor(np.array(window[1:self.max_seq_len + 1]), dtype=torch.long)
            yield input_ids, labels

def make_collate_fn(pad_id):
    def collate_fn(batch):
        valid_batch = []
        for inp, lbl in batch:
            try:
                if inp.numel() > 0 and lbl.numel() > 0:
                    valid_batch.append((inp, lbl))
            except:
                continue
        if not valid_batch:
            return torch.empty(0, 0, dtype=torch.long), torch.empty(0, 0, dtype=torch.long), torch.empty(0, 0, dtype=torch.long)
        input_ids_list, labels_list = zip(*valid_batch)
        max_len = max(len(ids) for ids in input_ids_list)
        padded_inputs = []
        padded_labels = []
        attention_masks = []
        for inp, lbl in zip(input_ids_list, labels_list):
            pad_len = max_len - len(inp)
            if pad_len > 0:
                inp = F.pad(inp, (0, pad_len), value=pad_id)
                lbl = F.pad(lbl, (0, pad_len), value=-100)
            mask = (inp != pad_id).long()
            padded_inputs.append(inp)
            padded_labels.append(lbl)
            attention_masks.append(mask)
        return torch.stack(padded_inputs), torch.stack(padded_labels), torch.stack(attention_masks)
    return collate_fn

# ------------------------------------------------------------
# MoE-FFN（SwiGLU 专家 + Top-K 软路由 + 专家卸载）与量化工具
# ------------------------------------------------------------
NVFP4_MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)   # E2M1 正数量级集合


def e4m3_round_ste(w: torch.Tensor) -> torch.Tensor:
    """权重按 E4M3 网格舍入后回 cast 成 fp32（训练时用于 STE：w + (wq - w).detach()）。
    仅 CUDA 支持（原生 float8 舍入）；非 CUDA 直接返回原值。"""
    if not w.is_cuda:
        return w
    w32 = w.float()
    scale = w32.detach().abs().amax().clamp_min(1e-24) / 448.0
    q = (w32 / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32)
    return q * scale


class FP8Linear(nn.Linear):
    """带 E4M3 STE 权重量化的线性层。FP8_TRAIN=False 时退化为普通 nn.Linear。"""
    def forward(self, x):
        w = self.weight
        if FP8_TRAIN and w.is_cuda and self.training:
            wq = e4m3_round_ste(w)
            w = w + (wq - w).detach()
        return F.linear(x, w, self.bias)


def nvfp4_quantize(w: torch.Tensor, block_size: int = NVFP4_BLOCK_SIZE):
    """权重 w（任意形状）-> (packed uint8, scales fp32, shape)。
    逐 block_size 元素一个 scale，量化到 E2M1 量级集合 {0,±0.5,±1,±1.5,±2,±3,±4,±6}，4bit 打包（2 元素/字节）。"""
    shape = tuple(w.shape)
    w = w.float().reshape(-1)
    n = w.numel()
    pad = (-n) % (2 * block_size)
    if pad:
        w = F.pad(w, (0, pad))
    nblk = w.numel() // block_size
    wb = w.view(nblk, block_size)
    max_abs = wb.abs().amax(dim=1, keepdim=True).clamp_min(1e-24)
    scales = max_abs / 6.0
    v = (wb / scales).clamp(-6.0, 6.0).reshape(-1)
    mag = v.abs()
    mag_t = torch.tensor(NVFP4_MAG, device=w.device, dtype=torch.float32)
    mid = (mag_t[1:] + mag_t[:-1]) / 2.0
    idx = torch.bucketize(mag, mid).to(torch.int64)      # 0..7
    sign = (v < 0).to(torch.int64)
    code = (idx & 0x7) | ((sign & 1) << 3)               # 4bit: [sign][exp2][m1]
    code = code.view(-1, 2)
    packed = ((code[:, 0] & 0xF) | ((code[:, 1] & 0xF) << 4)).to(torch.uint8)
    return packed, scales, shape


def nvfp4_dequantize(packed: torch.Tensor, scales: torch.Tensor, shape, block_size: int = NVFP4_BLOCK_SIZE) -> torch.Tensor:
    """nvfp4_quantize 的逆：还原为 fp32，形状恢复为 shape。"""
    import math as _math
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    code = torch.stack([lo, hi], dim=-1).reshape(-1).to(torch.int64)
    mag_t = torch.tensor(NVFP4_MAG, device=packed.device, dtype=torch.float32)
    mag = mag_t[code & 0x7]
    sign = ((code >> 3) & 1).to(torch.float32) * (-2.0) + 1.0
    val = (mag * sign).view(-1, block_size)
    out = (val * scales).reshape(-1)
    orig_n = _math.prod(shape)
    return out[:orig_n].reshape(shape)


class NVFP4Linear(nn.Module):
    """NVFP4 4-bit 权重量化线性层（推理用量化）：权重打包为 uint8 + 逐块 fp32 scale，前向时反量化。
    替换后释放原 fp32 主权重，权重内存约为原来的 1/4。"""
    def __init__(self, in_features, out_features, bias=True, block_size=NVFP4_BLOCK_SIZE):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    @classmethod
    def from_linear(cls, lin, block_size=NVFP4_BLOCK_SIZE):
        q = cls(lin.in_features, lin.out_features, lin.bias is not None, block_size)
        packed, scales, _ = nvfp4_quantize(lin.weight.data, block_size)
        q.register_buffer('packed', packed)
        q.register_buffer('scales', scales)
        if lin.bias is not None:
            q.bias = nn.Parameter(lin.bias.data.detach().clone())
        return q

    def weight_fp32(self):
        return nvfp4_dequantize(self.packed, self.scales, (self.out_features, self.in_features), self.block_size)

    def forward(self, x):
        return F.linear(x, self.weight_fp32(), self.bias)


def apply_fp8(module):
    """递归把模块内所有 nn.Linear 替换为 FP8Linear（拷贝原权重）。
    配合 FP8_TRAIN 开关：关闭时与 nn.Linear 完全等价，打开时训练用 E4M3 STE 量化。"""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, FP8Linear):
            repl = FP8Linear(child.in_features, child.out_features, child.bias is not None).to(child.weight.device)
            repl.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                repl.bias.data.copy_(child.bias.data)
            setattr(module, name, repl)
        else:
            apply_fp8(child)
    return module


def quantize_nvfp4(module, block_size=NVFP4_BLOCK_SIZE):
    """递归把模块内所有 nn.Linear 替换为 NVFP4Linear（量化并释放 fp32 主权重）。
    MoE 专家是 3D 参数矩阵，请改用 MOE_EXPERT_QUANT='nvfp4'；embedding 查表不参与本函数。"""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, NVFP4Linear.from_linear(child, block_size))
        else:
            quantize_nvfp4(child, block_size)
    return module


def decide_moe_offload(dim: int) -> bool:
    """按全局配置决定是否启用专家卸载（offload）。"""
    mode = MOE_OFFLOAD
    if mode == 'off':
        return False
    if mode == 'on':
        return True
    # auto：估算专家权重总字节数，超过 GPU 总显存×比例则卸载
    if not torch.cuda.is_available():
        return False
    H = MOE_EXPERT_HIDDEN
    bytes_per_expert = (2 * dim * H + H * dim + 2 * H + dim) * 4
    total = MOE_NUM_EXPERTS * bytes_per_expert
    total_gpu = torch.cuda.get_device_properties(0).total_memory
    return total > total_gpu * MOE_OFFLOAD_AUTO_RATIO


def clip_grad_mixed(parameters, max_norm: float) -> float:
    """跨设备梯度裁剪（offload 后 CPU/GPU 梯度并存，原生 clip_grad_norm_ 无法处理）。"""
    import math as _math
    acc = {}
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.detach()
        n2 = g.pow(2).sum()
        acc[g.device] = acc.get(g.device, torch.zeros((), device=g.device, dtype=torch.float32)) + n2.float()
    total_norm = _math.sqrt(sum(float(t.cpu()) for t in acc.values()))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in parameters:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef)
    return total_norm


def clip_grad_(parameters, max_norm: float):
    """梯度裁剪：单设备走原生 clip_grad_norm_，跨设备（offload）走 clip_grad_mixed。"""
    parameters = list(parameters)  # generator 转 list，避免二次迭代后变空
    devices = {p.grad.device for p in parameters if p.grad is not None}
    if len(devices) > 1:
        clip_grad_mixed(parameters, max_norm)
    else:
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)


class MoEFFN(nn.Module):
    """SwiGLU 专家 FFN + Top-K 软路由。forward 返回 (out, aux_loss)；step 供单 token 递归（严格等价）。

    专家权重组织为 [E, ...] 的 3D 张量；offload 时驻留 CPU，forward 中按命中专家搬移到 GPU。
    专家权重量化：'fp8_e4m3'（训练 STE）/ 'nvfp4'（推理 4bit）。"""

    _weight_names = ('w_gate', 'w_up', 'w_down')
    _bias_names = ('b_gate', 'b_up', 'b_down')

    def __init__(self, dim, num_experts, top_k, expert_hidden, offload=False, expert_quant=None):
        super().__init__()
        if expert_quant == 'nvfp4':
            offload = False
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_hidden = expert_hidden
        self.offload = offload
        self.expert_quant = expert_quant
        self.router = nn.Linear(dim, num_experts)
        dev = torch.device('cpu') if offload else None
        self.w_gate = nn.Parameter(torch.empty(num_experts, expert_hidden, dim, device=dev))
        self.w_up = nn.Parameter(torch.empty(num_experts, expert_hidden, dim, device=dev))
        self.w_down = nn.Parameter(torch.empty(num_experts, dim, expert_hidden, device=dev))
        self.b_gate = nn.Parameter(torch.zeros(num_experts, expert_hidden, device=dev))
        self.b_up = nn.Parameter(torch.zeros(num_experts, expert_hidden, device=dev))
        self.b_down = nn.Parameter(torch.zeros(num_experts, dim, device=dev))
        # weight_norm 幅度标量（每专家每矩阵一个，(E,)，offload 时同样驻 CPU）
        self.w_gate_scale = nn.Parameter(torch.ones(num_experts, device=dev))
        self.w_up_scale = nn.Parameter(torch.ones(num_experts, device=dev))
        self.w_down_scale = nn.Parameter(torch.ones(num_experts, device=dev))
        self._init_experts()
        if expert_quant == 'nvfp4':
            self.pack_nvfp4()

    def _init_experts(self):
        std = (self.dim * self.expert_hidden) ** -0.5
        for w in (self.w_gate, self.w_up, self.w_down):
            nn.init.normal_(w, std=std)
        for b in (self.b_gate, self.b_up, self.b_down):
            nn.init.zeros_(b)
        if MOE_EXPERT_WNORM:
            # 幅度标量初始化为各专家矩阵的 Frobenius 范数（逐专家，与取参时 wg.norm() 同一计算路径，
            #   保证首次 forward 与裸初始化严格等价，无浮点 roundoff 偏差）
            with torch.no_grad():
                for e in range(self.num_experts):
                    self.w_gate_scale[e] = self.w_gate[e].norm()
                    self.w_up_scale[e] = self.w_up[e].norm()
                    self.w_down_scale[e] = self.w_down[e].norm()

    def _apply(self, fn, recurse=True):
        # offload 时专家参数不随 model.to() 上 GPU（router 仍正常移动）
        if self.offload:
            self.router._apply(fn)
            return self
        return super()._apply(fn, recurse)

    # ---------- nvfp4 ----------
    def pack_nvfp4(self):
        if self.expert_quant != 'nvfp4':
            return
        for name in self._weight_names:
            p = getattr(self, name)
            if p is None:
                continue
            scale_p = getattr(self, name + '_scale') if MOE_EXPERT_WNORM else None
            for e in range(self.num_experts):
                w_e = p[e].detach()
                if scale_p is not None:
                    w_e = w_e * scale_p[e].detach()   # weight_norm 恢复实际权重后再量化
                packed, scales, shp = nvfp4_quantize(w_e, NVFP4_BLOCK_SIZE)
                self.register_buffer(f'{name}_q{e}_packed', packed)
                self.register_buffer(f'{name}_q{e}_scales', scales)
                self.register_buffer(f'{name}_q{e}_shape', torch.tensor(list(shp), dtype=torch.int64))
            setattr(self, name, None)   # 释放 fp32 主权重
            if scale_p is not None:
                setattr(self, name + '_scale', None)

    # ---------- 专家取参 ----------
    def _expert_mats(self, e, device):
        if self.expert_quant == 'nvfp4':
            outs = []
            for name in self._weight_names:
                packed = getattr(self, f'{name}_q{e}_packed')
                scales = getattr(self, f'{name}_q{e}_scales')
                shp = tuple(getattr(self, f'{name}_q{e}_shape').tolist())
                outs.append(nvfp4_dequantize(packed, scales, shp, NVFP4_BLOCK_SIZE).to(device))
            wg, wu, wd = outs
            bg = self.b_gate[e].to(device)
            bu = self.b_up[e].to(device)
            bd = self.b_down[e].to(device)
            return wg, wu, wd, bg, bu, bd
        if self.offload:
            wg = self.w_gate[e].to(device)
            wu = self.w_up[e].to(device)
            wd = self.w_down[e].to(device)
            bg = self.b_gate[e].to(device)
            bu = self.b_up[e].to(device)
            bd = self.b_down[e].to(device)
        else:
            wg = self.w_gate[e]; wu = self.w_up[e]; wd = self.w_down[e]
            bg = self.b_gate[e]; bu = self.b_up[e]; bd = self.b_down[e]
        if MOE_EXPERT_WNORM:
            # weight_norm：把权重范数归一化为可学习幅度，方向在球面上优化，范数恒等于 scale（结构性防发散）
            wg = wg * (self.w_gate_scale[e].to(device) / wg.norm().clamp_min(1e-8))
            wu = wu * (self.w_up_scale[e].to(device) / wu.norm().clamp_min(1e-8))
            wd = wd * (self.w_down_scale[e].to(device) / wd.norm().clamp_min(1e-8))
        if self.expert_quant == 'fp8_e4m3' and self.training:
            wg = wg + (e4m3_round_ste(wg) - wg).detach()
            wu = wu + (e4m3_round_ste(wu) - wu).detach()
            wd = wd + (e4m3_round_ste(wd) - wd).detach()
        return wg, wu, wd, bg, bu, bd

    # ---------- 路由 ----------
    def _router_topk(self, x):
        logits = self.router(x)                     # (N, E)
        probs = F.softmax(logits, dim=-1)
        topk_p, topk_i = probs.topk(self.top_k, dim=-1)
        topk_p = topk_p / topk_p.sum(-1, keepdim=True).clamp_min(1e-9)
        return topk_p, topk_i, probs

    def _aux_loss(self, probs, topk_i, N):
        E = self.num_experts
        onehot = F.one_hot(topk_i, num_classes=E).float()   # (N, k, E)
        f = onehot.sum(dim=1).sum(dim=0) / max(N, 1)        # (E,) 命中次数/N
        f = f / f.sum().clamp_min(1e-9)
        p = probs.mean(dim=0)                               # (E,)
        return E * (f.detach() * p).sum()

    def forward(self, x, return_aux=True):
        # x: (B, T, D)
        B, T, D = x.shape
        N = B * T
        xf = x.reshape(N, D)
        device = x.device
        topk_p, topk_i, probs = self._router_topk(xf)
        accum_idx, accum_val = [], []
        for e in range(self.num_experts):
            hit = (topk_i == e)
            if not bool(hit.any()):
                continue
            tok_idx, slot_idx = hit.nonzero(as_tuple=True)
            gp = topk_p[tok_idx, slot_idx]                 # (n,) 门控权重（可微）
            x_e = xf[tok_idx]                              # (n, D)
            wg, wu, wd, bg, bu, bd = self._expert_mats(e, device)
            h = F.silu(F.linear(x_e, wg, bg)) * F.linear(x_e, wu, bu)
            y_e = F.linear(h, wd, bd) * gp.unsqueeze(-1)
            accum_idx.append(tok_idx)
            accum_val.append(y_e)
        out = torch.zeros(xf.shape, device=device, dtype=xf.dtype)
        if accum_idx:
            idx = torch.cat(accum_idx).unsqueeze(-1).expand(-1, D)
            out = out.scatter_add(0, idx, torch.cat(accum_val))
        out = out.view(B, T, D)
        aux = self._aux_loss(probs, topk_i, N) if (self.training and return_aux and MOE_AUX_LOSS_COEF != 0.0) \
            else torch.zeros(1, device=device)
        return out, aux

    def step(self, x_t):
        """单 token（D,）-> （D,）。直接复用 forward（T=1），保证与 forward 严格等价。"""
        out, _ = self.forward(x_t.unsqueeze(0).unsqueeze(0), return_aux=False)
        return out[0, 0]


# ------------------------------------------------------------
# 主模型 WaveSNN
# ------------------------------------------------------------
class WaveSNN(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int,
                 pad_token_id: int = 0, use_residual: bool = True,
                 disable_coupling: bool = False,
                 freeze_seed: int = 42, freeze_layers: int = 0,
                 freeze_embedding: bool = False, seed_embedding: int = 42):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.use_residual = use_residual
        self.freeze_seed = freeze_seed
        self.freeze_layers = freeze_layers
        self.freeze_embedding = freeze_embedding
        self.seed_embedding = seed_embedding
        self.eos_token_id = None
        self.bos_token_id = None
        self.unk_token_id = None
        self.user_token_id = None
        self.assistant_token_id = None
        self.end_token_id = None

        self.input_scale = nn.Parameter(torch.tensor(2.0))

        # ---- 冻结层：用 freeze_seed 独立初始化 ----
        frozen_count = min(freeze_layers, num_layers)
        if frozen_count > 0:
            _cpu_state = torch.random.get_rng_state()
            _cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

            torch.manual_seed(freeze_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(freeze_seed)

            self.layers = nn.ModuleList()
            for i in range(frozen_count):
                self.layers.append(
                    ParallelComplexWaveLayer(hidden_size, hidden_size, state_dim=STATE_DIM,
                                             disable_coupling=disable_coupling)
                )

            torch.random.set_rng_state(_cpu_state)
            if _cuda_states is not None:
                torch.cuda.set_rng_state_all(_cuda_states)
        else:
            self.layers = nn.ModuleList()

        # ---- 可训练层：用全局种子初始化 ----
        for i in range(frozen_count, num_layers):
            self.layers.append(
                ParallelComplexWaveLayer(hidden_size, hidden_size, state_dim=STATE_DIM,
                                         disable_coupling=disable_coupling)
            )

        # 记录每层层号，供 forward 内 [Layer] 监控打印分层标识
        for _li, _layer in enumerate(self.layers):
            _layer.layer_idx = _li

        # 冻结区层梯度关闭
        for i in range(frozen_count):
            for p in self.layers[i].parameters():
                p.requires_grad = False

        # ---- 嵌入层：根据 freeze_embedding 控制种子和冻结 ----
        if freeze_embedding:
            _emb_cpu_state = torch.random.get_rng_state()
            _emb_cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

            torch.manual_seed(seed_embedding)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_embedding)

            self.embedding = nn.Embedding(vocab_size, hidden_size)
            nn.init.xavier_uniform_(self.embedding.weight, gain=1.0)

            for p in self.embedding.parameters():
                p.requires_grad = False

            torch.random.set_rng_state(_emb_cpu_state)
            if _emb_cuda_states is not None:
                torch.cuda.set_rng_state_all(_emb_cuda_states)
        else:
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            nn.init.xavier_uniform_(self.embedding.weight, gain=1.0)
        # =========================================================

        self.layer_norms = nn.ModuleList([RMSNorm(hidden_size) for _ in range(num_layers)])

        # ---- MoE-FFN（每层 SSM 后接 MoE；默认关闭） ----
        self.moe_enabled = MOE_ENABLED
        if self.moe_enabled:
            moe_offload = decide_moe_offload(hidden_size)
            self.moes = nn.ModuleList([
                MoEFFN(hidden_size, MOE_NUM_EXPERTS, MOE_TOP_K, MOE_EXPERT_HIDDEN,
                       offload=moe_offload, expert_quant=MOE_EXPERT_QUANT)
                for _ in range(num_layers)
            ])
            self.moe_norms = nn.ModuleList([RMSNorm(hidden_size) for _ in range(num_layers)])
        else:
            self.moes = nn.ModuleList()
            self.moe_norms = nn.ModuleList()

        self.output_norm = RMSNorm(hidden_size)
        self.output_fc = nn.Linear(hidden_size, vocab_size)
        if TIE_EMBEDDING_OUTPUT:
            if self.output_fc.weight.shape != self.embedding.weight.shape:
                raise ValueError("TIE_EMBEDDING_OUTPUT requires output_fc.weight and embedding.weight to have the same shape")
            self.output_fc.weight = self.embedding.weight
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        self.forbidden_ids = {pad_token_id}

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                return_per_sample_loss: bool = False,
                position_offset: Optional[int] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, float, List[float]]]:
        batch, seq_len = input_ids.shape if input_ids.numel() > 0 else (0, 0)
        if seq_len == 0:
            if labels is None:
                return torch.empty(0, 0, self.vocab_size, device=input_ids.device)
            else:
                if return_per_sample_loss:
                    return (torch.zeros(0, device=input_ids.device),
                            torch.zeros(0, device=input_ids.device),
                            0.0, [])
                else:
                    return torch.tensor(0.0, device=input_ids.device), 0.0, []

        device = input_ids.device

        if attention_mask is not None and attention_mask.shape[1] != seq_len:
            min_len = min(seq_len, attention_mask.shape[1])
            input_ids = input_ids[:, :min_len]
            if labels is not None:
                labels = labels[:, :min_len]
            attention_mask = attention_mask[:, :min_len]
            seq_len = min_len

        input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        token_emb = self.embedding(input_ids) * self.input_scale

        time_mask = attention_mask.float() if attention_mask is not None else None

        layer_input = token_emb
        layer_activities = []
        stab_loss_total = torch.zeros(1, device=device)
        moe_aux_total = torch.zeros(1, device=device)

        for i, layer in enumerate(self.layers):
            if USE_GRADIENT_CHECKPOINT and self.training:
                output_wave, stab_pen = checkpoint(
                    lambda l, x, m: l(x, time_mask=m),
                    layer, layer_input, time_mask, use_reentrant=True
                )
            else:
                output_wave, stab_pen = layer(layer_input, time_mask=time_mask)
            if labels is not None and self.training:
                stab_loss_total = stab_loss_total + stab_pen
            if self.use_residual:
                output_wave = output_wave + layer_input
            output_wave = self.layer_norms[i](output_wave)
            if self.moe_enabled:
                moe_out, moe_aux = self.moes[i](output_wave)
                if self.training and MOE_AUX_LOSS_COEF != 0.0:
                    moe_aux_total = moe_aux_total + moe_aux
                if MOE_RESIDUAL:
                    output_wave = output_wave + moe_out
                else:
                    output_wave = moe_out
                output_wave = self.moe_norms[i](output_wave)
            layer_activities.append(output_wave.mean().detach())  # 延迟 .item()：原每层一次 GPU→CPU 同步
            layer_input = output_wave

        norm_last = self.output_norm(layer_input)

        if labels is not None:
            labels_clamped = torch.where(labels != -100, torch.clamp(labels, 0, self.vocab_size-1), labels)
            mask = (labels != -100).float()
            chunk_size = int(LOSS_CHUNK_SIZE) if LOSS_CHUNK_SIZE else 0
            if chunk_size > 0 and seq_len > chunk_size:
                loss_sum_per_sample = torch.zeros(batch, dtype=norm_last.dtype, device=device)
                logit_scale = F.softplus(self.logit_scale)
                for start in range(0, seq_len, chunk_size):
                    end = min(start + chunk_size, seq_len)
                    logits_chunk = self.output_fc(norm_last[:, start:end, :]) * logit_scale
                    loss_chunk = F.cross_entropy(
                        logits_chunk.reshape(-1, self.vocab_size),
                        labels_clamped[:, start:end].reshape(-1),
                        ignore_index=-100,
                        reduction="none"
                    ).view(batch, -1)
                    loss_sum_per_sample = loss_sum_per_sample + (loss_chunk * mask[:, start:end]).sum(dim=1)
            else:
                logits = self.output_fc(norm_last) * F.softplus(self.logit_scale)
                loss_per_token = F.cross_entropy(logits.view(-1, self.vocab_size), labels_clamped.view(-1), ignore_index=-100, reduction="none")
                loss_per_token = loss_per_token.view(batch, -1)
                loss_sum_per_sample = (loss_per_token * mask).sum(dim=1)
            valid_tokens_per_sample = mask.sum(dim=1).clamp(min=1)
            avg_activity = (sum(layer_activities) / len(layer_activities)).item() if layer_activities else 0.0

            if return_per_sample_loss:
                if self.training:
                    stab_per_sample = STABILITY_LAMBDA * stab_loss_total / batch
                    loss_sum_per_sample = loss_sum_per_sample + stab_per_sample
                return loss_sum_per_sample, valid_tokens_per_sample, avg_activity, layer_activities
            else:
                loss = loss_sum_per_sample.sum() / valid_tokens_per_sample.sum()
                if self.training:
                    loss = loss + STABILITY_LAMBDA * stab_loss_total
                    if self.moe_enabled:
                        loss = loss + MOE_AUX_LOSS_COEF * moe_aux_total
                return loss, avg_activity, layer_activities
        logits = self.output_fc(norm_last) * F.softplus(self.logit_scale)
        return logits

    def init_states(self, device):
        return [layer.init_state(device) for layer in self.layers]

    def has_offloaded_params(self) -> bool:
        return any(getattr(m, 'offload', False) for m in self.moes) if self.moe_enabled else False

    def pack_moe_nvfp4(self):
        if not self.moe_enabled:
            return
        for m in self.moes:
            m.pack_nvfp4()

    def step(self, token_id, states):
        """单 token 生成步：token_id (int) -> logits (1, vocab)。与 forward 逐时间步等价。"""
        device = states[0]['h'].device
        emb = self.embedding(torch.tensor([[token_id]], dtype=torch.long, device=device)) * self.input_scale
        x_t = emb[0, 0]
        for i, layer in enumerate(self.layers):
            y_t = layer.step(x_t, states[i])
            if self.use_residual:
                y_t = y_t + x_t
            y_t = self.layer_norms[i](y_t)
            if self.moe_enabled:
                moe_out = self.moes[i].step(y_t)
                if MOE_RESIDUAL:
                    y_t = y_t + moe_out
                else:
                    y_t = moe_out
                y_t = self.moe_norms[i](y_t)
            x_t = y_t
        norm_last = self.output_norm(x_t)
        logits = self.output_fc(norm_last.unsqueeze(0)) * F.softplus(self.logit_scale)
        return logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = MAX_SEQ_LEN,
                 temperature: float = 0.7, top_p: float = 0.9, stream=None) -> List[int]:
        device = input_ids.device
        batch_size = input_ids.shape[0]
        if batch_size != 1:
            raise ValueError("生成时仅支持 batch_size = 1")

        stop_tokens = set()
        if self.eos_token_id is not None:
            stop_tokens.add(self.eos_token_id)
        if self.end_token_id is not None:
            stop_tokens.add(self.end_token_id)

        forbidden = self.forbidden_ids.copy()
        if self.pad_token_id is not None:
            forbidden.add(self.pad_token_id)

        # 递归推理：初始化状态，逐步跑完 prompt（O(prompt_len)），再逐 token 增量生成（O(1)/步）
        states = self.init_states(device)
        prompt_ids = input_ids[0].tolist()

        amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if USE_AMP and device.type == "cuda" else contextlib.nullcontext()

        logits = None
        with amp_ctx:
            for tid in prompt_ids:
                logits = self.step(tid, states)

        generated = []
        for _ in range(max_new_tokens):
            last_logits = logits[0]
            probs = F.softmax(last_logits / temperature, dim=-1)

            for fid in forbidden:
                if fid < self.vocab_size:
                    probs[fid] = 0.0

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum > top_p
            mask[0] = False
            sorted_probs[mask] = 0.0

            if sorted_probs.sum() == 0:
                uniform = torch.ones(self.vocab_size, device=device) / self.vocab_size
                for fid in forbidden:
                    if fid < self.vocab_size:
                        uniform[fid] = 0.0
                if uniform.sum() == 0:
                    uniform = torch.ones(self.vocab_size, device=device) / self.vocab_size
                else:
                    uniform = uniform / uniform.sum()
                sorted_probs, sorted_indices = torch.sort(uniform, descending=True)
            else:
                sorted_probs = sorted_probs / sorted_probs.sum()

            next_token = sorted_indices[torch.multinomial(sorted_probs, 1).item()].item()

            generated.append(next_token)
            if stream is not None:
                stream(next_token)
            if next_token in stop_tokens:
                break

            with amp_ctx:
                logits = self.step(next_token, states)

        return generated

# ------------------------------------------------------------
# 保存 checkpoint（只保存模型权重，不保存词表）
# ------------------------------------------------------------
def get_next_checkpoint_id() -> str:
    timestamp = int(time.time() * 1000)
    rand = random.randint(0, 10000)
    return f"{timestamp}_{rand}"

def save_checkpoint(model, epoch, custom_filename=None):
    if custom_filename is None:
        ckpt_id = get_next_checkpoint_id()
        filename = f"checkpoint_{ckpt_id}.pt"
    else:
        filename = custom_filename
    save_dict = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch}
    torch.save(save_dict, filename)
    log_and_print(f"保存 checkpoint 至 {filename} (epoch {epoch})")
    return filename

# ------------------------------------------------------------
# DataLoader 工厂
# ------------------------------------------------------------
def create_dataloader(dataset, batch_size, shuffle, collate_fn, num_workers=0, pin_memory=PIN_MEMORY, drop_last=False):
    g = torch.Generator()
    g.manual_seed(42)
    def worker_init_fn(worker_id):
        random.seed(42 + worker_id)
        np.random.seed(42 + worker_id)
        torch.manual_seed(42 + worker_id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
        generator=g
    )

# ------------------------------------------------------------
# 统一设置模型特殊 token 信息
# ------------------------------------------------------------
def setup_model_token_ids(model: WaveSNN, tokenizer: FastBPE):
    model.eos_token_id = tokenizer.eos_token_id
    model.bos_token_id = tokenizer.bos_token_id
    model.unk_token_id = tokenizer.unk_token_id
    model.user_token_id = tokenizer.user_token_id
    model.assistant_token_id = tokenizer.assistant_token_id
    model.pad_token_id = tokenizer.pad_token_id
    model.end_token_id = tokenizer.end_token_id
    model.forbidden_ids = {tokenizer.pad_token_id} if tokenizer.pad_token_id is not None else set()

# ------------------------------------------------------------
# 文件选择辅助函数
# ------------------------------------------------------------
def select_training_file():
    """扫描当前目录下 .txt 和 .jsonl 文件，让用户选择数据集文件。"""
    files = []
    for f in sorted(os.listdir('.')):
        if f.startswith('.'):
            continue
        if f.endswith('.txt') or f.endswith('.jsonl'):
            try:
                size = os.path.getsize(f)
            except OSError:
                size = 0
            files.append((f, size))
    if not files:
        log_and_print("当前目录下未找到 .txt 或 .jsonl 文件。")
        return None, None
    print("\n可用数据集文件：")
    for i, (name, size) in enumerate(files, 1):
        if size >= 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.1f} MB"
        elif size >= 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"  {i}. {name} ({size_str})")
    while True:
        raw = input("请输入文件序号: ").strip()
        try:
            idx = int(raw)
        except ValueError:
            print("输入无效，请输入数字。")
            continue
        if idx < 1 or idx > len(files):
            print(f"序号超出范围，请输入 1-{len(files)}。")
            continue
        selected, _ = files[idx - 1]
        ext = os.path.splitext(selected)[1].lower()
        log_and_print(f"已选择文件: {selected}")
        return selected, ext

# ------------------------------------------------------------
# 训练函数
# ------------------------------------------------------------
def train_from_scratch(model: WaveSNN, tokenizer: FastBPE,
                       data_path: str, batch_size: int, epochs: int, lr: float,
                       weight_decay: float, grad_clip: float, log_interval: int,
                       accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
                       file_type: str = 'jsonl'):
    device = DEVICE
    model.to(device)
    log_and_print(f"对角复数波架构：开始全序列训练。数据集: {data_path}")

    try:
        import psutil
        process = psutil.Process(os.getpid())
        _has_psutil = True
    except ImportError:
        process = None
        _has_psutil = False
        log_and_print("psutil 未安装，跳过内存监控。")

    with open("training_log.txt", 'a', encoding='utf-8') as f:
        f.write(f"--- 从头训练开始 ({data_path}) ---\n")

    setup_model_token_ids(model, tokenizer)

    collate_fn = make_collate_fn(tokenizer.pad_token_id)
    if file_type == '.txt':
        train_dataset = TextFileDataset(data_path, tokenizer, max_seq_len=MAX_SEQ_LEN,
                                        epoch=0, is_val=False, window_overlap=WINDOW_OVERLAP)
        train_dataset._ensure_tokenized()
        total_samples = train_dataset._num_windows
        est_batches_per_epoch = max(1, total_samples // batch_size) if total_samples > 0 else 1
        val_loader = None
        log_and_print(f"txt 流式模式：{total_samples} 个训练窗口，不验证不早停。")
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            total_samples = sum(1 for l in f if l.strip())
        est_batches_per_epoch = max(1, total_samples // batch_size) if total_samples > 0 else 1000

        val_indices = None
        train_indices = None
        if total_samples > 0:
            all_indices = list(range(total_samples))
            val_count = total_samples // 10
            val_indices = set(random.Random(42).sample(all_indices, val_count))
            train_indices = set(all_indices) - val_indices
            log_and_print(f"总样本 {total_samples}，验证集 {val_count} 条（种子 42），训练集 {total_samples - val_count} 条，每 epoch 约 {est_batches_per_epoch} batch")
        else:
            log_and_print("未输入样本数，跳过验证集划分。")
        train_dataset = StreamingJSONLDataset(
            file_path=data_path, tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN,
            epoch=0, is_val=False,
            assigned_indices=train_indices
        )
        val_loader = None
        if val_indices:
            val_dataset = StreamingJSONLDataset(
                file_path=data_path, tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN,
                is_val=True, assigned_indices=val_indices
            )
            val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False,
                                           collate_fn=collate_fn, num_workers=0,
                                           pin_memory=PIN_MEMORY, drop_last=False)

    num_workers = 0
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=False,
                                     collate_fn=collate_fn, num_workers=num_workers,
                                     pin_memory=PIN_MEMORY, drop_last=True)

    no_decay = {'bias', 'logit_scale', 'input_scale', 'input_gain', 'h_ema_beta', 'delta_scale', 'B_scale', 'original0',
            'w_gate_scale', 'w_up_scale', 'w_down_scale', 'theta_raw'}
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if n.split('.')[-1] not in no_decay and p.requires_grad and p.numel() > 0],
         'weight_decay': weight_decay, 'lr': lr},
        {'params': [p for n, p in model.named_parameters() if n.split('.')[-1] in no_decay and p.requires_grad and p.numel() > 0],
         'weight_decay': 0.0, 'lr': lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    total_updates_per_epoch = max(1, (est_batches_per_epoch + accumulation_steps - 1) // accumulation_steps)
    total_updates = total_updates_per_epoch * epochs
    warmup_steps = min(2000, int(total_updates * 0.02))
    decay_steps = max(1, total_updates - warmup_steps) if total_updates > 0 else None
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=warmup_steps, decay_steps=decay_steps, eta_min=0.1)
    log_and_print(f"训练总参数更新次数 = {total_updates}, 预热步数 = {warmup_steps}, 初始学习率 = {lr}")

    best_val_ppl = float('inf')
    patience_counter = 0

    optimizer_steps = 0
    cleanup_interval = 10

    amp_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if USE_AMP and device.type == 'cuda' else contextlib.nullcontext()

    interrupted = False
    try:
        for epoch in range(epochs):
            if hasattr(train_dataset, 'epoch'):
                train_dataset.epoch = epoch
            model.train()
            total_loss = 0.0
            total_activity = 0.0
            optimizer.zero_grad()
            gradients_counter = 0
            step_loss_sum = 0.0
            step_count = 0

            consumed_samples = 0

            for step, batch in enumerate(train_loader):
                input_ids, labels, attn_mask = batch
                if input_ids.numel() == 0:
                    continue
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                attn_mask = attn_mask.to(device)

                with amp_ctx:
                    loss, activity, layer_activities = model(input_ids, labels,
                                                         position_offset=0, attention_mask=attn_mask)
                if torch.isnan(loss):
                    log_and_print("NaN loss detected, skipping batch")
                    continue

                loss = loss / accumulation_steps
                loss.backward()
                gradients_counter += 1
                total_loss += loss.item() * accumulation_steps
                total_activity += activity

                if gradients_counter % accumulation_steps == 0:
                    clip_grad_(model.parameters(), grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    optimizer_steps += 1
                    if optimizer_steps % cleanup_interval == 0:
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        else:
                            import gc
                            gc.collect()

                step_loss_sum += loss.item() * accumulation_steps
                step_count += 1
                consumed_samples += input_ids.shape[0]

                if (step + 1) % log_interval == 0:
                    avg_loss = step_loss_sum / step_count if step_count > 0 else 0.0
                    avg_ppl = math.exp(avg_loss) if avg_loss < 50 else float('inf')
                    avg_act = total_activity / step_count if step_count > 0 else 0.0
                    current_lr = scheduler.get_last_lr()[0]
                    mem_str = ""
                    if _has_psutil and process is not None:
                        try:
                            mem_mb = process.memory_info().rss / 1024**2
                            mem_str = f" Memory: {mem_mb:.1f} MB"
                        except:
                            mem_str = ""
                    pct_str = ""
                    if total_samples > 0:
                        pct_str = f" ({consumed_samples * 100 / total_samples:.1f}%)"
                    log_and_print(f"Epoch {epoch+1}/{epochs} Step {step+1} Loss: {avg_loss:.4f} PPL: {avg_ppl:.2f} "
                                  f"Activity: {avg_act:.4f} LR: {current_lr:.6f} "
                                  f"Samples: {consumed_samples}{pct_str}{mem_str}")
                    # --- layer 状态监控（与 MON 同点打印，避免 forward 内 .item() 同步打断 GPU）---
                    for li, layer in enumerate(model.layers):
                        mon = getattr(layer, "_last_mon", None)
                        if mon is not None:
                            log_and_print(f"  [Layer L{li}] I_std={mon[0].item():.4f} amp_std={mon[1].item():.4f} wave_raw_mean={mon[2].item():.4f} wave_raw_std={mon[3].item():.4f} delta_mean={mon[4].item():.4f} delta_min={mon[5].item():.4f} delta_max={mon[6].item():.4f}")

                    # --- maxp param monitor（含 MoE 专家参数，统一在此打印；耦合强度见 Mn）---
                    for li, layer in enumerate(model.layers):
                        bn = layer.B_linear.weight.norm().item()
                        cn = (layer.summary_linear.weight.norm().item()
                             if hasattr(layer, 'summary_linear')
                             else layer.global_mod_linear.weight.norm().item())
                        dn = layer.delta_linear.weight.norm().item()
                        mn = torch.exp(layer.coupling_log).norm().item()
                        aw = torch.exp(layer.theta_raw).norm().item()
                        nu = torch.exp(layer.nu_log)
                        th = torch.exp(layer.theta_raw)
                        betan = layer.h_ema_beta.item() if hasattr(layer, 'h_ema_beta') else 0.0
                        cbnr = layer.C_base_real.norm().item()
                        cbni = layer.C_base_imag.norm().item()
                        cbn = cbnr + cbni
                        innn = layer.input_mod_linear.weight.norm().item()
                        log_and_print(f"  [MON L{li}] Bn={bn:.3f} Sn={cn:.3f} Dn={dn:.3f} Mn={mn:.3f} An={aw:.3f} nu={nu.mean():.3f}/{nu.max():.3f} th={th.min():.3f}/{th.max():.3f} betan={betan:.6f} Cbn={cbn:.3f} In={innn:.3f}")
                        if model.moe_enabled:
                            moe = model.moes[li]
                            rn = moe.router.weight.norm().item()
                            if MOE_EXPERT_WNORM:
                                # weight_norm 后每专家实际范数 = 幅度标量 scale（方向恒单位球面）
                                gn = moe.w_gate_scale.mean().item()
                                un = moe.w_up_scale.mean().item()
                                wdn = moe.w_down_scale.mean().item()
                            else:
                                gn = moe.w_gate.norm().item()
                                un = moe.w_up.norm().item()
                                wdn = moe.w_down.norm().item()
                            log_and_print(f"  [MOE L{li}] Rr={rn:.3f} Wg={gn:.3f} Wu={un:.3f} Wd={wdn:.3f}")

                    step_loss_sum = 0.0
                    step_count = 0
                    total_activity = 0.0

            if gradients_counter % accumulation_steps != 0:
                clip_grad_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                optimizer_steps += 1
                if optimizer_steps % cleanup_interval == 0:
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    else:
                        import gc
                        gc.collect()

            if device.type == 'cuda':
                torch.cuda.empty_cache()
            else:
                import gc
                gc.collect()

            if val_loader is not None:
                        model.eval()
                        val_total_loss = 0.0
                        val_total_tokens = 0
                        with torch.no_grad():
                                    for val_batch in val_loader:
                                                val_inputs, val_labels, val_attn_mask = val_batch
                                                if val_inputs.numel() == 0:
                                                    continue
                                                val_inputs = val_inputs.to(device)
                                                val_labels = val_labels.to(device)
                                                val_attn_mask = val_attn_mask.to(device)
                                                val_loss_sum, valid_tokens, _, _ = model(
                                                    val_inputs, val_labels,
                                                    position_offset=0, attention_mask=val_attn_mask,
                                                    return_per_sample_loss=True
                                                )
                                                val_total_loss += val_loss_sum.sum().item()
                                                val_total_tokens += valid_tokens.sum().item()
                        avg_val_loss = val_total_loss / val_total_tokens if val_total_tokens > 0 else 10.0
                        val_ppl = math.exp(avg_val_loss) if avg_val_loss < 50 else float('inf')
                        log_and_print(f"Epoch {epoch+1} 验证 Loss: {avg_val_loss:.4f} PPL: {val_ppl:.2f}")

                        with open("training_log.txt", 'a', encoding='utf-8') as f:
                                    f.write(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | Val PPL: {val_ppl:.2f}\n")

                        if val_ppl < best_val_ppl:
                                    best_val_ppl = val_ppl
                                    patience_counter = 0
                                    best_filename = f"best_epoch{epoch+1}_ppl{val_ppl:.2f}.pt"
                                    save_checkpoint(model, epoch+1, custom_filename=best_filename)
                                    log_and_print(f"保存最佳模型至 {best_filename}")
                        else:
                                    patience_counter += 1
                                    if patience_counter >= EARLY_STOP_PATIENCE:
                                                log_and_print(f"早停触发，停止训练")
                                                break
            else:
                        log_and_print(f"Epoch {epoch+1} 无验证集，跳过验证与早停。")
            ckpt_name = f"checkpoint_epoch{epoch+1}.pt"
            save_checkpoint(model, epoch+1, custom_filename=ckpt_name)

            # 重置日志计数器，防止跨 epoch 累积
            step_loss_sum = 0.0
            step_count = 0
            total_activity = 0.0
    except KeyboardInterrupt:
        interrupted = True
        log_and_print("\n训练被用户中断 (Ctrl+C)，正在保存当前模型...")
        save_checkpoint(model, epoch + 1, custom_filename="interrupted_model.pt")
        log_and_print("模型已保存至 interrupted_model.pt，返回主菜单。")
        return

    if not interrupted:
        log_and_print("从头训练完成")

# ------------------------------------------------------------
# 对话循环
# ------------------------------------------------------------
def chat_loop(model: WaveSNN, tokenizer: FastBPE, mode: str = "jsonl"):
    device = DEVICE
    model.to(device)
    model.eval()
    setup_model_token_ids(model, tokenizer)

    MAX_HISTORY_TOKENS = MAX_SEQ_LEN

    if mode == "txt":
        log_and_print("进入纯文本续写模式（txt），输入 exit 退出，reset 重置上下文。")
        context_ids = []
        while True:
            user_input = input("\n输入: ")
            if user_input.lower() == 'exit':
                break
            if user_input.lower() == 'reset':
                context_ids = []
                log_and_print("上下文已重置。")
                continue
            new_ids = tokenizer.encode(user_input)
            context_ids.extend(new_ids)
            if len(context_ids) > MAX_HISTORY_TOKENS:
                context_ids = context_ids[-MAX_HISTORY_TOKENS:]
            input_tensor = torch.tensor([context_ids], dtype=torch.long, device=device)
            generated_ids = []
            def stream_token(token_id):
                generated_ids.append(token_id)
                text = tokenizer.decode([token_id])
                if text:
                    print(text, end="", flush=True)
            model.generate(input_tensor, max_new_tokens=MAX_SEQ_LEN,
                           temperature=TEMPERATURE, top_p=TOP_P, stream=stream_token)
            context_ids.extend(generated_ids)
            if len(context_ids) > MAX_HISTORY_TOKENS:
                context_ids = context_ids[-MAX_HISTORY_TOKENS:]
            print()
        print()
    else:
        log_and_print("进入对话模式（jsonl），输入 exit 退出，reset 重置上下文。")
        history_turns = []
        while True:
            user_input = input("\n用户: ")
            if user_input.lower() == 'exit':
                break
            if user_input.lower() == 'reset':
                history_turns = []
                log_and_print("上下文已重置。")
                continue
            user_ids = [tokenizer.user_token_id] + tokenizer.encode(user_input) + [tokenizer.end_token_id]
            max_input_len = 10000000 - 200  # 无上限
            max_history_len = max_input_len - (1 + len(user_ids) + 1)
            history_len = sum(len(t) for t in history_turns)
            while history_len > max_history_len and len(history_turns) > 1:
                history_len -= len(history_turns.pop(0))
            flat_history = [tok for turn in history_turns for tok in turn]
            full_input = [tokenizer.bos_token_id] + flat_history + user_ids + [tokenizer.assistant_token_id]
            input_tensor = torch.tensor([full_input], dtype=torch.long, device=device)
            generated_ids = []
            print("助手: ", end="", flush=True)
            def stream_token_jsonl(token_id):
                generated_ids.append(token_id)
                text = tokenizer.decode([token_id])
                if text:
                    print(text, end="", flush=True)
            model.generate(input_tensor, max_new_tokens=MAX_SEQ_LEN,
                           temperature=TEMPERATURE, top_p=TOP_P, stream=stream_token_jsonl)
            print()
            history_turns.append(user_ids)
            history_turns.append([tokenizer.assistant_token_id] + generated_ids)
            total = sum(len(t) for t in history_turns)
            while total > MAX_HISTORY_TOKENS and len(history_turns) > 1:
                total -= len(history_turns.pop(0))

# ------------------------------------------------------------
# Ultra: rule-verified rejection sampling and memory-friendly GRPO
# ------------------------------------------------------------
def ultra_read_int(prompt: str, default: int, minimum: int = 1) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
            if value >= minimum:
                return value
        except ValueError:
            pass
        print(f"请输入不小于 {minimum} 的整数。")


def ultra_read_float(prompt: str, default: float, minimum: float = 0.0) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
            if value >= minimum:
                return value
        except ValueError:
            pass
        print(f"请输入不小于 {minimum} 的数字。")


def ultra_select_problem_file():
    files = []
    for name in sorted(os.listdir('.')):
        if name.startswith('.') or not name.endswith('.jsonl'):
            continue
        try:
            size = os.path.getsize(name)
        except OSError:
            size = 0
        files.append((name, size))
    if not files:
        log_and_print("当前目录没有可用的 .jsonl 题库。")
        return None
    print("\n可用 JSONL 题库：")
    for i, (name, size) in enumerate(files, 1):
        print(f"  {i}. {name} ({size / 1024:.1f} KB)")
    while True:
        raw = input("请输入题库序号: ").strip()
        try:
            index = int(raw)
            if 1 <= index <= len(files):
                return files[index - 1][0]
        except ValueError:
            pass
        print(f"请输入 1-{len(files)}。")


def ultra_iter_problems(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                log_and_print(f"跳过题库第 {line_no} 行：JSON 解析失败: {e}")
                continue
            question = item.get('question', item.get('prompt'))
            answers = item.get('answers', item.get('answer'))
            if isinstance(question, list):
                question = '\n'.join(str(turn.get('content', turn)) if isinstance(turn, dict) else str(turn)
                                     for turn in question)
            if not isinstance(question, str) or answers is None:
                log_and_print(f"跳过题库第 {line_no} 行：必须包含 question/prompt 和 answer/answers。")
                continue
            if not isinstance(answers, list):
                answers = [answers]
            answers = [str(answer) for answer in answers if str(answer).strip()]
            if not answers:
                log_and_print(f"跳过题库第 {line_no} 行：没有可用答案。")
                continue
            yield {
                'question': question,
                'answers': answers,
                'type': str(item.get('type', 'auto')).lower(),
                'line_no': line_no,
            }


def ultra_count_problems(path: str) -> int:
    n = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def ultra_iter_problems_shuffled(path: str, buffer_size: int = 10000, seed: int = 0):
    rng = random.Random(seed)
    buf = []
    for p in ultra_iter_problems(path):
        buf.append(p)
        if len(buf) >= buffer_size:
            rng.shuffle(buf)
            while buf:
                yield buf.pop()
    if buf:
        rng.shuffle(buf)
        while buf:
            yield buf.pop()


def ultra_infer_answer_type(problem) -> str:
    import re
    answer_type = problem.get('type', 'auto')
    if answer_type != 'auto':
        return answer_type
    answers = [answer.strip() for answer in problem['answers']]
    if answers and all(re.fullmatch(r'[A-Za-z]', answer) for answer in answers):
        return 'choice'
    if answers and all(re.fullmatch(r'[-+]?\d+(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?', answer.replace(',', ''))
                       for answer in answers):
        return 'number'
    return 'text'


def ultra_normalize_answer(value: str, answer_type: str) -> str:
    import re
    import unicodedata
    from decimal import Decimal, InvalidOperation
    from fractions import Fraction

    text = unicodedata.normalize('NFKC', str(value)).strip()
    if answer_type == 'choice':
        choices = re.findall(r'(?<![A-Za-z])[A-Za-z](?![A-Za-z])', text)
        return choices[-1].upper() if choices else text.upper()
    if answer_type == 'number':
        matches = re.findall(r'[-+]?\d+(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?', text.replace(',', ''))
        if matches:
            raw = matches[-1]
            try:
                if '/' in raw:
                    left, right = raw.split('/', 1)
                    value_fraction = Fraction(Decimal(left)) / Fraction(Decimal(right))
                else:
                    value_fraction = Fraction(Decimal(raw))
                return f"{value_fraction.numerator}/{value_fraction.denominator}"
            except (InvalidOperation, ZeroDivisionError):
                pass
    text = re.sub(r'\s+', '', text.lower())
    return text.strip('。.!！?？,，;；:：')


def ultra_extract_final_answer(response: str) -> str:
    import re
    boxed = re.findall(r'\\boxed\s*\{([^{}]+)\}', response)
    if boxed:
        return boxed[-1].strip()
    marked = re.findall(r'(?:最终答案|答案|final\s+answer)\s*[:：]\s*([^\r\n]+)', response,
                        flags=re.IGNORECASE)
    if marked:
        return marked[-1].strip()
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return lines[-1] if lines else ''


def ultra_rule_reward(response: str, problem):
    predicted = ultra_extract_final_answer(response)
    answer_type = ultra_infer_answer_type(problem)
    predicted_normalized = ultra_normalize_answer(predicted, answer_type)
    accepted = {ultra_normalize_answer(answer, answer_type) for answer in problem['answers']}
    reward = 1.0 if predicted_normalized in accepted else 0.0
    return reward, predicted


def ultra_make_prompt(problem) -> str:
    return (f"{problem['question']}\n\n"
            "请自由推理。最后必须单独写一行：最终答案：你的答案")


def ultra_prompt_ids(tokenizer: FastBPE, prompt: str):
    return ([tokenizer.bos_token_id, tokenizer.user_token_id]
            + tokenizer.encode(prompt)
            + [tokenizer.end_token_id, tokenizer.assistant_token_id])


def ultra_generate_one(model: WaveSNN, tokenizer: FastBPE, prompt_ids, max_new_tokens: int):
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    generated_ids = model.generate(input_tensor, max_new_tokens=max_new_tokens,
                                   temperature=TEMPERATURE, top_p=TOP_P)
    return generated_ids, tokenizer.decode(generated_ids)


def ultra_write_sft_line(f, prompt: str, response: str):
    record = {'conversations': [
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': response},
    ]}
    f.write(json.dumps(record, ensure_ascii=False) + '\n')
    f.flush()


def ultra_rejection_sampling(model: WaveSNN, tokenizer: FastBPE, problem_path: str,
                             output_path: str, samples_per_question: int,
                             max_new_tokens: int):
    if os.path.abspath(problem_path) == os.path.abspath(output_path):
        raise ValueError("输出文件不能覆盖输入题库。")
    total_problems = ultra_count_problems(problem_path)
    if total_problems == 0:
        log_and_print("题库没有可用题目。")
        return
    was_training = model.training
    model.to(DEVICE)
    model.eval()
    kept = 0
    tried = 0
    seen = set()
    with open(output_path, 'w', encoding='utf-8') as out:
        with torch.no_grad():
            for index, problem in enumerate(ultra_iter_problems(problem_path), 1):
                prompt = ultra_make_prompt(problem)
                prompt_ids = ultra_prompt_ids(tokenizer, prompt)
                local_kept = 0
                for _ in range(samples_per_question):
                    generated_ids, response = ultra_generate_one(model, tokenizer, prompt_ids, max_new_tokens)
                    tried += 1
                    reward, predicted = ultra_rule_reward(response, problem)
                    key = (problem['line_no'], response)
                    if reward > 0 and key not in seen:
                        ultra_write_sft_line(out, prompt, response)
                        seen.add(key)
                        kept += 1
                        local_kept += 1
                log_and_print(f"[RS {index}/{total_problems}] 正确保留 {local_kept}/{samples_per_question}")
    if was_training:
        model.train()
    log_and_print(f"拒绝采样完成：尝试 {tried} 个回答，保留 {kept} 个正确回答，输出至 {output_path}")


def ultra_completion_log_probs(model: WaveSNN, prompt_ids, completion_ids):
    if not completion_ids:
        return torch.empty(0, device=DEVICE)
    full_ids = prompt_ids + completion_ids
    input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=DEVICE)
    attention_mask = torch.ones_like(input_ids)
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if USE_AMP and DEVICE.type == "cuda" else contextlib.nullcontext()
    with amp_ctx:
        logits = model(input_ids, position_offset=0, attention_mask=attention_mask)
    start = len(prompt_ids) - 1
    selected_logits = logits[:, start:start + len(completion_ids), :]
    targets = torch.tensor(completion_ids, dtype=torch.long, device=DEVICE).view(1, -1, 1)
    return F.log_softmax(selected_logits, dim=-1).gather(-1, targets).squeeze(0).squeeze(-1)


def ultra_save_model_only(model: WaveSNN, filename: str, epoch: int):
    torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, filename)
    log_and_print(f"仅保存模型至 {filename}")


def ultra_grpo_train(model: WaveSNN, tokenizer: FastBPE, problem_path: str,
                     positive_output_path: str, checkpoint_prefix: str,
                     epochs: int, group_size: int, max_new_tokens: int,
                     lr: float, inner_steps: int, clip_epsilon: float,
                     old_policy_penalty: float):
    if os.path.abspath(problem_path) == os.path.abspath(positive_output_path):
        raise ValueError("GRPO 正确样本输出文件不能覆盖输入题库。")
    total_problems = ultra_count_problems(problem_path)
    if total_problems == 0:
        log_and_print("题库没有可用题目。")
        return
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    model.to(DEVICE)
    seen_positive = set()
    total_updates = 0
    with open(positive_output_path, 'w', encoding='utf-8') as positive_out:
        for epoch in range(epochs):
            for index, problem in enumerate(ultra_iter_problems_shuffled(problem_path, buffer_size=10000, seed=epoch), 1):
                prompt = ultra_make_prompt(problem)
                prompt_ids = ultra_prompt_ids(tokenizer, prompt)
                completions = []
                responses = []
                rewards = []
                model.eval()
                with torch.no_grad():
                    for _ in range(group_size):
                        completion_ids, response = ultra_generate_one(model, tokenizer, prompt_ids, max_new_tokens)
                        reward, predicted = ultra_rule_reward(response, problem)
                        completions.append(completion_ids)
                        responses.append(response)
                        rewards.append(reward)
                        key = (problem['line_no'], response)
                        if reward > 0 and key not in seen_positive:
                            ultra_write_sft_line(positive_out, prompt, response)
                            seen_positive.add(key)
                reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
                reward_mean = reward_tensor.mean()
                reward_std = reward_tensor.std(unbiased=False)
                if reward_std.item() < 1e-8:
                    log_and_print(f"[GRPO E{epoch+1} {index}/{total_problems}] reward={reward_mean.item():.3f} 全组同分，跳过更新")
                    continue
                advantages = (reward_tensor - reward_mean) / (reward_std + 1e-8)
                model.eval()
                with torch.no_grad():
                    old_log_probs = [ultra_completion_log_probs(model, prompt_ids, completion).detach()
                                     for completion in completions]
                for _ in range(inner_steps):
                    model.train()
                    optimizer.zero_grad()
                    losses = []
                    for completion, old_log_prob, advantage in zip(completions, old_log_probs, advantages):
                        if not completion:
                            continue
                        current_log_prob = ultra_completion_log_probs(model, prompt_ids, completion)
                        ratio = torch.exp(current_log_prob - old_log_prob)
                        clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
                        surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
                        anchor_penalty = (current_log_prob - old_log_prob).pow(2).mean()
                        losses.append(-surrogate.mean() + old_policy_penalty * anchor_penalty)
                    if not losses:
                        break
                    loss = torch.stack(losses).mean()
                    loss.backward()
                    clip_grad_(trainable, GRAD_CLIP_NORM)
                    optimizer.step()
                    total_updates += 1
                log_and_print(f"[GRPO E{epoch+1} {index}/{total_problems}] rewards={rewards} updates={total_updates}")
            ultra_save_model_only(model, f"{checkpoint_prefix}_epoch{epoch+1}.pt", epoch + 1)
    log_and_print(f"GRPO 完成：共更新 {total_updates} 次；正确 rollout 已输出至 {positive_output_path}")


def ultra_rule_rl_train(model: WaveSNN, tokenizer: FastBPE, problem_path: str,
                        positive_output_path: str, checkpoint_prefix: str,
                        epochs: int, max_new_tokens: int, lr: float,
                        baseline_momentum: float):
    if os.path.abspath(problem_path) == os.path.abspath(positive_output_path):
        raise ValueError("基础 RL 正确样本输出文件不能覆盖输入题库。")
    total_problems = ultra_count_problems(problem_path)
    if total_problems == 0:
        log_and_print("题库没有可用题目。")
        return
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    model.to(DEVICE)
    baseline = 0.5
    total_updates = 0
    seen_positive = set()
    with open(positive_output_path, 'w', encoding='utf-8') as positive_out:
        for epoch in range(epochs):
            for index, problem in enumerate(ultra_iter_problems_shuffled(problem_path, buffer_size=10000, seed=epoch), 1):
                prompt = ultra_make_prompt(problem)
                prompt_ids = ultra_prompt_ids(tokenizer, prompt)
                model.eval()
                with torch.no_grad():
                    completion_ids, response = ultra_generate_one(model, tokenizer, prompt_ids, max_new_tokens)
                    reward, predicted = ultra_rule_reward(response, problem)
                advantage = reward - baseline
                baseline = baseline_momentum * baseline + (1.0 - baseline_momentum) * reward
                if reward > 0:
                    key = (problem['line_no'], response)
                    if key not in seen_positive:
                        ultra_write_sft_line(positive_out, prompt, response)
                        seen_positive.add(key)
                if not completion_ids or abs(advantage) < 1e-8:
                    log_and_print(f"[RL E{epoch+1} {index}/{total_problems}] reward={reward:.1f} baseline={baseline:.3f} 跳过更新")
                    continue
                model.train()
                optimizer.zero_grad()
                log_probs = ultra_completion_log_probs(model, prompt_ids, completion_ids)
                loss = -advantage * log_probs.mean()
                loss.backward()
                clip_grad_(trainable, GRAD_CLIP_NORM)
                optimizer.step()
                total_updates += 1
                log_and_print(f"[RL E{epoch+1} {index}/{total_problems}] reward={reward:.1f} baseline={baseline:.3f} updates={total_updates}")
            ultra_save_model_only(model, f"{checkpoint_prefix}_epoch{epoch+1}.pt", epoch + 1)
    log_and_print(f"基础规则奖励 RL 完成：共更新 {total_updates} 次；正确 rollout 已输出至 {positive_output_path}")


def ultra_rejection_sampling_menu(model: WaveSNN, tokenizer: FastBPE):
    problem_path = ultra_select_problem_file()
    if problem_path is None:
        return
    output_path = input("正确回答 SFT 输出文件 [ultra_rejection_sft.jsonl]: ").strip() or 'ultra_rejection_sft.jsonl'
    samples = ultra_read_int("每题采样回答数", ULTRA_RS_SAMPLES_PER_QUESTION)
    max_new_tokens = ultra_read_int("每个回答最多生成 token 数", ULTRA_RS_MAX_NEW_TOKENS)
    ultra_rejection_sampling(model, tokenizer, problem_path, output_path, samples, max_new_tokens)


def ultra_rule_rl_menu(model: WaveSNN, tokenizer: FastBPE):
    problem_path = ultra_select_problem_file()
    if problem_path is None:
        return
    positive_output_path = input("正确 rollout 的 SFT 输出文件 [ultra_rl_positive.jsonl]: ").strip() or 'ultra_rl_positive.jsonl'
    checkpoint_prefix = input("基础 RL 模型 checkpoint 前缀 [ultra_rl]: ").strip() or 'ultra_rl'
    epochs = ultra_read_int("基础 RL epoch 数", ULTRA_RL_EPOCHS)
    max_new_tokens = ultra_read_int("每个回答最多生成 token 数", ULTRA_RL_MAX_NEW_TOKENS)
    lr = ultra_read_float("基础 RL 学习率", ULTRA_RL_LEARNING_RATE)
    baseline_momentum = ultra_read_float("奖励移动平均基线动量", ULTRA_RL_BASELINE_MOMENTUM)
    if baseline_momentum >= 1.0:
        log_and_print("基线动量必须小于 1.0。")
        return
    ultra_rule_rl_train(model, tokenizer, problem_path, positive_output_path, checkpoint_prefix,
                        epochs, max_new_tokens, lr, baseline_momentum)


def ultra_grpo_menu(model: WaveSNN, tokenizer: FastBPE):
    problem_path = ultra_select_problem_file()
    if problem_path is None:
        return
    positive_output_path = input("正确 rollout 的 SFT 输出文件 [ultra_grpo_positive.jsonl]: ").strip() or 'ultra_grpo_positive.jsonl'
    checkpoint_prefix = input("GRPO 模型 checkpoint 前缀 [ultra_grpo]: ").strip() or 'ultra_grpo'
    epochs = ultra_read_int("GRPO epoch 数", ULTRA_GRPO_EPOCHS)
    group_size = ultra_read_int("每题同组采样回答数", ULTRA_GRPO_GROUP_SIZE, minimum=2)
    max_new_tokens = ultra_read_int("每个回答最多生成 token 数", ULTRA_GRPO_MAX_NEW_TOKENS)
    lr = ultra_read_float("GRPO 学习率", ULTRA_GRPO_LEARNING_RATE)
    inner_steps = ultra_read_int("每组参数更新次数", ULTRA_GRPO_INNER_STEPS)
    clip_epsilon = ultra_read_float("策略变化裁剪范围", ULTRA_GRPO_CLIP_EPSILON)
    old_policy_penalty = ultra_read_float("旧策略偏离惩罚系数", ULTRA_GRPO_OLD_POLICY_PENALTY)
    ultra_grpo_train(model, tokenizer, problem_path, positive_output_path, checkpoint_prefix,
                     epochs, group_size, max_new_tokens, lr, inner_steps,
                     clip_epsilon, old_policy_penalty)

# ------------------------------------------------------------
# 主菜单
# ------------------------------------------------------------
def main():
    global TEMPERATURE, TOP_P
    init_log_file()
    tokenizer = FastBPE(VOCAB_SIZE)
    tokenizer_loaded = False
    try:
        if os.path.exists("tokenizer_incremental.bin") and os.path.exists("tokenizer_incremental_tokenizer.json"):
            tokenizer_loaded = tokenizer.load("tokenizer_incremental.bin")
            if tokenizer_loaded:
                log_and_print("成功加载已有分词器。")
            else:
                log_and_print("分词器文件存在但加载失败，将重新训练。")
        else:
            log_and_print("未找到分词器文件，将重新训练。")
    except Exception as e:
        log_and_print(f"加载分词器异常: {e}")

    if not tokenizer_loaded:
        log_and_print("准备训练新分词器。正在统计数据集字符规模...")
        try:
            with open("dialogues.jsonl", 'r', encoding='utf-8') as f:
                all_text = []
                char_set = set()
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    convs = data.get('conversations', [])
                    for turn in convs:
                        content = turn.get('content', '')
                        normalized = normalize_text(content)
                        all_text.append(normalized)
                        for ch in normalized:
                            char_set.add(ch)
                unique_chars = len(char_set)
                special_count = len(SPECIAL_TOKENS_LIST)
                required_min = unique_chars + special_count
                log_and_print(f"数据集包含 {unique_chars} 个不同字符。若使用字符级词表至少需要词汇量 {required_min}。")
                log_and_print(f"当前配置 VOCAB_SIZE = {VOCAB_SIZE}。")
                if VOCAB_SIZE < required_min:
                    log_and_print("警告：当前 VOCAB_SIZE 小于字符级最小需求，部分字符将被映射为 UNK。建议增大 VOCAB_SIZE 配置后重新运行。")
                tokenizer.train(all_text, verbose=True)
                tokenizer.save("tokenizer_incremental.bin")
                if tokenizer._tokenizer is not None:
                    tokenizer._tokenizer.save("tokenizer_incremental_tokenizer.json")
                    log_and_print("分词器训练完成并已保存。")
                else:
                    log_and_print("分词器训练失败。")
                    return
        except FileNotFoundError:
            log_and_print("未找到 dialogues.jsonl，扫描 .txt 文件用于分词器训练...")
            all_text = []
            char_set = set()
            for f in sorted(os.listdir('.')):
                if f.startswith('.'):
                    continue
                if f.endswith('.txt'):
                    try:
                        with open(f, 'r', encoding='utf-8') as tf:
                            for line in tf:
                                line = line.strip()
                                if line:
                                    normalized = normalize_text(line)
                                    all_text.append(normalized)
                                    for ch in normalized:
                                        char_set.add(ch)
                        log_and_print(f"  已读取 {f}")
                    except Exception as e:
                        log_and_print(f"  跳过 {f}: {e}")
            if all_text:
                unique_chars = len(char_set)
                special_count = len(SPECIAL_TOKENS_LIST)
                required_min = unique_chars + special_count
                log_and_print(f"文本文件包含 {unique_chars} 个不同字符。若使用字符级词表至少需要词汇量 {required_min}。")
                log_and_print(f"当前配置 VOCAB_SIZE = {VOCAB_SIZE}。")
                if VOCAB_SIZE < required_min:
                    log_and_print("警告：当前 VOCAB_SIZE 小于字符级最小需求，部分字符将被映射为 UNK。建议增大 VOCAB_SIZE 配置后重新运行。")
                tokenizer.train(all_text, verbose=True)
            else:
                log_and_print("未找到任何可用的 .txt 文件，将使用空文本训练分词器（仅保留特殊token）。")
                tokenizer.train([""], verbose=False)
            tokenizer.save("tokenizer_incremental.bin")
            if tokenizer._tokenizer is not None:
                tokenizer._tokenizer.save("tokenizer_incremental_tokenizer.json")

    # ================= 冻结层控制 =================
    # ===============================================

    # ================= 对照实验入口 =================
    model = WaveSNN(
        vocab_size=tokenizer.vocab_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        pad_token_id=tokenizer.pad_token_id,
        use_residual=USE_RESIDUAL,
        disable_coupling=DISABLE_COUPLING,
        freeze_seed=SEED_FREEZE,
        freeze_layers=FREEZE_LAYERS,
        freeze_embedding=FREEZE_EMBEDDING,
        seed_embedding=SEED_EMBEDDING
    )
    # ===============================================

    setup_model_token_ids(model, tokenizer)

    if os.path.exists("brain.pt"):
        try:
            checkpoint = torch.load("brain.pt", map_location=DEVICE, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                log_and_print("已加载 brain.pt 中的模型参数。")
            else:
                model.load_state_dict(checkpoint)
                log_and_print("已加载 brain.pt 模型权重（旧格式）。")
        except Exception as e:
            log_and_print(f"Model load failed: {e}. Stopping to avoid accidental random initialization.")
            close_log_file()
            return
    else:
        log_and_print("未找到 brain.pt，使用随机初始化模型。")

    while True:
        print("\n=== Wave-SNN 训练与对话系统菜单 ===")
        print("1. 从头训练模型")
        print("2. 对话模式")
        print("3. 保存模型")
        print("4. 加载模型")
        print("5. 调整生成参数")
        print("6. 查看模型统计信息")
        print("7. 规则验证拒绝采样：生成正确回答 SFT 数据")
        print("8. 基础规则奖励 RL：单回答试错训练")
        print("9. GRPO 规则强化学习：同题分组相对优化")
        print("10. 退出")
        choice = input("请选择: ").strip()

        if choice == '1':
            data_path, file_ext = select_training_file()
            if data_path is None:
                continue
            if input("开始从头训练？(y/n): ").lower() == 'y':
                train_from_scratch(model, tokenizer, data_path,
                                   BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
                                   GRAD_CLIP_NORM, LOG_INTERVAL,
                                   accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                                   file_type=file_ext)
        elif choice == '2':
            chat_mode = input("对话格式 (jsonl) 还是纯文本续写 (txt)？[jsonl]: ").strip().lower()
            if chat_mode not in ("jsonl", "txt"):
                chat_mode = "jsonl"
            chat_loop(model, tokenizer, mode=chat_mode)
        elif choice == '3':
            try:
                save_dict = {
                    'model_state_dict': model.state_dict(),
                    'epoch': 0}
                torch.save(save_dict, "brain.pt")
                log_and_print("Model saved to brain.pt.")
            except Exception as e:
                log_and_print(f"保存失败: {e}")
        elif choice == '4':
            try:
                checkpoint = torch.load("brain.pt", map_location=DEVICE, weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                setup_model_token_ids(model, tokenizer)
                log_and_print("模型加载成功。")
            except Exception as e:
                log_and_print(f"Model load failed: {e}. Stopping to avoid accidental random initialization.")
                close_log_file()
                return
        elif choice == '5':
            try:
                TEMPERATURE = float(input(f"temperature (0.1-1.5) [当前 {TEMPERATURE:.2f}]: ") or TEMPERATURE)
                TOP_P = float(input(f"top_p (0.0-1.0) [当前 {TOP_P:.2f}]: ") or TOP_P)
            except ValueError:
                log_and_print("输入无效，保持原值")
        elif choice == '6':
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            frozen_layers = model.freeze_layers
            log_and_print(f"冻结层数: {frozen_layers}/{model.num_layers}")
            log_and_print(f"嵌入层冻结: {'是' if model.freeze_embedding else '否'}")
            log_and_print(f"嵌入层随机种子: {model.seed_embedding}")
            log_and_print(f"可训练参数: {trainable_params:} / 总参数: {total_params:} (冻结: {frozen_params:})")
            log_and_print(f"冻结区随机种子: {model.freeze_seed}")
            log_and_print(f"词表大小: {model.vocab_size}")
            log_and_print("架构：对角复数波神经元 + 并行前缀扫描 + 层间RMSNorm + 残差")
            log_and_print(f"读取方式：全局摘要器（SUMMARY_TYPE='{SUMMARY_TYPE}'，写入 cat([I_gated_norm, g])，读取 input_mod_linear(I)+global_mod_linear(g)）")
        elif choice == '7':
            ultra_rejection_sampling_menu(model, tokenizer)
        elif choice == '8':
            ultra_rule_rl_menu(model, tokenizer)
        elif choice == '9':
            ultra_grpo_menu(model, tokenizer)
        elif choice == '10':
            break
        else:
            log_and_print("无效选择，请重新输入")
    close_log_file()

if __name__ == "__main__":
    main()


