# TRANSFORMER 对照版：经典 Transformer 架构（MHSA + FFN），与 312003elite.py 做对照实验。
# ============================================================
# 基于 312003elite.py 框架改造：ParallelComplexWaveLayer 替换为经典 TransformerBlock
# 内核：Multi-Head Self-Attention + SwiGLU FFN + RMSNorm + 残差连接
# 支持 KV Cache 用于单步递归解码（init_state/step/prefill）
# 参数配置、训练流程、对话接口与 312003elite.py 完全一致，便于对照实验。
# 保留功能：显式冻结层控制（FREEZE_LAYERS / SEED_FREEZE）
#           嵌入层冻结控制（FREEZE_EMBEDDING / SEED_EMBEDDING）
#           MoE-FFN、AMP、梯度检查点等全部保留
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
VOCAB_SIZE = 10000
HIDDEN_SIZE = 352
NUM_LAYERS = 4
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 5.0
BATCH_SIZE = 50
EPOCHS = 5
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
NUM_WORKERS = 0                    # DataLoader 进程数；0=主进程(Windows 最稳)，>0 启用并行 tokenize（配合 persistent_workers）


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


# Transformer 层超参数
TRANSFORMER_NUM_HEADS = 8         # 注意力头数（HIDDEN_SIZE 必须能被 NUM_HEADS 整除）
TRANSFORMER_FF_EXPAND = 4         # FFN 中间维度倍数（SwiGLU 使用 8/3 倍以保持参数量）
TRANSFORMER_DROPOUT = 0.0         # 注意力和FFN的dropout（0.0表示不使用）
TRANSFORMER_MAX_SEQ_LEN = 4096    # 最大序列长度（用于位置编码）

# 稳定性惩罚超参数（Transformer 中暂不使用，保留占位以兼容框架）
STABILITY_LAMBDA = 0.0            # 惩罚系数（设为0禁用）

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
# Transformer 特有参数
# ============================================================
# USE_ROPE: 是否使用旋转位置编码（RoPE）
USE_ROPE = True

# MAX_POSITIONS: 最大位置数（用于RoPE）
MAX_POSITIONS = TRANSFORMER_MAX_SEQ_LEN

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
MOE_ENABLED = False            # True=每层 SSM 后接 MoE-FFN（新增容量，非对照基线）
MOE_NUM_EXPERTS = 8            # 专家数 E
MOE_TOP_K = 2                  # 每个 token 激活的专家数
MOE_EXPERT_HIDDEN = 1024       # SwiGLU 专家中间维度
MOE_AUX_LOSS_COEF = 0.01       # 路由负载均衡辅助损失系数
MOE_RESIDUAL = True            # MoE 输出加残差（并接后置 RMSNorm）
# 专家卸载：'off'=全驻 GPU / 'on'=全驻 CPU 按需搬移 / 'auto'=按显存估算自动决定
MOE_OFFLOAD = 'auto'
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
USE_EIG_SCAN = False           # Transformer 不需要此参数，保留占位以兼容配置
EIG_MIN_GAP = 2e-2             # Transformer 不需要此参数，保留占位以兼容配置
USE_CUDA_GRAPH_DECODE = False  # CUDA Graph 解码（Transformer 中可选启用）
CUDA_GRAPH_WARMUP = 3          # 抓图前预热步数

# ============================================================
# 独立量化开关（FP8 训练 / NVFP4 量化）—— 供 FP8Linear 与工具函数使用
# ============================================================
FP8_TRAIN = False              # True=FP8Linear 在训练时对权重做 E4M3 STE 量化
NVFP4_QUANTIZE = False         # True=权重打包为 E2M1 4bit（推理/微调，逐块缩放）
NVFP4_BLOCK_SIZE = 32          # NVFP4 量化块大小（scale 粒度）

# 安全有界范围
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
        self._vocab_size_cache = None

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
        self._vocab_size_cache = self._tokenizer.get_vocab_size()
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
        return [tid if tid < self._vocab_size_cache else self.unk_token_id for tid in ids]

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
        if self._vocab_size_cache is None:
            self._vocab_size_cache = self._tokenizer.get_vocab_size()
        return self._vocab_size_cache

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


class RotaryPositionEmbedding(nn.Module):
    """旋转位置编码 (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int = TRANSFORMER_MAX_SEQ_LEN):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :]
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                          cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention with causal mask and RoPE"""
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, use_rope: bool = True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, time_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                position_offset: int = 0
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, D = x.shape
        H, HD = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, HD).transpose(1, 2)  # (B, H, T, HD)
        k_new = self.k_proj(x).view(B, T, H, HD).transpose(1, 2)
        v_new = self.v_proj(x).view(B, T, H, HD).transpose(1, 2)

        # RoPE - 先对新token应用位置编码，再拼接历史
        if self.use_rope:
            # q使用当前位置：position_offset 到 position_offset + T - 1
            cos, sin = self.rope(x, position_offset + T)
            q_positions = torch.arange(position_offset, position_offset + T, device=x.device)
            cos_q = cos[:, :, q_positions, :]  # (1, 1, T, HD)
            sin_q = sin[:, :, q_positions, :]
            q = (q * cos_q) + (rotate_half(q) * sin_q)
            # k_new使用当前位置
            cos_k = cos[:, :, q_positions, :]
            sin_k = sin[:, :, q_positions, :]
            k_new = (k_new * cos_k) + (rotate_half(k_new) * sin_k)

        # KV Cache - 拼接历史（历史已经带RoPE）
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k_new], dim=2)
            v = torch.cat([prev_v, v_new], dim=2)
        else:
            k = k_new
            v = v_new
        new_kv_cache = (k, v)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, KV_len)

        # Causal mask
        kv_len = k.shape[2]
        causal_mask = torch.triu(torch.ones(T, kv_len, device=x.device, dtype=torch.bool), diagonal=kv_len - T + 1)
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Attention mask (padding mask)
        if time_mask is not None:
            # time_mask: (B, kv_len) -> (B, 1, 1, kv_len)
            mask = time_mask.float().unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, T, HD)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        return out, new_kv_cache


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network"""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


class TransformerBlock(nn.Module):
    """经典 Transformer Block: RMSNorm + MHSA + RMSNorm + SwiGLU FFN"""
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = TRANSFORMER_NUM_HEADS,
                 ff_expand: int = TRANSFORMER_FF_EXPAND, dropout: float = TRANSFORMER_DROPOUT,
                 use_rope: bool = USE_ROPE):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_idx = -1

        # Attention
        self.attn_norm = RMSNorm(input_dim)
        self.attn = MultiHeadSelfAttention(input_dim, num_heads, dropout, use_rope)

        # FFN
        self.ffn_norm = RMSNorm(input_dim)
        hidden_dim = int(input_dim * ff_expand * 2 / 3)  # SwiGLU 使用 8/3 倍以保持参数量
        self.ffn = FeedForward(input_dim, hidden_dim, dropout)

        self._init_weights()

    def _init_weights(self):
        # Transformer标准初始化：所有输出投影用较小的缩放
        for name, param in self.attn.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param, gain=0.1)  # 缩小10倍
            elif 'bias' in name:
                nn.init.zeros_(param)
        for name, param in self.ffn.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param, gain=0.1)  # 缩小10倍
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, time_mask: Optional[torch.Tensor] = None,
                return_state: bool = False,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                position_offset: int = 0
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        B, T, D = x.shape
        device = x.device

        # Attention
        normed = self.attn_norm(x)
        attn_out, new_kv_cache = self.attn(normed, time_mask=time_mask, 
                                           kv_cache=kv_cache, position_offset=position_offset)
        x = x + attn_out

        # FFN
        normed = self.ffn_norm(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out

        # 返回格式与 ParallelComplexWaveLayer 兼容
        stab_penalty = torch.zeros(1, device=device)
        if return_state:
            # Transformer 状态就是 KV cache
            state = {
                'kv_cache': new_kv_cache,
                'seq_len': position_offset + T
            }
            return x, stab_penalty, state
        return x, stab_penalty

    def forward_prefill(self, x: torch.Tensor, time_mask: Optional[torch.Tensor] = None
                       ) -> Tuple[torch.Tensor, dict]:
        """并行 prefill：一次前向跑完整个序列，返回 (out, state)"""
        out, _, state = self.forward(x, time_mask=time_mask, return_state=True)
        return out, state

    def init_state(self, device: torch.device) -> dict:
        """初始化递归推理状态 - 空的 KV cache"""
        return {
            'kv_cache': None,
            'seq_len': 0
        }

    def step(self, x_t: torch.Tensor, state: dict) -> torch.Tensor:
        """单 token 递归步：x_t (D,) -> y_t (D,)，使用 KV cache"""
        # x_t: (D,) -> (1, 1, D)
        x_t = x_t.unsqueeze(0).unsqueeze(0)
        
        # 获取之前的 KV cache 和当前位置
        kv_cache = state.get('kv_cache', None)
        position_offset = state.get('seq_len', 0)
        
        # Attention
        normed = self.attn_norm(x_t)
        attn_out, new_kv_cache = self.attn(normed, kv_cache=kv_cache, position_offset=position_offset)
        x_t = x_t + attn_out
        
        # FFN
        normed = self.ffn_norm(x_t)
        ffn_out = self.ffn(normed)
        x_t = x_t + ffn_out
        
        # 更新状态中的 KV cache 和位置
        state['kv_cache'] = new_kv_cache
        state['seq_len'] = position_offset + 1
        
        return x_t.squeeze(0).squeeze(0)  # (D,)

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

class Collator:
    """顶层可 pickle 的批整理器。Windows/macOS 用 spawn 启动 DataLoader worker，
    要求 collate_fn 可 pickle；局部闭包无法 pickle（num_workers>0 时直接 PicklingError），故用类。"""
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        pad_id = self.pad_id
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


def make_collate_fn(pad_id):
    # 返回顶层类实例（可 pickle），保持原调用接口不变
    return Collator(pad_id)


def _seed_worker(worker_id):
    """DataLoader worker 种子初始化（顶层函数以兼容 spawn pickle）。"""
    random.seed(42 + worker_id)
    np.random.seed(42 + worker_id)
    torch.manual_seed(42 + worker_id)

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
        return clip_grad_mixed(parameters, max_norm)
    else:
        return float(torch.nn.utils.clip_grad_norm_(parameters, max_norm))


class MoEFFN(nn.Module):
    """SwiGLU 专家 FFN + Top-K 软路由。forward 返回 (out, aux_loss)；step 供单 token 递归（严格等价）。

    专家权重组织为 [E, ...] 的 3D 张量；offload 时驻留 CPU，forward 中按命中专家搬移到 GPU。
    专家权重量化：'fp8_e4m3'（训练 STE）/ 'nvfp4'（推理 4bit）。"""

    _weight_names = ('w_gate', 'w_up', 'w_down')
    _bias_names = ('b_gate', 'b_up', 'b_down')

    def __init__(self, dim, num_experts, top_k, expert_hidden, offload=False, expert_quant=None):
        super().__init__()
        if expert_quant in ('nvfp4', 'bf16'):
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
        # 独立 H2D 拷贝流（P0-4）：offload 时专家权重在 copy engine 上异步搬运，与默认流的计算重叠
        self._copy_stream = torch.cuda.Stream() if (offload and torch.cuda.is_available()) else None
        self._copy_event = None
        if offload and torch.cuda.is_available():
            # pinned memory：offload 时 H2D 带宽从 ~3GB/s 升到 ~10GB/s（非阻塞拷贝的前置条件）
            for _name in self._weight_names + self._bias_names + ('w_gate_scale', 'w_up_scale', 'w_down_scale'):
                _p = getattr(self, _name, None)
                if _p is not None:
                    _p.data = _p.data.pin_memory()
        if expert_quant == 'nvfp4':
            self.pack_nvfp4()
        elif expert_quant == 'bf16':
            self.pack_bf16()

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

    def pack_bf16(self):
        """推理用 bf16 常驻打包（P2-10）：专家权重整体转 bf16 buffer，释放 fp32 主权重（2× 压缩，零反量化开销）。"""
        if self.expert_quant != 'bf16':
            return
        for name in self._weight_names:
            p = getattr(self, name)
            if p is None:
                continue
            scale_p = getattr(self, name + '_scale') if MOE_EXPERT_WNORM else None
            w = p.detach()
            if scale_p is not None:
                w = w * scale_p.detach().unsqueeze(-1).unsqueeze(-1)   # weight_norm 恢复实际权重
            self.register_buffer(f'{name}_bf16', w.to(torch.bfloat16))
            setattr(self, name, None)
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
        if self.expert_quant == 'bf16':
            wg = self.w_gate_bf16[e].to(device).float()
            wu = self.w_up_bf16[e].to(device).float()
            wd = self.w_down_bf16[e].to(device).float()
            bg = self.b_gate[e].to(device)
            bu = self.b_up[e].to(device)
            bd = self.b_down[e].to(device)
            return wg, wu, wd, bg, bu, bd
        if self.offload:
            # non_blocking：pinned 内存的异步 H2D，多个专家拷贝在 copy engine 上流水线化
            wg = self.w_gate[e].to(device, non_blocking=True)
            wu = self.w_up[e].to(device, non_blocking=True)
            wd = self.w_down[e].to(device, non_blocking=True)
            bg = self.b_gate[e].to(device, non_blocking=True)
            bu = self.b_up[e].to(device, non_blocking=True)
            bd = self.b_down[e].to(device, non_blocking=True)
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

    def _dispatch_experts(self, expert_ids, device):
        """批量、尽早派发命中专家的取参（offload 时即 H2D 拷贝），返回 {e: mats}。
        - 推理（eval/no_grad）+ offload：在独立 copy stream 上异步搬运并以 event 同步，
          copy engine 与默认流的索引/计算重叠，隐藏 PCIe 传输；
        - 训练：留在默认流一次性派发全部 non_blocking 拷贝（copy engine 流水线化），
          保留 .to() 的可微 Copy 节点，梯度仍回传 CPU 专家参数（跨 stream 不用于训练以避免 autograd 竞态）。
        数学结果与逐个 _expert_mats 完全一致，仅改变派发时序。"""
        if not expert_ids:
            return {}
        use_stream = (self.offload and self._copy_stream is not None
                      and device.type == 'cuda' and not self.training
                      and not torch.is_grad_enabled())
        if use_stream:
            cur = torch.cuda.current_stream(device)
            with torch.cuda.stream(self._copy_stream):
                self._copy_stream.wait_stream(cur)
                mats = {e: self._expert_mats(e, device) for e in expert_ids}
            self._copy_event = torch.cuda.Event()
            self._copy_event.record(self._copy_stream)
            cur.wait_event(self._copy_event)   # 默认流使用权重前确保 H2D 完成
            return mats
        return {e: self._expert_mats(e, device) for e in expert_ids}

    def forward(self, x, return_aux=True):
        # x: (B, T, D)
        B, T, D = x.shape
        N = B * T
        xf = x.reshape(N, D)
        device = x.device
        topk_p, topk_i, probs = self._router_topk(xf)
        # 阶段1：遍历路由，收集命中专家与其 token（此刻不搬权重）
        plan = []
        for e in range(self.num_experts):
            hit = (topk_i == e)
            if not bool(hit.any()):
                continue
            tok_idx, slot_idx = hit.nonzero(as_tuple=True)
            gp = topk_p[tok_idx, slot_idx]                 # (n,) 门控权重（可微）
            plan.append((e, tok_idx, gp))
        # 批量异步派发全部命中专家的权重（独立流重叠 / 默认流提前排队）
        mats = self._dispatch_experts([e for e, _, _ in plan], device)
        # 阶段2：逐专家计算（权重已就位），累加顺序与原实现一致 → 结果严格相同
        accum_idx, accum_val = [], []
        for e, tok_idx, gp in plan:
            x_e = xf[tok_idx]                              # (n, D)
            wg, wu, wd, bg, bu, bd = mats[e]
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
                    TransformerBlock(hidden_size, hidden_size, num_heads=TRANSFORMER_NUM_HEADS,
                                     use_rope=USE_ROPE)
                )

            torch.random.set_rng_state(_cpu_state)
            if _cuda_states is not None:
                torch.cuda.set_rng_state_all(_cuda_states)
        else:
            self.layers = nn.ModuleList()

        # ---- 可训练层：用全局种子初始化 ----
        for i in range(frozen_count, num_layers):
            self.layers.append(
                TransformerBlock(hidden_size, hidden_size, num_heads=TRANSFORMER_NUM_HEADS,
                                 use_rope=USE_ROPE)
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

        # P2-9：CUDA Graph 解码缓存（惰性抓图，states 变化即重抓；抓图失败自动回退 eager）
        self._dg = None
        self._dg_states = None
        self._dg_tok = None
        self._dg_logits = None
        self._dg_disabled = False

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
                    layer, layer_input, time_mask, use_reentrant=False
                )
            else:
                output_wave, stab_pen = layer(layer_input, time_mask=time_mask)
            if labels is not None and self.training:
                stab_loss_total = stab_loss_total + stab_pen
            # TransformerBlock 内部已有 attn_norm 和 ffn_norm，不再额外归一化
            if self.moe_enabled:
                moe_out, moe_aux = self.moes[i](output_wave)
                if self.training and MOE_AUX_LOSS_COEF != 0.0:
                    moe_aux_total = moe_aux_total + moe_aux
                if MOE_RESIDUAL:
                    output_wave = output_wave + moe_out
                else:
                    output_wave = moe_out
                output_wave = self.moe_norms[i](output_wave)
            layer_activities.append(output_wave.mean().detach())
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

    @torch.no_grad()
    def prefill(self, input_ids):
        """并行 prefill（P1-8）：一次前向并行跑完整个 prompt，返回 (末位置 logits, 各层状态)。与逐 token step 严格等价。"""
        device = input_ids.device
        token_emb = self.embedding(input_ids) * self.input_scale
        layer_input = token_emb
        states = []
        for i, layer in enumerate(self.layers):
            out, st = layer.forward_prefill(layer_input)
            # TransformerBlock 内部已有 attn_norm 和 ffn_norm，不再额外归一化
            if self.moe_enabled:
                moe_out, _ = self.moes[i](out)
                if MOE_RESIDUAL:
                    out = out + moe_out
                else:
                    out = moe_out
                out = self.moe_norms[i](out)
            layer_input = out
            states.append(st)
        norm_last = self.output_norm(layer_input)
        logits = self.output_fc(norm_last[:, -1:]) * F.softplus(self.logit_scale)
        return logits, states

    # ---------- P2-9：CUDA Graph 解码（状态快照/恢复 + 惰性抓图 + 失败回退）----------
    @staticmethod
    def _state_snapshot(st):
        """深拷贝状态里的张量（不可变缓存如 eig 元组保留引用），用于抓图预热后回滚。"""
        if torch.is_tensor(st):
            return st.clone()
        if isinstance(st, dict):
            return {k: WaveSNN._state_snapshot(v) for k, v in st.items()}
        if isinstance(st, list):
            return [WaveSNN._state_snapshot(v) for v in st]
        return st

    @staticmethod
    def _state_restore(snap, dst):
        """把快照值原地拷回目标状态（目标与快照结构一致，且张量地址不变）。"""
        if torch.is_tensor(snap):
            dst.copy_(snap)
        elif isinstance(snap, dict):
            for k, v in snap.items():
                if torch.is_tensor(v):
                    dst[k].copy_(v)
                elif isinstance(v, (dict, list)):
                    WaveSNN._state_restore(v, dst[k])
        elif isinstance(snap, list):
            for a, b in zip(snap, dst):
                WaveSNN._state_restore(a, b)

    def _capture_decode_graph(self, device, states):
        """抓一个「单 token 解码步」的 CUDA Graph。不可捕获/开关关闭时返回 None（调用方回退 eager）。"""
        if not USE_CUDA_GRAPH_DECODE or device.type != 'cuda' or self.moe_enabled:
            return None
        try:
            self._dg_tok = torch.zeros((1, 1), dtype=torch.long, device=device)
            snap = WaveSNN._state_snapshot(states)
            # 预热：在图外执行若干步（预分配 workspace，同时让 eig 缓存落位），再把状态回滚
            for _ in range(int(CUDA_GRAPH_WARMUP)):
                self.step_tensor(self._dg_tok, states)
            WaveSNN._state_restore(snap, states)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._dg_logits = self.step_tensor(self._dg_tok, states)
            # 抓图只记录算子、不执行，状态未被改动；统一同步一次保证图外/图内 stream 对齐
            torch.cuda.synchronize()
            return g
        except Exception:
            self._dg_logits = None
            return None

    def decode_step_graphed(self, token_id, states):
        """CUDA Graph 解码步：与 step 数学一致，但每步只回放一次图（P2-9）。失败自动回退 step。"""
        # 对于 Transformer，状态是 dict，包含 kv_cache 和 seq_len
        # kv_cache 可能是 None 或 (k, v) 元组
        if 'kv_cache' in states[0] and states[0]['kv_cache'] is not None:
            device = states[0]['kv_cache'][0].device  # 从 KV cache 获取设备
        else:
            device = torch.device('cpu')
        if self._dg is None or self._dg_states is not states:
            self._dg = self._capture_decode_graph(device, states)
            self._dg_states = states
            if self._dg is None:
                self._dg_disabled = True
        if self._dg is None:
            return self.step(token_id, states)
        self._dg_tok[0, 0] = token_id           # 写静态输入 buffer（H2D 拷贝，在图外）
        self._dg.replay()
        return self._dg_logits

    def has_offloaded_params(self) -> bool:
        return any(getattr(m, 'offload', False) for m in self.moes) if self.moe_enabled else False

    def pack_moe_nvfp4(self):
        if not self.moe_enabled:
            return
        for m in self.moes:
            m.pack_nvfp4()

    def pack_moe_bf16(self):
        if not self.moe_enabled:
            return
        for m in self.moes:
            m.pack_bf16()

    def step(self, token_id, states):
        """单 token 生成步：token_id (int) -> logits (1, vocab)。与 forward 逐时间步等价。"""
        # 对于 Transformer，状态是 dict，包含 kv_cache 和 seq_len
        if 'kv_cache' in states[0] and states[0]['kv_cache'] is not None:
            device = states[0]['kv_cache'][0].device  # 从 KV cache 获取设备
        else:
            device = torch.device('cpu')
        emb = self.embedding(torch.tensor([[token_id]], dtype=torch.long, device=device)) * self.input_scale
        return self._step_emb(emb[0, 0], states)

    def step_tensor(self, tok, states):
        """同 step，但 token 来自静态 buffer（1,1) long —— CUDA Graph 回放时复用同一输入地址（P2-9）。"""
        emb = self.embedding(tok) * self.input_scale
        return self._step_emb(emb[0, 0], states)

    def _step_emb(self, x_t, states):
        """从嵌入向量 x_t (D,) 起推一层层前向，返回 logits (1, vocab)。step/step_tensor 共用。"""
        # TransformerBlock 内部已有 attn_norm 和 ffn_norm，不再额外归一化
        for i, layer in enumerate(self.layers):
            y_t = layer.step(x_t, states[i])
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

        # 递归推理：并行 prefill 一次跑完 prompt（O(1) kernel 序列），再逐 token 增量生成（O(1)/步）
        prompt_ids = input_ids[0].tolist()

        amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if USE_AMP and device.type == "cuda" else contextlib.nullcontext()

        logits = None
        with amp_ctx:
            if len(prompt_ids) > 0:
                logits, states = self.prefill(input_ids)
            else:
                states = self.init_states(device)

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
                logits = self.decode_step_graphed(next_token, states)

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
    # Windows/macOS 用 spawn：num_workers>0 要求 dataset/collate_fn/worker_init_fn 全部可 pickle，
    # 且训练入口必须在 if __name__=='__main__' 守卫内（否则 worker 反复 re-import 导致无限派生/卡死）。
    # collate_fn 已改为顶层 Collator、worker 初始化改为顶层 _seed_worker；NUM_WORKERS 默认 0（Windows 最稳）。
    g = torch.Generator()
    g.manual_seed(42)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
        persistent_workers=(num_workers > 0)
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

    num_workers = NUM_WORKERS
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=False,
                                     collate_fn=collate_fn, num_workers=num_workers,
                                     pin_memory=PIN_MEMORY, drop_last=True)

    no_decay = {'bias', 'logit_scale', 'input_scale', 'norm', 'ln', 'rope'}
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
                    # --- layer 状态监控（Transformer 版本）---
                    for li, layer in enumerate(model.layers):
                        # 监控注意力和FFN参数范数
                        attn_q_norm = layer.attn.q_proj.weight.norm().item()
                        attn_k_norm = layer.attn.k_proj.weight.norm().item()
                        attn_v_norm = layer.attn.v_proj.weight.norm().item()
                        ffn_gate_norm = layer.ffn.w_gate.weight.norm().item()
                        ffn_up_norm = layer.ffn.w_up.weight.norm().item()
                        ffn_down_norm = layer.ffn.w_down.weight.norm().item()
                        log_and_print(f"  [Layer L{li}] Q={attn_q_norm:.3f} K={attn_k_norm:.3f} V={attn_v_norm:.3f} "
                                     f"FG={ffn_gate_norm:.3f} FU={ffn_up_norm:.3f} FD={ffn_down_norm:.3f}")

                    step_loss_sum = 0.0
                    step_count = 0
                    total_activity = 0.0

            if gradients_counter % accumulation_steps != 0:
                clip_grad_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                optimizer_steps += 1

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

def load_brain_pt_into(model: WaveSNN) -> bool:
    """将 brain.pt 加载到模型；checkpoint 临时放到内存，写入后立即释放。返回 False 表示 brain.pt 不存在。"""
    if not os.path.exists("brain.pt"):
        return False
    checkpoint = torch.load("brain.pt", map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    del checkpoint
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return True


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
        freeze_seed=SEED_FREEZE,
        freeze_layers=FREEZE_LAYERS,
        freeze_embedding=FREEZE_EMBEDDING,
        seed_embedding=SEED_EMBEDDING
    )
    # ===============================================

    setup_model_token_ids(model, tokenizer)

    model_loaded = False
    if os.path.exists("brain.pt"):
        log_and_print("检测到 brain.pt，训练或对话时按需加载（避免启动即占用冗余显存）。")
    else:
        log_and_print("未找到 brain.pt，使用随机初始化模型。")

    def ensure_model_loaded() -> bool:
        """首次用到模型时，若 brain.pt 存在则加载（checkpoint 临时放内存，写入后立即释放）。"""
        nonlocal model_loaded
        if model_loaded:
            return True
        if not os.path.exists("brain.pt"):
            return False
        try:
            load_brain_pt_into(model)
            setup_model_token_ids(model, tokenizer)
        except Exception as e:
            log_and_print(f"加载 brain.pt 失败: {e}。为避免随机初始化误训练，程序退出。")
            close_log_file()
            raise SystemExit(1)
        model_loaded = True
        log_and_print("已加载 brain.pt 模型权重。")
        return True

    while True:
        print("\n=== Wave-SNN 训练与对话系统菜单 ===")
        print("1. 训练模型（继续训练）")
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
            if not ensure_model_loaded():
                log_and_print("未找到 brain.pt，使用随机初始化模型从头训练。")
            if input("开始训练？(y/n): ").lower() == 'y':
                train_from_scratch(model, tokenizer, data_path,
                                   BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
                                   GRAD_CLIP_NORM, LOG_INTERVAL,
                                   accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                                   file_type=file_ext)
        elif choice == '2':
            ensure_model_loaded()
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
                model_loaded = True
            except Exception as e:
                log_and_print(f"保存失败: {e}")
        elif choice == '4':
            try:
                if load_brain_pt_into(model):
                    setup_model_token_ids(model, tokenizer)
                    log_and_print("模型加载成功。")
                    model_loaded = True
                else:
                    log_and_print("未找到 brain.pt。")
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
            log_and_print("架构：经典 Transformer (MHSA + SwiGLU FFN) + RoPE + 层间RMSNorm + 残差")
            log_and_print(f"注意力头数: {TRANSFORMER_NUM_HEADS}, FFN倍数: {TRANSFORMER_FF_EXPAND}, 最大序列长度: {TRANSFORMER_MAX_SEQ_LEN}")
        elif choice == '7':
            ensure_model_loaded()
            ultra_rejection_sampling_menu(model, tokenizer)
        elif choice == '8':
            ensure_model_loaded()
            ultra_rule_rl_menu(model, tokenizer)
        elif choice == '9':
            ensure_model_loaded()
            ultra_grpo_menu(model, tokenizer)
        elif choice == '10':
            break
        else:
            log_and_print("无效选择，请重新输入")
    close_log_file()

if __name__ == "__main__":
    main()


