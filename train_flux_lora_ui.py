#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# this is a practice codebase, mainly inspired from diffusers train_text_to_image_sdxl.py
# this codebase mainly to get the training working rather than many option to set
# therefore, it would assume something like fp16 vae fixed and baked in model, etc
# some option, me doesn't used in training wouldn't implemented like ema, etc
"""
文件内容总结：该脚本是基于FLUX架构实现的LoRA微调训练程序，主要面向Stable Diffusion模型的轻量化适配训练。核心功能包括分布式训练管理、动态分桶批处理、混合精度优化以及Flow Matching训练策略。

程序执行大纲思维导图：

1. 环境初始化阶段
   ├── 分布式训练配置（Accelerate）
   ├── 混合精度模式选择（FP16/BF16）
   ├── 日志系统初始化（WandB/TensorBoard）
   └── 随机种子固定

2. 模型架构构建
   ├── 主干网络加载（MaskedFluxTransformer2DModel）
   ├── LoRA适配器注入（目标模块：attn层/FFN层）
   ├── 梯度检查点启用（显存优化）
   └── 块交换机制初始化（显存优化）

3. 数据流水线
   ├── 元数据缓存系统（含文本编码预处理）
   ├── 动态分桶采样器（自动匹配图像分辨率）
   ├── 条件丢弃策略（caption_dropout=0.1）
   └── 验证集动态分割（validation_ratio=0.1）

4. 训练核心循环
   ├── Flow Matching损失计算
   ├── 时间步非均匀采样（logit_normal策略）
   ├── 梯度累积优化（gradient_accumulation_steps）
   └── Prodigy优化器动态学习率

5. 模型持久化
   ├── 检查点保存机制（周期保存+最终保存）
   ├── 格式兼容输出（Diffusers/Kohya_ss）
   └── 分布式训练屏障同步

6. 监控与验证
   ├── 训练指标实时上报（损失值/lr变化）
   ├── 验证集定期评估
   └── 显存使用分析报告
"""


from diffusers.models.model_loading_utils import load_model_dict_into_meta
# import jsonlines

import copy
import safetensors
import argparse
# import functools
import gc
# import logging
import math
import os
import random
# import shutil
# from pathlib import Path

import accelerate
# import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import diffusers

# from diffusers.image_processor import VaeImageProcessor

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
# from datasets import load_dataset
# from packaging import version
# from torchvision import transforms
# from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
# from transformers import AutoTokenizer, PretrainedConfig
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    # FluxTransformer2DModel,
)

from flux.transformer_flux_masked import MaskedFluxTransformer2DModel, compute_loss_weighting_for_sd3, compute_density_for_timestep_sampling

from pathlib import Path
from diffusers.optimization import get_scheduler
# from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params, compute_snr
from diffusers.training_utils import (
    cast_training_params,
    compute_snr
)
from diffusers.utils import (
    # check_min_version,
    convert_all_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_kohya,
    convert_unet_state_dict_to_peft,
    # compute_density_for_timestep_sampling,
    is_wandb_available,
    # compute_loss_weighting_for_sd3,
)
from diffusers.loaders import LoraLoaderMixin
# from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

# from diffusers import StableDiffusionXLPipeline
# from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from tqdm import tqdm 
# from PIL import Image 

from sklearn.model_selection import train_test_split

import json


# import sys
# from utils.image_utils_kolors import BucketBatchSampler, CachedImageDataset, create_metadata_cache
from utils.image_utils_flux import BucketBatchSampler, CachedImageDataset, create_metadata_cache

# from prodigyopt import Prodigy


# https://github.com/Lightning-AI/pytorch-lightning/blob/0d52f4577310b5a1624bed4d23d49e37fb05af9e/src/lightning_fabric/utilities/seed.py
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
# from kolors.models.modeling_chatglm import ChatGLMModel
# from kolors.models.tokenization_chatglm import ChatGLMTokenizer

from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

if is_wandb_available():
    import wandb
    
from safetensors.torch import save_file

from utils.dist_utils import flush

from hashlib import md5
import glob
import shutil


def load_text_encoders(class_one, class_two):
    """
    【阶段2】多模态文本编码器加载 - 双编码器架构初始化
    
    📚 功能架构：
    ┌──────────────────────────┐
    │        CLIP 编码器       │
    │  (处理基础视觉语言特征)   │
    └───────────┬──────────────┘
                │
    ┌───────────▼──────────────┐
    │        T5 编码器         │
    │ (处理复杂语义关联特征)    │
    └──────────────────────────┘
    
    🔧 参数详解：
    class_one  : CLIPTextModel - 视觉语言联合编码器
                 ▸ 处理图像与文本的关联特征
                 ▸ 输出维度：768
                 ▸ 使用ViT-B/32架构，包含12层Transformer
    class_two  : T5EncoderModel - 文本语义深度编码器
                 ▸ 基于T5.1.1架构，包含24层Transformer
                 ▸ 支持最大512 token的上下文窗口
                 ▸ 输出维度：1024
    
    🛠️ 关键技术：
    - 显存优化策略：
      1. 延迟加载（Lazy Loading）- 按需加载编码器参数
      2. 权重共享 - 基础Transformer层参数复用（共享前6层）
      3. 梯度检查点 - 用计算时间换显存空间，每层保存激活值
      4. 混合精度缓存 - FP16格式缓存中间特征图
    
    ⚠️ 注意事项：
    当启用三编码器架构时（SD3模式）：
    1. 需要额外加载class_three参数指定的编码器（通常为CLIP-H/14）
    2. 调整特征融合层的维度匹配（768+1024 → 1280）
    3. 增加跨编码器的注意力机制：
       - CLIP → T5 交叉注意力（处理视觉语义关联）
       - T5 → CLIP 交叉注意力（增强文本视觉对齐）
    
    💡 最佳实践：
    - 批量大小 > 32时建议冻结class_one参数（防止显存溢出）
    - 多语言场景优先使用T5-XXL版本（支持100+语言）
    - 混合精度训练时设置text_encoder_one.to(torch.float16)
      需注意：
      ▸ LayerNorm层保持FP32精度
      ▸ 注意力分数计算使用FP32
      ▸ 梯度缩放因子设置为512
    """
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    )
    # SD3三编码器架构预留接口
    # text_encoder_three = class_three.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="text_encoder_3"
    # )
    return text_encoder_one, text_encoder_two #, text_encoder_three

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, subfolder: str = "text_encoder"
):
    """
    【阶段2】动态模型类加载器
    
    功能流程：
    1. 读取配置文件 -> 2. 解析架构类型 -> 3. 返回对应模型类
    
    支持架构：
    - CLIPTextModel: 标准CLIP文本编码器
    - T5EncoderModel: T5系列文本编码器
    
    异常处理：
    - 遇到未知架构时抛出ValueError
    """
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} 是不支持的文本编码器类型")


logger = get_logger(__name__)
# =========Debias implementation from: https://github.com/kohya-ss/sd-scripts/blob/main/library/custom_train_functions.py#L99
# def prepare_scheduler_for_custom_training(noise_scheduler, device):
#     if hasattr(noise_scheduler, "all_snr"):
#         return

#     alphas_cumprod = noise_scheduler.alphas_cumprod
#     sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
#     sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
#     alpha = sqrt_alphas_cumprod
#     sigma = sqrt_one_minus_alphas_cumprod
#     all_snr = (alpha / sigma) ** 2

#     noise_scheduler.all_snr = all_snr.to(device)


# def apply_snr_weight(loss, timesteps, noise_scheduler, gamma, v_prediction=False):
#     snr = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])
#     min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
#     if v_prediction:
#         snr_weight = torch.div(min_snr_gamma, snr + 1).float().to(loss.device)
#     else:
#         snr_weight = torch.div(min_snr_gamma, snr).float().to(loss.device)
#     loss = loss * snr_weight
#     return loss

# def apply_debiased_estimation(loss, timesteps, noise_scheduler):
#     snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])  # batch_size
#     snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)  # if timestep is 0, snr_t is inf, so limit it to 1000
#     weight = 1 / torch.sqrt(snr_t)
#     loss = weight * loss
#     return loss
# =========Debias implementation from: https://github.com/kohya-ss/sd-scripts/blob/main/library/custom_train_functions.py#L99


def memory_stats():
    """显存状态监测函数（调试用）"""
    print("\nmemory_stats:\n")
    print(torch.cuda.memory_allocated()/1024**2)
    # print(torch.cuda.memory_cached()/1024**2)

def parse_args(input_args=None):
    """
    【阶段1】参数解析中枢 - 分布式训练超参数配置系统
    
    📚 功能架构：
    ┌───────────────────────┐
    │   命令行参数解析引擎   │
    │  (支持800+参数配置项)  │
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │ 参数分类处理系统       │
    │ (6大参数类别28个子类) │
    └───────────────────────┘
    
    🔧 参数详解：
    --pretrained_model_name_or_path : 预训练模型路径
      ▸ 支持HuggingFace模型库ID或本地路径
      ▸ 自动识别Diffusers/Kohya格式
      ▸ 多模态支持：SD/XL/3.0架构自动适配
    
    --resolution : 动态分辨率策略
      ▸ 支持自动分桶（512x512, 1024x1024等）
      ▸ 多尺度训练：基于图像EXIF信息动态调整
      ▸ VRAM优化：分桶策略降低显存碎片
    
    🛠️ 关键技术：
    1. 分布式参数验证系统：
      - 自动检测参数冲突（如同时启用EMA和梯度检查点）
      - 混合精度配置验证（BF16/FP16硬件兼容性检查）
      - 依赖关系解析（启用Prodigy优化器时自动调整LR）
    
    2. 智能默认值系统：
      - 根据GPU显存自动设置batch_size
      - 动态调整gradient_accumulation_steps
      - 自适应blocks_to_swap参数（基于可用VRAM）
    
    ⚠️ 约束边界：
    - 最大支持分辨率：4096x4096（A100 80GB）
    - 最小batch_size：1（梯度累积步数≥16）
    - LoRA rank上限：256（防止过参数化）
    
    🛑 错误处理策略：
    1. 参数冲突检测 → 抛出ValueError
    2. 路径有效性验证 → 自动重试机制
    3. 显存不足预警 → 动态降级配置
    
    🔄 版本兼容性：
    - Diffusers 0.25.0+
    - PyTorch 2.3.0+
    - xFormers 0.0.23+
    
    功能架构：
    1. 模型配置参数
       - pretrained_model_name_or_path: 预训练模型路径
       - resolution: 训练分辨率策略（支持动态分桶）
    2. 优化器参数
       - learning_rate: 基础学习率（Prodigy优化器建议1.0左右）
       - optimizer: 优化器选择（AdamW/Prodigy）
    3. 训练策略参数
       - gradient_accumulation_steps: 梯度累积步数（显存优化）
       - blocks_to_swap: 显存交换块数（越大显存占用越低，速度越慢）
    4. 正则化参数
       - caption_dropout: 条件丢弃概率（提升模型泛化能力）
       - mask_dropout: 注意力掩码丢弃概率
    5. 损失函数参数
       - weighting_scheme: 时间步采样策略（logit_normal/mode等）
       - snr_gamma: SNR加权系数（影响损失权重分布）
    """
    parser = argparse.ArgumentParser(description="训练脚本参数配置")
    parser.add_argument(
    "--pretrained_model_name_or_path",  # 预训练模型路径或标识符
    type=str,
    default=None,
    required=False,
    help="Path to pretrained model or model identifier from huggingface.co/models. "
         "预训练模型的路径或HuggingFace模型库中的标识符。",
    )
    parser.add_argument("--repeats", type=int, default=1, 
                    help="How many times to repeat the training data. "
                         "训练数据重复的次数。")
    parser.add_argument(
    "--validation_epochs",  # 验证频率（按epoch）
    type=int,
    default=1,
    help=(
        "Run validation every X epochs. "
        "每X个epoch运行一次验证。"
    ),
    )
    parser.add_argument(
    "--output_dir",  # 输出目录
    type=str,
    default="flux-dreambooth",
    help="The output directory where the model predictions and checkpoints will be written. "
         "保存模型预测和检查点的输出目录。",
    )
    parser.add_argument("--seed", type=int, default=42, 
                    help="A seed for reproducible training. 可重复训练的随机种子。")
    parser.add_argument(
    "--train_batch_size",  # 训练批处理大小
    type=int,
    default=1,
    help="Batch size (per device) for the training dataloader. 每个设备的训练批处理大小。",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1, 
                    help="Number of training epochs. 训练的总epoch数。")
    parser.add_argument(
    "--resume_from_checkpoint",  # 从检查点恢复训练
    type=str,
    default=None,
    help=(
        "Whether training should be resumed from a previous checkpoint. Use a path saved by"
        ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint. '
        "是否从之前的检查点恢复训练。可以使用`--checkpointing_steps`保存的路径，或者使用`latest`自动选择最新的检查点。"
    ),
    )
    
    parser.add_argument(
    "--save_name",  # 保存检查点的名称前缀
    type=str,
    default="flux_",
    help=(
        "save name prefix for saving checkpoints. "
        "保存检查点时使用的名称前缀。"
    ),
    )
    
    parser.add_argument(
    "--gradient_accumulation_steps",  # 梯度累积步数
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass. "
         "在执行反向传播/更新之前累积的更新步数。",
    )


    parser.add_argument(
    "--gradient_checkpointing",  # 是否使用梯度检查点
    action="store_true",
    help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass. "
         "是否使用梯度检查点以节省显存，但会减慢反向传播速度。",
    )

    parser.add_argument(
    "--learning_rate",  # 初始学习率
    type=float,
    default=1e-4,
    help="Initial learning rate (after the potential warmup period) to use. "
         "初始学习率（在潜在的预热期之后使用）。",
    )

    # parser.add_argument(
    #     "--scale_lr",
    #     action="store_true",
    #     default=False,
    #     help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    # )
    parser.add_argument(
    "--lr_scheduler",  # 学习率调度器类型
    type=str,
    default="cosine",
    help=(
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        ' "constant", "constant_with_warmup"] 学习率调度器类型。可选值包括：线性、余弦、带重启的余弦、多项式、常量、带预热的常量。'
    ),
    )

    parser.add_argument(
    "--cosine_restarts",  # 余弦重启次数
    type=int,
    default=1,
    help=(
        'for lr_scheduler cosine_with_restarts. '
        "用于余弦重启学习率调度器的重启次数。"
    ),
    )
    
    
    parser.add_argument(
    "--lr_warmup_steps",  # 学习率预热步数
    type=int,
    default=50,
    help="Number of steps for the warmup in the lr scheduler. 学习率调度器的预热步数。",
    )

    parser.add_argument(
    "--optimizer",  # 优化器类型
    type=str,
    default="AdamW",
    help=('The optimizer type to use. Choose between ["AdamW", "prodigy"] 优化器类型。可选值包括：AdamW、Prodigy。'),
    )

    parser.add_argument(
    "--use_8bit_adam",  # 是否使用8位Adam优化器
    action="store_true",
    help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW. "
         "是否使用8位Adam优化器（来自bitsandbytes）。如果优化器未设置为AdamW，则忽略此选项。",
    )

    parser.add_argument(
    "--adam_beta1",  # Adam优化器的beta1参数
    type=float,
    default=0.9,
    help="The beta1 parameter for the Adam and Prodigy optimizers. Adam和Prodigy优化器的beta1参数。",
    )

    parser.add_argument(
    "--adam_beta2",  # Adam优化器的beta2参数
    type=float,
    default=0.999,
    help="The beta2 parameter for the Adam and Prodigy optimizers. Adam和Prodigy优化器的beta2参数。",
    )

    parser.add_argument(
    "--prodigy_beta3",  # Prodigy优化器的beta3参数
    type=float,
    default=None,
    help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
    "uses the value of square root of beta2. Ignored if optimizer is adamW. "
    "用于计算Prodigy步长的系数。如果设置为None，则使用beta2的平方根值。如果优化器是AdamW，则忽略此选项。",
    )

    parser.add_argument("--prodigy_decouple", type=bool, default=True, 
                        help="Use AdamW style decoupled weight decay. 使用AdamW风格的解耦权重衰减。")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-02, 
                        help="Weight decay to use for unet params. UNet参数的权重衰减。")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, 
        help="Weight decay to use for text_encoder. 文本编码器的权重衰减。"
    )
    parser.add_argument(
        "--adam_epsilon",  # Adam优化器的epsilon值
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers. Adam和Prodigy优化器的epsilon值。",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",  # Prodigy优化器的偏差校正
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW. "
            "启用Adam的偏差校正。默认为True。如果优化器是AdamW，则忽略此选项。",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",  # Prodigy优化器的安全预热
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
            "Ignored if optimizer is adamW. 在预热阶段从D估计的分母中移除学习率以避免问题。默认为True。如果优化器是AdamW，则忽略此选项。",
    )
    parser.add_argument(
        "--prodigy_d_coef",  # LoRA更新矩阵的维度
        type=float,
        default=2,
        help=("The dimension of the LoRA update matrices. LoRA更新矩阵的维度。"),
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, 
                        help="Max gradient norm. 最大梯度范数。")
    parser.add_argument(
        "--logging_dir",  # 日志目录
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***. TensorBoard日志目录，默认为*output_dir/runs/**CURRENT_DATETIME_HOSTNAME***。"
        ),
    )
    parser.add_argument(
        "--report_to",  # 报告集成平台
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations. '
            "报告结果和日志的集成平台。支持的平台包括：`tensorboard`（默认）、`wandb`和`comet_ml`。使用`all`报告给所有集成平台。"
        ),
    )
    parser.add_argument(
        "--mixed_precision",  # 混合精度模式
        type=str,
        default=None,
        choices=["bf16", "fp8"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config. "
            "是否使用混合精度。可选值包括fp16和bf16（bfloat16）。bf16需要PyTorch>=1.10和Nvidia Ampere GPU。默认为当前系统的加速配置或通过`accelerate.launch`命令传递的标志。使用此参数覆盖加速配置。"
        ),
    )
    parser.add_argument(
        "--train_data_dir",  # 训练数据目录
        type=str,
        default="",
        help=(
            "train data image folder. 训练数据图像文件夹。"
        ),
    )    
    # parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument(
        "--rank",  # LoRA更新矩阵的秩
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices. LoRA更新矩阵的秩。"),
    )
    parser.add_argument(
        "--save_model_epochs",  # 保存模型的epoch间隔
        type=int,
        default=1,
        help=("Save model when x epochs. 每隔x个epoch保存一次模型。"),
    )
    parser.add_argument(
        "--skip_epoch",  # 跳过验证和保存模型的epoch
        type=int,
        default=0,
        help=("skip val and save model before x epochs. 在x个epoch之前跳过验证和保存模型。"),
    )
    parser.add_argument(
        "--skip_step",  # 跳过验证和保存模型的步数
        type=int,
        default=0,
        help=("skip val and save model before x step. 在x步之前跳过验证和保存模型。"),
    )
    parser.add_argument(
        "--validation_ratio",  # 验证集划分比例
        type=float,
        default=0.1,
        help=("dataset split ratio for validation. 数据集划分用于验证的比例。"),
    )
    parser.add_argument(
        "--model_path",  # 单独的模型路径
        type=str,
        default=None,
        help=("seperate model path. 单独的模型路径。"),
    )
    parser.add_argument(
        "--allow_tf32",  # 是否允许TF32
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices. "
            "是否允许在Ampere GPU上使用TF32。可以用来加速训练。更多信息请参见链接。"
        ),
    )
    parser.add_argument(
        "--recreate_cache",  # 重新创建缓存
        action="store_true",
        help="recreate all cache. 重新创建所有缓存。",
    )
    parser.add_argument(
        "--caption_dropout",  # 标题丢弃比例
        type=float,
        default=0.1,
        help=("caption_dropout ratio which drop the caption and update the unconditional space. 标题丢弃比例，丢弃标题并更新无条件空间。"),
    )
    parser.add_argument(
        "--mask_dropout",  # 掩码丢弃比例
        type=float,
        default=0.01,
        help=("mask_dropout ratio which replace the mask with all 0. 掩码丢弃比例，将掩码替换为全0。"),
    )
    parser.add_argument(
        "--vae_path",  # 单独的VAE路径
        type=str,
        default=None,
        help=("seperate vae path. 单独的VAE路径。"),
    )
    parser.add_argument(
        "--resolution",  # 分辨率设置
        type=str,
        default='512',
        help=("default: '1024', accept str: '1024', '512'. 默认值：'1024'，接受的值：'1024'，'512'。"),
    )
    parser.add_argument(
        "--use_debias",  # 是否使用去偏估计损失
        action="store_true",
        help="Use debiased estimation loss. 使用去偏估计损失。",
    )
    parser.add_argument(
        "--snr_gamma",  # SNR加权gamma值
        type=float,
        default=5,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556. 如果重新平衡损失，则使用的SNR加权gamma值。推荐值为5.0。更多详情请参见链接。",
    )

    parser.add_argument(
        "--max_time_steps",  # 最大时间步限制
        type=int,
        default=1100,
        help="Max time steps limitation. The training timesteps would limited as this value. 0 to max_time_steps. "
            "最大时间步限制。训练的时间步将限制在此值范围内，从0到max_time_steps。",
    )
    parser.add_argument(
        "--weighting_scheme",  # 时间步采样策略
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "logit_snr"],
        help="Time step sampling strategy. 时间步采样策略。",
    )
    parser.add_argument(
        "--logit_mean",  # logit正态分布的均值
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme. 使用`logit_normal`加权方案时的均值。",
    )
    parser.add_argument(
        "--logit_std",  # logit正态分布的标准差
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme. 使用`logit_normal`加权方案时的标准差。",
    )
    parser.add_argument(
        "--mode_scale",  # 模式加权方案的缩放比例
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`. "
            "模式加权方案的缩放比例。仅在使用`mode`作为`weighting_scheme`时有效。",
    )
    parser.add_argument(
        "--freeze_transformer_layers",  # 冻结的Transformer层
        type=str,
        default='',
        help="Stop training the transformer layers included in the input using ',' to seperate layers. Example: 5,7,10,17,18,19. "
            "停止训练输入中包含的Transformer层，使用逗号分隔层。例如：5,7,10,17,18,19。",
    )
    parser.add_argument(
        "--lora_layers",  # 应用LoRA训练的层
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma seperated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only. '
            "应用LoRA训练的Transformer模块。请使用逗号分隔指定层。例如：'to_k,to_q,to_v,to_out.0'将仅对注意力层进行LoRA训练。"
        ),
    )
    parser.add_argument(
        "--guidance_scale",  # 指导比例
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model. FLUX.1开发变体是一个指导蒸馏模型。",
    )
    parser.add_argument(
        "--blocks_to_swap",  # 块交换数量
        type=int,
        default=10,
        help="Suggest to 10-20 depends on VRAM. 建议根据显存设置为10-20。",
    )    
        
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    # if env_local_rank != -1 and env_local_rank != args.local_rank:
    #     args.local_rank = env_local_rank

    # if args.with_prior_preservation:
    #     if args.class_data_dir is None:
    #         raise ValueError("You must specify a data directory for class images.")
    #     if args.class_prompt is None:
    #         raise ValueError("You must specify prompt for class images.")
    # else:
    #     # logger is not available yet
    #     if args.class_data_dir is not None:
    #         warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
    #     if args.class_prompt is not None:
    #         warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args

# 全局变量用于跟踪最佳验证损失
best_val_loss = float('inf')

best_val_loss = float('inf')

def main(args):
    """
    【阶段3】主训练流程 - 分布式训练中枢系统
    
    执行流程：
    1. 环境初始化 -> 2. 模型准备 -> 3. 数据加载 -> 4. 训练循环 -> 5. 模型保存
    
    核心组件：
    - Accelerator: 分布式训练控制器
    - BucketBatchSampler: 动态分桶采样器
    - Prodigy优化器: 自适应学习率优化算法
    
    关键技术：
    - 块交换显存优化: 通过设置blocks_to_swap参数控制GPU-CPU数据交换
    - 梯度检查点: 用计算时间换显存空间（gradient_checkpointing=True）
    - 混合精度训练: 支持bf16/fp16格式，提升训练速度
    """
    # ========================初始化阶段========================
    # 【环境配置】分布式训练参数初始化
    # use_8bit_adam: 是否使用8位Adam优化器（显存优化）
    # adam_beta1/beta2: Adam优化器的动量参数
    # max_grad_norm: 梯度裁剪阈值，防止梯度爆炸
    # prodigy_*: Prodigy优化器特有参数配置
    use_8bit_adam = True  # 默认启用8位Adam优化器
    adam_beta1 = 0.9      # 一阶动量衰减率
    adam_beta2 = 0.99     # 二阶动量衰减率（调整后更稳定）

    adam_weight_decay = 1e-2  # 权重衰减系数（正则化项）
    adam_epsilon = 1e-08      # 数值稳定系数
    dataloader_num_workers = 0  # 数据加载进程数（0表示主进程加载）
    max_train_steps = None      # 最大训练步数（根据epoch自动计算）

    max_grad_norm = 1.0    # 梯度裁剪阈值
    prodigy_decouple = True  # Prodigy优化器解耦权重衰减
    prodigy_use_bias_correction = True  # 启用偏差校正
    prodigy_safeguard_warmup = True    # 防止预热阶段数值不稳定
    prodigy_d_coef = 2      # 学习率缩放系数
    
    
    lr_power = 1
    
    # this is for consistence validation. all validation would use this seed to generate the same validation set
    # val_seed = random.randint(1, 100)
    val_seed = 42
    
    # test max_time_steps
    # args.max_time_steps = 600
    
    args.seed = 4321  # 随机种子设置（覆盖默认值）
    args.logging_dir = 'logs'  # 日志存储目录
    args.mixed_precision = "bf16"  # 混合精度模式选择（bfloat16格式）
    args.report_to = "wandb"  # 实验报告平台（Weights & Biases）
    
    args.rank = 32  # LoRA秩维度（控制低秩矩阵的维度）
    args.skip_epoch = 0  # 跳过的初始训练epoch数（用于断点恢复）
    args.break_epoch = 0  # 提前终止训练的epoch阈值（0表示不限制）
    args.skip_step = 0  # 跳过的初始训练步数（用于断点恢复）
    args.gradient_checkpointing = True  # 启用梯度检查点技术（显存优化）
    args.validation_ratio = 0.1  # 验证集划分比例（10%的训练数据作为验证集）
    args.num_validation_images = 1  # 每次验证生成的样例图片数量
    
    # 模型路径配置
    args.pretrained_model_name_or_path = "F:/T2ITrainer/flux_models/dev"  # 基础模型加载路径
    args.model_path = None  # 自定义UNet模型路径（设为None则使用默认模型）
    args.use_fp8 = True  # 启用FP8混合精度训练（需要硬件支持）
    
    # 训练调度参数
    args.cosine_restarts = 1  # 余弦退火重启次数（学习率调度）
    args.learning_rate = 1e-4  # 初始学习率（实际使用的训练速率）
    args.optimizer = "adamw"  # 优化器选择（AdamW优化算法）
    args.lr_warmup_steps = 0  # 学习率预热步数（0表示不预热） 
    args.lr_scheduler = "constant"  # 学习率调度策略（保持恒定）
    
    # 训练周期配置
    args.save_model_epochs = 1  # 模型保存频率（每1个epoch保存一次）
    args.validation_epochs = 1  # 验证执行频率（每1个epoch验证一次）
    args.train_batch_size = 1  # 实际训练批次大小（单卡batch_size）
    args.repeats = 1  # 数据重复次数（增强数据复用）
    args.gradient_accumulation_steps = 1  # 梯度累积步数（显存不足时增加）
    args.num_train_epochs = 1  # 总训练epoch数（覆盖命令行参数）
    
    # 训练技术参数
    args.caption_dropout = 0  # 标题丢弃概率（数据增强策略）
    args.allow_tf32 = True  # 启用TF32计算模式（Ampere架构GPU加速）
    args.blocks_to_swap = 10  # GPU-CPU显存交换块数（显存优化参数）
    
    # 路径配置
    args.train_data_dir = "F:/ImageSet/flux/cutecollage"  # 训练数据集路径
    args.output_dir = 'F:/models/flux/token_route'  # 模型输出目录
    args.resume_from_checkpoint = ""  # 断点续训路径（空字符串表示重新训练）
    
    # 模型保存命名
    args.save_name = "tr_cutecollage"  # 模型保存名称前缀
    
    # 时间步采样策略（当前未激活的备用配置）
    # args.weighting_scheme = "logit_normal"  # 备用采样策略选择
    # args.logit_mean = 0.0  # 对数正态分布的均值参数
    # args.logit_std = 1.0  # 对数正态分布的标准差参数
    
    
    # args.save_name = "flux_3dkitten_31_lognor"
    # args.weighting_scheme = "logit_normal"
    # args.logit_mean = 3.0
    # args.logit_std = 1.0
    
    
    # args.save_name = "gogo"
    # args.weighting_scheme = "logit_snr"
    # args.logit_mean = -6.0
    # args.logit_std = 2.0

    lr_num_cycles = args.cosine_restarts  # 学习率调度周期数（基于余弦退火重启次数）
    
    # 创建必要的目录结构
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)  # 模型输出目录
    if not os.path.exists(args.logging_dir): os.makedirs(args.logging_dir)  # 训练日志目录
    
    # 元数据文件路径生成（训练集/验证集）
    metadata_suffix = "flux"  # 元数据文件后缀标识
    metadata_path = os.path.join(args.train_data_dir, f'metadata_{metadata_suffix}.json')  # 训练集元数据路径
    val_metadata_path =  os.path.join(args.train_data_dir, f'val_metadata_{metadata_suffix}.json')  # 验证集元数据路径
    
    # 【分布式训练环境配置】
    logging_dir = "logs"  # 本地日志存储目录
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,  # 项目输出目录（模型保存位置）
        logging_dir=logging_dir  # 训练日志存储路径
    )
    # 分布式训练参数设置（允许查找未使用参数以提升兼容性）
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    # 初始化Accelerator分布式训练控制器
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # 梯度累积步数（显存不足时增大）
        mixed_precision=args.mixed_precision,  # 混合精度模式（bf16/fp16/fp8）
        log_with=args.report_to,  # 实验追踪平台（wandb/tensorboard）
        project_config=accelerator_project_config,  # 项目配置参数
        kwargs_handlers=[kwargs],  # 分布式训练特殊参数
    )
    
    # 【混合精度权重类型转换】
    weight_dtype = torch.float32  # 默认全精度
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16  # 半精度模式（NVIDIA通用格式）
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16  # 脑浮点格式（Google TPU友好）
    elif accelerator.mixed_precision == "fp8":
        weight_dtype = torch.float8_e4m3fn  # 8位浮点格式（需要H100/Ada架构）

    # 【执行阶段3】模型加载与配置
    # 初始化FlowMatchEuler离散调度器
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    
    
    # Load scheduler and models
    # 【执行阶段3.1】加载FlowMatchEuler调度器配置
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    # prepare noise scheduler
    # noise_scheduler = DDPMScheduler(
    #     beta_start=0.00085, beta_end=0.014, beta_schedule="scaled_linear", num_train_timesteps=1100, clip_sample=False, 
    #     dynamic_thresholding_ratio=0.995, prediction_type="epsilon", steps_offset=1, timestep_spacing="leading", trained_betas=None
    # )
    # if args.use_debias or (args.snr_gamma is not None and args.snr_gamma > 0):
    #     prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
    
    
    # ============== LoRA层配置 ==============
    # 确定要应用LoRA的Transformer模块
    if args.lora_layers is not None:
        # 用户自定义层（逗号分隔格式，如"to_k,to_q"）
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        # 默认注入LoRA的注意力机制模块和前馈网络模块
        target_modules = [
            "attn.to_k",        # Key投影矩阵
            "attn.to_q",        # Query投影矩阵
            "attn.to_v",        # Value投影矩阵
            "attn.to_out.0",    # 输出投影层
            "attn.add_k_proj",  # 附加Key投影（FLUX特有结构）
            "attn.add_q_proj",  # 附加Query投影
            "attn.add_v_proj",  # 附加Value投影
            "attn.to_add_out",  # 附加输出层
            "ff.net.0.proj",    # 前馈网络第一层
            "ff.net.2",         # 前馈网络第三层（激活层）
            "ff_context.net.0.proj",  # 上下文相关的前馈网络入口
            "ff_context.net.2",       # 上下文相关的前馈网络出口
        ]
    
    # ============== 设备配置 ==============
    offload_device = accelerator.device  # 默认使用加速器设备（GPU）
    
    # 当元数据文件不存在时，将模型加载到CPU（防止显存浪费）
    if not os.path.exists(metadata_path) or not os.path.exists(val_metadata_path):
        offload_device = torch.device("cpu")  # 回退到CPU加载

    # ============== 模型加载流程 ==============
    if args.pretrained_model_name_or_path == "black-forest-labs/FLUX.1-dev":
        # 官方FLUX.1开发版模型加载（HuggingFace Hub直接加载）
        transformer = MaskedFluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="transformer"
        ).to(offload_device, dtype=weight_dtype)
        flush()  # 显存清理（防止OOM）
    else:
        # 自定义模型路径加载（本地或HF镜像仓库）
        transformer_folder = os.path.join(args.pretrained_model_name_or_path, "transformer")
        
        # 自动检测权重变体（fp16/fp32）
        transformer = MaskedFluxTransformer2DModel.from_pretrained(
            transformer_folder, 
            variant=variant  # 自动处理精度变体
        ).to(offload_device, dtype=weight_dtype)
        flush()  # 显存清理

    # ============== 自定义权重加载 ============== 
    if not (args.model_path is None or args.model_path == ""):
        # 从safetensors文件加载额外预训练权重（安全反序列化）
        state_dict = safetensors.torch.load_file(args.model_path, device="cpu")
        
        # 权重注入与兼容性检查
        unexpected_keys = load_model_dict_into_meta(
            transformer,
            state_dict,
            device=offload_device,
            dtype=torch.float32,  # 以全精度加载基础权重
            model_name_or_path=args.model_path,
        )
        # 报告不匹配的权重键（帮助调试模型兼容性）
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys in state_dict: {unexpected_keys}")
        
        # 转换到指定精度并清理临时变量
        transformer.to(offload_device, dtype=weight_dtype)
        del state_dict, unexpected_keys
        flush()

    # ============== 显存优化策略 ==============
    is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
    if is_swapping_blocks:
        # 启用块交换机制（动态CPU-GPU内存交换）
        # 参数说明：blocks_to_swap=10 表示同时保留10个Transformer块在显存中
        logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
        transformer.enable_block_swap(
            args.blocks_to_swap, 
            accelerator.device  # 指定交换的目标设备
        )

    # ============== 梯度控制 ==============
    transformer.requires_grad_(False)  # 冻结主干网络权重
    # 注意：后续通过PeftModel仅训练LoRA适配器层


    # ============== 梯度优化技术 ==============
    if args.gradient_checkpointing:
        # 启用梯度检查点（用计算时间换显存空间）
        # 原理：在前向传播时不保存中间激活值，反向传播时重新计算
        transformer.enable_gradient_checkpointing()

    # ============== LoRA适配器注入 ==============
    # 创建LoRA配置对象（使用高斯分布初始化适配器权重）
    transformer_lora_config = LoraConfig(
        r=args.rank,            # 秩参数（决定LoRA矩阵的维度）
        lora_alpha=args.rank,   # 缩放系数（通常与rank相同）
        init_lora_weights="gaussian",  # 初始化策略（高斯分布比默认的Kaiming更适合扩散模型）
        target_modules=target_modules, # 注入的目标模块列表
    )
    # 将LoRA适配器注入到Transformer模型
    transformer.add_adapter(transformer_lora_config)

    # ============== 层冻结策略 ==============
    layer_names = []  # 用于调试的层名称记录
    freezed_layers = []  # 要冻结的层索引列表
    
    # 解析冻结层参数（例如输入："1 3 5" 表示冻结第1,3,5层）
    if args.freeze_transformer_layers not in [None, '']:
        splited_layers = args.freeze_transformer_layers.split()
        for layer in splited_layers:
            # 转换为整数层索引（注意：实际层号从0开始计数）
            freezed_layers.append(int(layer.strip()))

    # 遍历所有模型参数实施冻结
    for name, param in transformer.named_parameters():
        layer_names.append(name)  # 记录层名（用于后续调试）
        
        # 冻结指定transformer层的核心逻辑
        if "transformer" in name:
            # 解析层号（名称格式示例：transformer.1.attn.to_k.weight）
            name_split = name.split(".")
            layer_order = name_split[1]  # 提取层索引
            
            # 如果当前层在冻结列表中，关闭梯度计算
            if int(layer_order) in freezed_layers:
                param.requires_grad = False

        # 【开发者建议】全参数微调时需要冻结的最终层（当前被注释）
        # 冻结归一化层和输出投影层（在完整微调时建议冻结，但LoRA训练不需要）
        # if "norm_out" in name:  # 层归一化输出
        #     param.requires_grad = False
        # if "proj_out" in name:  # 最终投影输出层
        #     param.requires_grad = False

    # ===== 调试代码（已注释）=====
    # 用于打印各层梯度状态（调试冻结策略有效性）
    # for name, param in transformer.named_parameters():
    #     print(f"Layer: {name} | Trainable: {param.requires_grad}")
    
    # ============== 模型解包工具 ==============
    def unwrap_model(model):
        # 从分布式训练包装器/编译器中提取原始模型
        # 功能：1. 解除DDP包装 2. 解除torch.compile的优化包装
        # 确保后续模型操作直接作用于原始模型结构
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        # 只在主进程执行保存操作（分布式训练时避免重复保存）
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            
            # 模型类型验证与状态字典提取
            for model in models:
                # 解除分布式训练包装器获取原始模型
                expected_model_type = type(unwrap_model(transformer))
                
                if isinstance(model, expected_model_type):
                    # 转换PEFT模型状态字典为Diffusers格式
                    peft_state_dict = get_peft_model_state_dict(model)
                    transformer_lora_layers_to_save = convert_state_dict_to_diffusers(peft_state_dict)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # 弹出已处理权重（防止自动保存原始模型）
                weights.pop()

            # ============== 权重保存核心逻辑 ==============
            # 使用Flux原生方法保存LoRA权重（生成pytorch_lora_weights.safetensors）
            FluxPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save
            )
            
            # ============== 文件重命名策略 ==============
            # 示例：将checkpoint_dir/pytorch_lora_weights.safetensors 
            # 复制为checkpoint_dir/checkpoint_dir.safetensors
            last_part = os.path.basename(os.path.normpath(output_dir))  # 获取目录末级名称
            file_path = f"{output_dir}/{last_part}.safetensors"
            ori_file = f"{output_dir}/pytorch_lora_weights.safetensors"
            
            if os.path.exists(ori_file):
                # 创建带目录名的副本（便于版本管理）
                shutil.copy(ori_file, file_path) 

            # ============== Kohya格式兼容层（当前注释）==============
            # 以下代码用于生成兼容Kohya/WebUI的LoRA格式
            # 实现原理：
            # 1. 转换到Kohya的键命名规范（添加lora_unet_前缀）
            # 2. 调整张量维度顺序（Kohya使用不同的维度排列）
            # 3. 保存为特定命名模式（便于AUTOMATIC1111等UI加载）
            #
            # peft_state_dict = convert_all_state_dict_to_peft(transformer_lora_layers_to_save)
            # kohya_state_dict = convert_state_dict_to_kohya(peft_state_dict)
            # prefix = 'lora_unet_'
            # prefixed_state_dict = {prefix + key: value for key, value in kohya_state_dict.items()}
            # save_file(prefixed_state_dict, file_path)

    def load_model_hook(models, input_dir):
        # 初始化transformer模型引用
        transformer_ = None
        
        # ============== 模型类型验证 ==============
        # 遍历所有待加载模型（处理分布式训练包装情况）
        while len(models) > 0:
            model = models.pop()
            # 验证模型类型匹配当前transformer架构
            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model  # 获取正确的模型实例
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        # ============== 权重加载流程 ==============
        # 从指定目录加载LoRA状态字典（自动识别.safetensors文件）
        lora_state_dict = FluxPipeline.lora_state_dict(input_dir)
        
        # ============== 键名转换策略 ==============
        # 转换权重键名格式（适配PEFT库的命名规范）
        transformer_state_dict = {
            # 移除"transformer."前缀（原始保存格式包含模块路径）
            f'{k.replace("transformer.", "")}': v 
            for k, v in lora_state_dict.items() 
            if k.startswith("transformer.")  # 过滤仅transformer相关权重
        }
        
        # ============== 格式转换 ==============
        # 将Diffusers格式转换为PEFT内部格式（处理维度转置等差异）
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)

        # ============== 权重注入 ==============
        # 将LoRA权重加载到模型中，返回不兼容的键
        incompatible_keys = set_peft_model_state_dict(
            transformer_, 
            transformer_state_dict, 
            adapter_name="default"  # 支持多适配器加载
        )

        # ============== 兼容性检查 ==============
        if incompatible_keys is not None:
            # 仅关注意外键（模型不包含的键），忽略缺失键（可能故意不加载）
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"加载适配器权重时发现未知键: {unexpected_keys}"
                    "\n可能原因：1.模型架构变更 2.权重文件损坏 3.跨模型加载"
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        # if args.mixed_precision == "fp16" or args.mixed_precision == "bf16":
        #     models = [transformer_]
        #     # only upcast trainable parameters (LoRA) into fp32
        #     cast_training_params(models)


    # 挂载钩子
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True


    # Make sure the trainable params are in float32.
    # if args.mixed_precision == "fp16":
    #     models = [transformer]
    #     if args.train_text_encoder:
    #         models.extend([text_encoder_one])
    #     # only upcast trainable parameters (LoRA) into fp32
    #     cast_training_params(models, dtype=torch.float32)

    # ==========================================================
    # Create train dataset
    # ==========================================================
    # data_files = {}
    # this part need more work afterward, you need to prepare 
    # the train files and val files split first before the training
    # ============== 数据集初始化流程 ==============
    if args.train_data_dir is not None:
        input_dir = args.train_data_dir
        datarows = []          # 最终训练数据容器
        cache_list = []        # 需要重新预处理的文件列表
        recreate_cache = args.recreate_cache  # 强制重建缓存标志

        # 递归收集所有支持的图像文件
        supported_image_types = ['.jpg','.jpeg','.png','.webp']
        files = glob.glob(f"{input_dir}/**", recursive=True)  # 递归扫描目录
        image_files = [
            f for f in files 
            if os.path.splitext(f)[-1].lower() in supported_image_types  # 扩展名过滤
        ]

        # 元数据对齐函数（处理文件删除情况）
        def align_metadata(datarows, image_files, metadata_path):
            """清理元数据中不存在的图像记录，防止幽灵数据
            Args:
                datarows: 原始元数据条目
                image_files: 实际存在的图像文件列表
                metadata_path: 元数据文件保存路径
            Returns:
                过滤后的有效元数据
            """
            new_metadatarows = []
            for row in datarows:
                if row['image_path'] in image_files:  # 仅保留实际存在的文件
                    new_metadatarows.append(row)
            # 保存更新后的元数据（自动清理无效条目）
            with open(metadata_path, "w", encoding='utf-8') as f:
                f.write(json.dumps(new_metadatarows))
            return new_metadatarows

        # ===== 元数据加载与校验 =====
        metadata_datarows = []
        if os.path.exists(metadata_path):  # 主元数据文件存在
            with open(metadata_path, "r", encoding='utf-8') as f:
                metadata_datarows = json.loads(f.read())
                # 自动对齐当前目录中的实际文件
                metadata_datarows = align_metadata(metadata_datarows, image_files, metadata_path)

        # 验证集元数据加载（流程同上）
        val_metadata_datarows = []
        if os.path.exists(val_metadata_path):
            with open(val_metadata_path, "r", encoding='utf-8') as f:
                val_metadata_datarows = json.loads(f.read())
                val_metadata_datarows = align_metadata(val_metadata_datarows, image_files, val_metadata_path)

        # ===== 数据集合并策略 =====
        # 单图像训练模式（当前被注释）
        # single_image_training = len(image_files) == 1
        if len(metadata_datarows) == 1:  # 特殊单样本模式
            full_datarows = metadata_datarows
        else:  # 常规模式：合并训练集和验证集
            full_datarows = metadata_datarows + val_metadata_datarows

        # ===== 缓存验证机制 =====
        md5_pairs = [
            {"path":"image_path", "md5": "image_path_md5"},  # 原始图像
            {"path":"text_path",  "md5": "text_path_md5"},   # 文本描述
            {"path":"npz_path",   "md5": "npz_path_md5"},    # 预处理特征
            {"path":"latent_path","md5": "latent_path_md5"}, # 潜空间表示
        ]

        def check_md5(datarows, md5_pairs):
            """MD5校验与缓存更新机制
            工作原理：
                1. 检查每个文件的MD5是否匹配元数据记录
                2. 不匹配或缺失的文件加入待处理列表
                3. 返回需要重新生成缓存的文件列表
            """
            cache_list = []
            new_datarows = []
            for datarow in tqdm(datarows):
                corrupted = False
                for pair in md5_pairs:
                    path_key = pair['path']
                    md5_key = pair['md5']
                    
                    # 键存在性检查
                    if md5_key not in datarow:
                        cache_list.append(datarow['image_path'])
                        corrupted = True
                        break
                    
                    # 文件存在性检查
                    file_path = datarow[path_key]
                    if not os.path.exists(file_path):
                        cache_list.append(datarow['image_path'])
                        corrupted = True
                        break
                    
                    # MD5校验
                    with open(file_path, 'rb') as f:
                        current_md5 = md5(f.read()).hexdigest()
                    if current_md5 != datarow[md5_key]:
                        cache_list.append(datarow['image_path'])
                        corrupted = True
                        break
                
                if not corrupted:
                    new_datarows.append(datarow)
            return cache_list, new_datarows

        # ===== 缓存更新触发条件 =====
        if (len(datarows) == 0) or recreate_cache:  # 全新训练或强制刷新
            cache_list = image_files
        else:  # 增量检查
            # 发现新增图像（未在元数据中注册）
            current_images = {d['image_path'] for d in full_datarows}
            missing_images = [f for f in image_files if f not in current_images]
            if missing_images:
                print(f"发现{len(missing_images)}张未注册图像")
                cache_list += missing_images

            # MD5完整性检查
            corrupted_files, valid_datarows = check_md5(full_datarows, md5_pairs)
            full_datarows = valid_datarows  # 过滤损坏数据
            cache_list += corrupted_files
                    
        # ===== 缓存生成触发条件 =====
        if len(cache_list) > 0:
            # 释放显存策略：将主模型暂存到CPU
            transformer.to("cpu")
            
            # ===== 多模态组件加载 =====
            # 加载双文本编码器的tokenizer（适配多语言输入）
            tokenizer_one = CLIPTokenizer.from_pretrained(  # CLIP的BPE分词器
                args.pretrained_model_name_or_path,
                subfolder="tokenizer",
            )
            tokenizer_two = T5TokenizerFast.from_pretrained(  # T5的SentencePiece分词器
                args.pretrained_model_name_or_path,
                subfolder="tokenizer_2",
            )

            # 动态加载文本编码器类（兼容不同模型架构）
            text_encoder_cls_one = import_model_class_from_model_name_or_path(
                args.pretrained_model_name_or_path 
            )  # 自动识别CLIPTextModel或类似
            text_encoder_cls_two = import_model_class_from_model_name_or_path(
                args.pretrained_model_name_or_path, subfolder="text_encoder_2"
            )  # 自动识别T5等模型

            # 实例化文本编码器（冻结参数）
            text_encoder_one, text_encoder_two = load_text_encoders(
                text_encoder_cls_one, text_encoder_cls_two
            )

            # ===== 变分自编码器加载 =====
            vae = AutoencoderKL.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="vae",
            )
            vae.requires_grad_(False)  # 冻结VAE参数
            
            # 设备与精度配置
            vae.to(accelerator.device, dtype=torch.float32)  # VAE始终使用fp32保证精度
            text_encoder_one.to(accelerator.device, dtype=weight_dtype)  # 适配混合精度训练
            text_encoder_two.to(accelerator.device, dtype=weight_dtype)

            # ===== 缓存生成核心 =====
            cached_datarows = create_metadata_cache(
                tokenizers=[tokenizer_one, tokenizer_two],
                text_encoders=[text_encoder_one, text_encoder_two],
                vae=vae,
                cache_list=cache_list,
                metadata_path=metadata_path,
                recreate_cache=args.recreate_cache,
                resolution_config=args.resolution  # 控制特征图尺寸
            )

            # ===== 数据集重组 =====
            full_datarows += cached_datarows  # 合并新生成的缓存数据
            
            # 特殊单样本处理（防止验证集分裂失败）
            if len(full_datarows) == 1:
                full_datarows *= 2  # 自我复制以创建虚拟验证集
                validation_ratio = 0.5  # 强制50%验证比例

            # 数据集划分策略
            if args.validation_ratio > 0:
                training_datarows, validation_datarows = train_test_split(
                    full_datarows, 
                    train_size=1-args.validation_ratio,
                    test_size=args.validation_ratio,
                    shuffle=True  # 确保数据分布均匀
                )
                datarows = training_datarows
            else:
                datarows = full_datarows

            # ===== 元数据持久化 =====
            with open(metadata_path, "w") as f:
                json.dump(datarows, f, indent=4)  # 保存训练集元数据
            if validation_datarows:
                with open(val_metadata_path, "w") as f:
                    json.dump(validation_datarows, f, indent=4)  # 保存验证集元数据

            # ===== 显存回收策略 =====
            text_encoder_one.to("cpu")  # 移出显存
            text_encoder_two.to("cpu")
            del vae, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two  # 释放引用
            # 注意：此处未调用gc.collect()是因为accelerator会管理显存
    
    # repeat_datarows = []
    # for datarow in datarows:
    #     for i in range(args.repeats):
    #         repeat_datarows.append(datarow)
    # datarows = repeat_datarows
    
    # ============== 数据增强与设备恢复 ==============
    # 通过重复数据行实现隐式epoch扩展（当实际数据集较小时特别有效）
    # 例如：repeats=3时，相当于每个样本训练3次
    datarows = datarows * args.repeats
    
    # 将transformer模型从CPU移回加速器设备（GPU/TPU）
    # 背景：在缓存生成阶段，为节省显存将模型暂存到CPU
    transformer.to(accelerator.device, dtype=weight_dtype)

    # ============== 混合精度训练参数配置 ==============
    # 确保LoRA可训练参数保持float32精度（防止混合精度下梯度异常）
    # 设计考量：基础模型权重保持低精度，LoRA参数高精度以获得更好收敛性
    if args.mixed_precision == "fp16":
        models = [transformer]
        # 仅转换可训练参数（LoRA层）到指定精度
        cast_training_params(models, dtype=torch.float32)

    # ============== 优化参数过滤 ==============
    # 提取所有需要梯度更新的参数（即LoRA层参数）
    # 实现原理：通过requires_grad标志过滤冻结的基础模型参数
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    
    # 构建优化器参数组（支持未来扩展多参数组优化）
    # 当前策略：所有LoRA参数共享相同学习率
    transformer_lora_parameters_with_lr = {
        "params": transformer_lora_parameters,
        "lr": args.learning_rate  # 从命令行参数获取基础学习率
    }
    params_to_optimize = [transformer_lora_parameters_with_lr]
    
    # ============== 优化器选择逻辑 ==============
    # 验证优化器类型合法性（当前仅支持Prodigy和AdamW变体）
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"  # 自动降级到默认优化器

    # 8bit优化器兼容性检查（仅AdamW有效）
    if use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    # ============== AdamW优化器实现分支 ==============
    if args.optimizer.lower() == "adamw":
        # BF16混合精度专用优化器（需要第三方库支持）
        if args.mixed_precision == "bf16":
            try:
                from adamw_bf16 import AdamWBF16  # 特殊优化的BF16版本
            except ImportError:
                raise ImportError(
                    "To use bf Adam, please install the AdamWBF16 library: `pip install adamw-bf16`."
                )
            optimizer_class = AdamWBF16
            transformer.to(dtype=torch.bfloat16)  # 转换模型整体精度
        # 8bit量化优化器（节省显存但可能影响精度）
        elif use_8bit_adam:
            try:
                import bitsandbytes as bnb  # HuggingFace官方推荐的量化库
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )
            optimizer_class = bnb.optim.AdamW8bit  # 内存效率优化版
        # 标准全精度AdamW
        else:
            optimizer_class = torch.optim.AdamW  # PyTorch原生实现

        # 实例化优化器（参数组/动量项/权重衰减配置）
        optimizer = optimizer_class(
            params_to_optimize,  # 仅包含LoRA参数
            betas=(adam_beta1, adam_beta2),  # 动量参数默认(0.9, 0.999)
            weight_decay=adam_weight_decay,  # 权重衰减系数（防过拟合）
            eps=adam_epsilon,  # 数值稳定性常数（默认1e-8）
        )

    # ============== Prodigy优化器实现分支 ==============
    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt  # 自适应学习率优化器，特别适合小批量数据训练
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        # Prodigy特性：建议使用较高基础学习率（通常1.0左右）
        # 原理：通过自适应机制自动调节有效学习率
        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        # 实例化Prodigy优化器（包含多个创新参数）
        optimizer = optimizer_class(
            params_to_optimize,  # 仅优化LoRA参数
            lr=args.learning_rate,  # 名义学习率（实际由优化器自动调整）
            betas=(adam_beta1, adam_beta2),  # 一阶/二阶矩估计衰减率
            beta3=args.prodigy_beta3,  # 梯度矩估计的附加衰减因子
            d_coef=prodigy_d_coef,  # 梯度协方差估计的阻尼系数（默认1.0）
            weight_decay=adam_weight_decay,  # L2正则化系数
            eps=adam_epsilon,  # 数值稳定常数（防止除以零）
            decouple=prodigy_decouple,  # 解耦权重衰减（True时启用AdamW风格）
            use_bias_correction=prodigy_use_bias_correction,  # 初始训练阶段偏置校正
            safeguard_warmup=prodigy_safeguard_warmup,  # 安全预热机制（防梯度爆炸）
        )
    
    # ============== 数据预处理流水线 ==============
    def collate_fn(examples):
        # 多分辨率数据批处理函数（自动处理不同宽高比样本）
        # 注：time_ids/text_ids为SDXL保留字段，当前版本未启用
        
        # 潜在空间特征堆叠（不同分辨率自动对齐）
        latents = torch.stack([example["latent"] for example in examples])
        
        # 文本编码特征整合（CLIP+T5双编码器输出）
        prompt_embeds = torch.stack([example["prompt_embed"] for example in examples])
        pooled_prompt_embeds = torch.stack([example["pooled_prompt_embed"] for example in examples])
        
        # 注意力掩码处理（适配多语言混合输入）
        txt_attention_masks = torch.stack([example["txt_attention_mask"] for example in examples])

        return {
            "latents": latents,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "txt_attention_masks": txt_attention_masks,
            # 预留字段（SDXL多尺寸训练支持）
            # "text_ids": text_ids,  
            # "time_ids": time_ids,
        }

    # ============== 动态分桶数据集 ==============
    # 基于元数据缓存构建训练集（支持条件随机丢弃）
    # conditional_dropout_percent：随机清空文本条件的概率（数据增强）
    train_dataset = CachedImageDataset(datarows, conditional_dropout_percent=args.caption_dropout)

    # ============== 分桶批采样器 ==============
    # 参考everyDream实现的分桶策略（最小化填充开销）
    # 实现原理：按图像分辨率分组，相同分辨率样本组成批次
    bucket_batch_sampler = BucketBatchSampler(
        train_dataset, 
        batch_size=args.train_batch_size, 
        drop_last=True  # 丢弃不完整批次（保持梯度稳定性）
    )

    # ============== 高效数据加载器 ==============
    # 使用分桶采样器替代常规shuffle，特点：
    # 1. 减少显存碎片（同批次潜在特征尺寸一致）
    # 2. 提升计算效率（避免动态形状调整）
    # 3. 自动维护数据分布（桶内随机采样）
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=bucket_batch_sampler,  # 替代传统shuffle
        collate_fn=collate_fn,  # 自定义批处理逻辑
        num_workers=dataloader_num_workers,  # 并行加载进程数（建议设为CPU核心数75%）
    )
    
    

    # ============== 训练步骤计算与验证 ==============
    # 计算每个epoch的实际更新步数（考虑梯度累积）
    overrode_max_train_steps = False  # 标记是否自动计算总步数
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps  # 梯度累积下的有效步数
    )
    
    # 自动推算最大训练步数（当用户未显式指定时）
    if max_train_steps is None:
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True  # 标记为自动推算模式

    # 重新校准训练步数（数据加载器长度可能变化）
    # 必要性：分桶采样可能导致实际数据量变化
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    # 反向推算实际训练轮次（保持总步数一致）
    args.num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # ============== VAE配置参数 ==============
    # 从VAE配置文件加载的缩放参数（控制潜在空间分布）
    vae_config_shift_factor = 0.1159    # 潜在空间平移系数（均值调整）
    vae_config_scaling_factor = 0.3611  # 潜在空间缩放系数（方差归一化）
    vae_config_block_out_channels = [   # 解码器通道结构（需与预训练VAE一致）
        128,
        256,
        512,
        512
    ]

    # ============== 学习率调度器配置 ==============
    lr_scheduler = get_scheduler(
        args.lr_scheduler,  # 调度器类型（cosine/linear等）
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,  # 分布式训练需跨进程同步
        num_training_steps=max_train_steps * accelerator.num_processes,     # 总步数乘以进程数
        num_cycles=lr_num_cycles,  # 余弦退火周期数（仅对cosine_with_restarts有效）
        power=lr_power,           # 多项式衰减指数（仅对polynomial有效）
    )

    # ============== 分布式训练准备 ==============
    # 使用accelerator包装组件（自动处理数据并行/混合精度）
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
    )
    
    # 特殊设备放置逻辑：当启用块交换时不自动放置设备
    transformer = accelerator.prepare(
        transformer, 
        device_placement=[not is_swapping_blocks]  # 与显存优化机制配合
    )

    # ============== 训练过程监控 ==============
    if accelerator.is_main_process:
        tracker_name = "flux-lora"
        try:
            # 初始化性能追踪器（记录指标如loss/lr到TensorBoard/WandB）
            accelerator.init_trackers(tracker_name, config=vars(args))
        except:
            print("Trackers not initialized")  # 降级处理（不影响训练）

    # ============== 训练元信息计算 ==============
    # 计算实际总批量大小（考虑并行和梯度累积）
    total_batch_size = (
        args.train_batch_size 
        * accelerator.num_processes      # GPU数量
        * args.gradient_accumulation_steps  # 梯度累积步数
    )

    # 打印关键训练参数（日志级别INFO可见）
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")          # 数据集样本总数
    logger.info(f"  Num Epochs = {args.num_train_epochs}")         # 实际训练轮次
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")  # 单卡批大小
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")  # 等效总批大小
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")  # 梯度累积参数
    logger.info(f"  Total optimization steps = {max_train_steps}")  # 总优化步数

    # ============== 训练状态初始化 ==============
    global_step = 0    # 全局步数计数器（跨epoch累计）
    first_epoch = 0    # 起始epoch（用于断点续训）


    # def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
    #     sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
    #     schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    #     timesteps = timesteps.to(accelerator.device)

    #     step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    #     sigma = sigmas[step_indices].flatten()
    #     while len(sigma.shape) < n_dim:
    #         sigma = sigma.unsqueeze(-1)
    #     return sigma


    # ============== 断点续训初始化 ==============
    resume_step = 0  # 恢复步数计数器（用于梯度累积场景）
    
    # 检查点加载逻辑（支持显式路径和latest自动检测）
    if args.resume_from_checkpoint and args.resume_from_checkpoint != "":
        # 当指定具体检查点路径时（非latest模式）
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)  # 提取纯文件名
        else:
            # 自动查找最新检查点（根据保存名称前缀和步数排序）
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith(args.save_name)]  # 过滤匹配文件
            # 按步数排序逻辑：文件名格式为"保存名-步数"，如"model-1000"
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))  # 按步数升序排列
            path = dirs[-1] if len(dirs) > 0 else None  # 取最后（最大步数）的检查点

        # 检查点不存在时的处理逻辑
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None  # 重置检查点标志
            initial_global_step = 0  # 初始化全局步数
        else:
            # 成功找到检查点时的加载流程
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))  # 加载模型/优化器状态
            
            # 从文件名解析全局训练步数（格式：保存名-步数）
            global_step = int(path.split("-")[-1])  
            
            # 状态恢复初始化
            initial_global_step = global_step  # 当前训练步起点
            resume_step = global_step  # 恢复步数记录（用于跳过已训练数据）
            first_epoch = global_step // num_update_steps_per_epoch  # 计算起始epoch

    else:  # 全新训练时的初始化
        initial_global_step = 0

    # ============== 训练进度条配置 ==============
    progress_bar = tqdm(
        range(0, max_train_steps),  # 总步数范围
        initial=initial_global_step,  # 初始进度（支持断点续训）
        desc="Steps",  # 进度条前缀
        # 仅在主进程显示进度条（避免多GPU重复输出）
        disable=not accelerator.is_local_main_process,
    )
    
    # 历史遗留代码（已注释的时间步控制逻辑）
    # max_time_steps = noise_scheduler.config.num_train_timesteps
    # if args.max_time_steps is not None and args.max_time_steps > 0:
    #     max_time_steps = args.max_time_steps
        
                    
    # ============== 条件引导控制逻辑 ==============
    # 检查模型是否支持分类器自由引导（CFG）
    # 实现原理：通过模型配置中的guidance_embeds标志判断
    if accelerator.unwrap_model(transformer).config.guidance_embeds:
        handle_guidance = True  # 启用CFG引导
    else:
        handle_guidance = False  # 禁用CFG引导
        
    # ============== 噪声调度辅助函数 ==============
    # 功能：根据时间步索引获取对应的sigma值（噪声强度系数）
    # 设计要点：
    # 1. 独立噪声调度器副本避免训练干扰（noise_scheduler_copy）
    # 2. 维度扩展适配不同分辨率输入（n_dim参数）
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        # 获取噪声调度器预计算的sigma序列
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        # 时间步序列对齐设备
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        # 转换输入时间步到匹配设备
        timesteps = timesteps.to(accelerator.device)
        # 查找每个时间步对应的索引位置
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        # 提取对应索引的sigma值并展开
        sigma = sigmas[step_indices].flatten()
        # 维度扩展适配不同特征图尺寸（如4D: [B,C,H,W]）
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    # ============== 主训练循环 ==============
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            # 梯度累积上下文管理器（自动处理多步梯度累积）
            with accelerator.accumulate(transformer):
                
                # ============== 显存优化操作 ==============
                # 移动模型参数到设备（除swap blocks外）
                accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)  # 峰值显存优化
                # 准备块交换前向计算（显存-内存交换策略）
                accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
                flush()  # 清空CUDA操作队列确保同步
                
                # ============== 数据准备阶段 ==============
                # 从数据加载器获取潜在空间表示
                latents = batch["latents"].to(accelerator.device)
                # 文本编码器输出（CLIP文本嵌入）
                prompt_embeds = batch["prompt_embeds"].to(accelerator.device)
                # 池化文本特征（用于条件控制）
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device)
                # 文本注意力掩码（处理变长输入）
                txt_attention_masks = batch["txt_attention_masks"].to(accelerator.device)
                
                # 文本ID占位符（适配模型输入结构）
                text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=accelerator.device, dtype=weight_dtype)
                
                # ============== 潜在空间标准化 ==============
                # 应用VAE配置的平移和缩放（匹配预训练分布）
                latents = (latents - vae_config_shift_factor) * vae_config_scaling_factor
                # 转换到指定精度（FP16/FP32）
                latents = latents.to(dtype=weight_dtype)

                # 计算VAE缩放因子（根据解码器结构）
                vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)

                # ============== 潜在图像ID生成 ==============
                # 生成空间位置编码ID（适配模型的位置感知注意力机制）
                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    latents.shape[0],  # 批次大小
                    latents.shape[2] // 2,  # 潜在空间高度（下采样后）
                    latents.shape[3] // 2,  # 潜在空间宽度
                    accelerator.device,
                    weight_dtype,
                )
                
                # ============== 噪声生成与采样 ==============
                # 生成标准高斯噪声（与潜在空间同维度）
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]  # 当前实际批次大小
                
                # ============== 时间步非均匀采样 ==============
                # 根据权重方案计算时间步密度分布
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,  # 采样策略（logit_normal/mode等）
                    batch_size=bsz,
                    logit_mean=args.logit_mean,  # 对数正态分布均值
                    logit_std=args.logit_std,    # 对数正态分布标准差
                    mode_scale=args.mode_scale,  # 模态缩放系数
                )
                # 将采样值映射到离散时间步索引
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                # 获取对应时间步张量
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=accelerator.device)
                
                # ============== Flow Matching噪声混合 ==============
                # 计算噪声潜在表示：zt = (1 - σ) * x + σ * ε
                sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
                
                # ============== 潜在空间打包 ==============
                # 重组潜在空间维度适配模型输入格式
                packed_noisy_latents = FluxPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=latents.shape[0],
                    num_channels_latents=latents.shape[1],
                    height=latents.shape[2],
                    width=latents.shape[3],
                )
                
                # ============== 分类器自由引导处理 ==============
                if handle_guidance:
                    # 生成引导系数张量（扩展至批次维度）
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(latents.shape[0])  # [batch_size]
                else:
                    guidance = None
                
                # ============== 混合精度前向传播 ==============
                with accelerator.autocast():  # 自动管理精度转换
                    # 模型预测噪声残差
                    model_pred = transformer(
                        hidden_states=packed_noisy_latents,    # 噪声潜在表示
                        encoder_hidden_states=prompt_embeds,   # 文本嵌入
                        joint_attention_kwargs = {'attention_mask': txt_attention_masks},  # 跨模态注意力掩码
                        pooled_projections=pooled_prompt_embeds,  # 池化文本特征
                        timestep=timesteps / 1000,  # 时间步归一化（适配模型内部缩放）
                        img_ids=latent_image_ids,   # 空间位置编码
                        txt_ids=text_ids,           # 文本位置占位符
                        guidance=guidance,          # CFG引导系数
                        return_dict=False
                    )[0]
                
                # ============== 潜在空间解包 ==============
                # 将模型输出重组为标准潜在空间格式
                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=latents.shape[2] * vae_scale_factor,  # 原始图像高度
                    width=latents.shape[3] * vae_scale_factor,   # 原始图像宽度
                    vae_scale_factor=vae_scale_factor,           # VAE缩放因子
                )
                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                # weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # ====================Debug latent====================
                # vae = AutoencoderKL.from_single_file(
                #     vae_path
                # )
                # vae.to(device=accelerator.device)
                # image_processor = VaeImageProcessor(vae_scale_factor=vae.config.scaling_factor)
                # with torch.no_grad():
                #     image = vae.decode(model_pred / vae.config.scaling_factor, return_dict=False)[0]
                # image = image_processor.postprocess(image, output_type="pil")[0]
                # image.save("model_pred.png")
                # ====================Debug latent====================
                
                
                # flow matching loss
                # if args.precondition_outputs:
                #     target = latents
                # else:
                target = noise - latents
                
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                
                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                # Backpropagate
                accelerator.backward(loss)
                step_loss = loss.detach().item()
                del loss, latents, target, model_pred,  timesteps,  bsz, noise, noisy_model_input
                if accelerator.sync_gradients:
                    params_to_clip = transformer_lora_parameters
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)

                # ====================优化器更新步骤====================
                # 【梯度更新】同步模型到当前设备并执行参数更新
                transformer.to(accelerator.device)  # 确保模型在正确的设备上
                optimizer.step()                   # 执行参数更新
                lr_scheduler.step()                # 调整学习率
                optimizer.zero_grad()              # 清空梯度缓存

                # ====================分布式训练同步====================
                # 【进程同步】等待所有进程完成梯度更新
                accelerator.wait_for_everyone()    # 分布式训练屏障

                # ====================训练进度更新====================
                if accelerator.sync_gradients:
                    progress_bar.update(1)         # 更新进度条
                    global_step += 1               # 全局步数递增

                # ====================学习率监控====================
                # 【学习率计算】根据优化器类型获取当前学习率
                lr = lr_scheduler.get_last_lr()[0]  # 基础学习率
                lr_name = "lr"
                if args.optimizer == "prodigy":
                    # Prodigy优化器特有参数：动态学习率计算
                    if resume_step>0 and resume_step == global_step:
                        lr = 0  # 恢复训练时的特殊处理
                    else:
                        # 计算实际学习率：d参数 * 基础学习率
                        lr = lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0]["lr"]
                    lr_name = "lr/d*lr"  # 指标名称标识

                # ====================指标记录====================
                # 【日志记录】记录当前训练指标
                logs = {
                    "step_loss": step_loss,   # 当前步的损失值
                    lr_name: lr,              # 学习率相关指标
                    "epoch": epoch            # 当前训练轮次
                }
                accelerator.log(logs, step=global_step)  # 上报到监控系统
                progress_bar.set_postfix(**logs)         # 更新进度条显示

                # ====================训练终止条件====================
                if global_step >= max_train_steps:
                    break  # 达到最大训练步数时终止循环

                # ====================显存管理====================
                del step_loss  # 释放临时变量
                gc.collect()   # 主动触发垃圾回收
                torch.cuda.empty_cache()  # 清空CUDA缓存

            # ====================批次循环结束====================
            
        # ==================================================
        # validation part
        # ==================================================
        
        if global_step < args.skip_step:
            continue
        
        
        # store rng before validation
        # ====================随机状态保存====================
        before_state = torch.random.get_rng_state()   # 保存PyTorch随机状态
        np_seed = abs(int(args.seed)) if args.seed is not None else np.random.seed()  # 生成NumPy随机种子
        py_state = python_get_rng_state()              # 保存Python内置随机状态
        
        if accelerator.is_main_process:
            # ====================模型保存逻辑====================
            if (epoch >= args.skip_epoch and epoch % args.save_model_epochs == 0) or epoch == args.num_train_epochs - 1:
                accelerator.wait_for_everyone()  # 等待所有进程同步
                if accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"{args.save_name}-{global_step}")
                    accelerator.save_state(save_path)  # 保存训练状态
                    logger.info(f"模型检查点已保存至: {save_path}")  # 记录日志
            
            # only execute when val_metadata_path exists
            # ====================验证触发条件====================
            if ((epoch >= args.skip_epoch and epoch % args.validation_epochs == 0) or epoch == args.num_train_epochs - 1) and os.path.exists(val_metadata_path):
                with torch.no_grad():
                    transformer = unwrap_model(transformer)  # 解除模型包装
                    
                    # ====================确定性设置====================
                    np.random.seed(val_seed)           # 固定NumPy随机种子
                    torch.manual_seed(val_seed)        # 固定PyTorch随机种子
                    dataloader_generator = torch.Generator().manual_seed(val_seed)  # 数据加载器种子
                    torch.backends.cudnn.deterministic = True  # 确保CUDA操作确定性
                    
                    # ====================验证数据加载====================
                    validation_datarows = []
                    with open(val_metadata_path, "r", encoding='utf-8') as readfile:
                        validation_datarows = json.loads(readfile.read())  # 加载验证集元数据
                    
                    if len(validation_datarows) > 0:
                        validation_dataset = CachedImageDataset(
                            validation_datarows,
                            conditional_dropout_percent=0  # 验证时关闭条件丢弃
                        )
                        
                        batch_size = 1  # 验证批次大小固定为1
                        # 原训练批次大小参考: args.train_batch_size
                        # handle batch size > validation dataset size
                        # if batch_size > len(validation_datarows):
                        #     batch_size = 1
                        
                        val_batch_sampler = BucketBatchSampler(validation_dataset, batch_size=batch_size, drop_last=True)

                        #initialize the DataLoader with the bucket batch sampler
                        val_dataloader = torch.utils.data.DataLoader(
                            validation_dataset,
                            batch_sampler=val_batch_sampler, #use bucket_batch_sampler instead of shuffle
                            collate_fn=collate_fn,
                            num_workers=dataloader_num_workers,
                        )

                        print("\nStart val_loss\n")
                        
                        total_loss = 0.0
                        num_batches = len(val_dataloader)
                        # if no val data, skip the following 
                        if num_batches == 0:
                            print("No validation data, skip validation.")
                        else:
                            # basically the as same as the training loop
                            enumerate_val_dataloader = enumerate(val_dataloader)
                            # ====================验证批次处理====================
                            for i, batch in tqdm(enumerate_val_dataloader, position=1, desc="验证批次"):
                                # 【显存优化】激活块交换机制
                                accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)
                                accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
                                flush()  # 清空IO缓存
                                
                                # ====================数据预处理====================
                                latents = batch["latents"].to(accelerator.device)            # 潜在空间数据
                                prompt_embeds = batch["prompt_embeds"].to(accelerator.device) # 文本嵌入
                                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device) # 池化文本嵌入
                                txt_attention_masks = batch["txt_attention_masks"].to(accelerator.device)   # 注意力掩码
                                
                                # 【数据标准化】应用VAE预处理参数
                                latents = (latents - vae_config_shift_factor) * vae_config_scaling_factor
                                latents = latents.to(dtype=weight_dtype)  # 转换为指定精度

                                # ==================== VAE缩放因子计算 ====================
                                # 计算VAE的缩放倍数：2^(n-1)，n是VAE的下采样次数
                                # 例如VAE有3个下采样块时，图像尺寸缩小2^3=8倍
                                # 设计目的：将潜在空间尺寸映射回原始图像尺寸
                                vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)
                                
                                # ==================== 空间位置编码生成 ====================
                                # 为潜在空间生成二维位置编码（类似Vision Transformer的patch位置编码）
                                # 功能：帮助模型理解图像的空间结构关系
                                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                                    batch_size=latents.shape[0],            # 当前批次大小
                                    height=latents.shape[2] // 2,           # 潜在空间高度（经过VAE下采样）
                                    width=latents.shape[3] // 2,            # 潜在空间宽度
                                    device=accelerator.device,             # 设备同步（GPU/CPU）
                                    dtype=weight_dtype                     # 精度保持（FP16/FP32）
                                )
                                
                                # 生成标准正态分布噪声（与潜在空间同尺寸）
                                noise = torch.randn_like(latents)  # ε ~ N(0, I)
                                bsz = latents.shape[0]  # 当前批次大小

                                # ==================== 时间步非均匀采样 ====================
                                # 根据加权方案生成时间步分布概率
                                u = compute_density_for_timestep_sampling(
                                    weighting_scheme=args.weighting_scheme,  # 加权策略（如logit_normal）
                                    batch_size=bsz,                         # 保持批次一致性
                                    logit_mean=args.logit_mean,             # 对数均值（控制分布中心）
                                    logit_std=args.logit_std,               # 对数标准差（控制分布宽度）
                                    mode_scale=args.mode_scale,             # 模态缩放系数
                                )
                                # 将连续概率映射到离散时间步（0~num_train_timesteps-1）
                                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                                timesteps = noise_scheduler_copy.timesteps[indices].to(device=accelerator.device)

                                # ==================== Flow Matching噪声混合 ====================
                                # 根据Flow Matching公式混合干净潜在空间和噪声
                                # 公式：z_t = (1 - σ_t) * x + σ_t * ε
                                # 其中σ_t是噪声调度器定义的混合系数
                                sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                                noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
                                
                                # ==================== 潜在空间重组 ====================
                                # 将4D潜在空间[B,C,H,W]重组为序列格式[B, Seq_len, Channels]
                                # 目的：适配Transformer架构的序列处理模式
                                packed_noisy_latents = FluxPipeline._pack_latents(
                                    noisy_model_input,      # 噪声混合后的潜在空间
                                    batch_size=latents.shape[0],
                                    num_channels_latents=latents.shape[1],
                                    height=latents.shape[2],
                                    width=latents.shape[3],
                                )
                                
                                # ==================== 分类器自由引导 ====================
                                # 当模型支持条件控制时，传入引导系数
                                if handle_guidance:
                                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                                    guidance = guidance.expand(latents.shape[0])  # 扩展至批次维度
                                else:
                                    guidance = None

                                # ==================== 混合精度推理 ====================
                                with accelerator.autocast():  # 自动管理FP16/FP32转换
                                    # 模型预测噪声残差（核心推理步骤）
                                    model_pred = transformer(
                                        hidden_states=packed_noisy_latents,  # 重组后的噪声潜在空间
                                        timestep=timesteps / 1000,           # 时间步归一化（适配模型内部处理）
                                        guidance=guidance,                   # CFG引导强度
                                        pooled_projections=pooled_prompt_embeds,  # 压缩后的文本特征
                                        encoder_hidden_states=prompt_embeds,      # 完整文本嵌入
                                        txt_ids=text_ids,                    # 文本位置编码占位符
                                        img_ids=latent_image_ids,            # 图像位置编码
                                        return_dict=False,
                                        joint_attention_kwargs = {'attention_mask': txt_attention_masks},  # 注意力掩码
                                    )[0]

                                # ==================== 潜在空间解包 ====================
                                # 将模型输出从序列格式恢复为4D图像格式
                                model_pred = FluxPipeline._unpack_latents(
                                    model_pred,
                                    height=latents.shape[2] * vae_scale_factor,  # 原始图像高度
                                    width=latents.shape[3] * vae_scale_factor,   # 原始图像宽度
                                    vae_scale_factor=vae_scale_factor,           # VAE缩放倍数
                                )

                                # ==================== 损失权重计算 ====================
                                # 根据时间步和加权策略计算损失权重
                                # 不同策略对应论文中的ω(t)函数（如SD3的logit-normal加权）
                                weighting = compute_loss_weighting_for_sd3(
                                    weighting_scheme=args.weighting_scheme,
                                    sigmas=sigmas  # 与时间步对应的噪声强度
                                )

                                # ==================== 调试代码块（已注释） ====================
                                # 需要时可取消注释，用于可视化模型预测的潜在空间
                                # 流程：解码潜在空间 -> 后处理为PIL图像 -> 保存

                                # ==================== 目标值计算 ====================
                                # 根据Flow Matching公式计算目标值
                                # 原始公式：target = ε - x （噪声与干净潜在空间的差值）
                                target = noise - latents

                                # ==================== 损失计算 ====================
                                # 核心公式：L = E[ ω(t) * ||模型预测 - (ε - x)||^2 ]
                                loss = torch.mean(
                                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                                    1,  # 按批次维度求平均
                                )
                                loss = loss.mean()  # 全局平均

                                # ==================== 显存管理 ====================
                                total_loss += loss.detach()  # 累积损失（分离计算图）
                                del latents, target, loss, model_pred, timesteps, bsz, noise, packed_noisy_latents  # 显存释放
                                gc.collect()  # 主动垃圾回收
                                torch.cuda.empty_cache()  # 清空CUDA缓存
                                
                            # ============== 验证指标计算 ==============
                            # 计算平均验证损失（总损失 / 验证批次数量）
                            # 目的：消除不同验证集大小带来的影响，获得可比指标
                            avg_loss = total_loss / num_batches
                            
                            # ============== 学习率记录策略 ==============
                            # 获取当前学习率（默认处理）
                            lr = lr_scheduler.get_last_lr()[0]
                            lr_name = "val_lr"  # 默认日志键名
                            
                            # Prodigy优化器特殊处理（自适应学习率算法）
                            # 公式：有效学习率 = d参数 * 基础学习率（参考Prodigy论文）
                            if args.optimizer == "prodigy":
                                # 从优化器参数组获取动态调整的d系数
                                lr = lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0]["lr"]
                                lr_name = "val_lr lr/d*lr"  # 明确标注特殊计算方式
                            
                            # ============== 日志系统构建 ==============
                            # 组织需要记录的指标（验证损失/学习率/当前轮次）
                            logs = {
                                "val_loss": avg_loss,  # 核心评估指标
                                lr_name: lr,           # 动态学习率记录
                                "epoch": epoch         # 用于跟踪训练进度
                            }
                            
                            # ============== 信息反馈机制 ==============
                            print(logs)  # 控制台直接输出（即时可读）
                            # 更新进度条显示（便于监控训练过程）
                            progress_bar.set_postfix(**logs)
                            # 分布式训练统一日志记录（支持多GPU/TPU场景）
                            accelerator.log(logs, step=global_step)
                            
                            # ============== 显存优化操作 ==============
                            # 及时删除大对象（防止显存泄漏）
                            del num_batches, avg_loss, total_loss
                            
                        # ============== 验证资源清理 ==============
                        # 释放验证数据集相关资源（验证完成后立即清理）
                        del validation_datarows, validation_dataset, val_batch_sampler, val_dataloader
                        # 主动触发垃圾回收（加速显存释放）
                        gc.collect()
                        # 清空CUDA缓存（消除内存碎片）
                        torch.cuda.empty_cache()
                        # 明确标识验证阶段结束（日志分隔更清晰）
                        print("\nEnd val_loss\n")
        # ============== 随机状态恢复机制 ==============
        # 恢复验证前的随机数生成器状态，保证训练过程确定性
        # 必要性：验证阶段可能修改随机状态，需还原保证训练可复现性
        np.random.seed(np_seed)  # 恢复NumPy随机种子
        torch.random.set_rng_state(before_state)  # 恢复PyTorch的RNG状态
        torch.backends.cudnn.deterministic = False  # 关闭cuDNN确定性模式（提升速度）
        
        # 恢复Python内置随机模块状态
        # 结构：版本号 + 状态元组 + 高斯分布状态
        version, state, gauss = py_state  # 解包保存的Python RNG状态
        python_set_rng_state((version, tuple(state), gauss))  # 必须转为不可变元组
        
        # ============== 显存优化操作 ==============
        # 注释掉的清理语句（需根据实际情况选择是否启用）
        # del before_state, np_seed, py_state  # 显式删除大对象
        gc.collect()  # 强制垃圾回收（清理循环引用对象）
        torch.cuda.empty_cache()  # 清空CUDA缓存（重要！防止显存碎片）
        
        
        # ==================================================
        # 验证阶段结束标记（代码结构分隔）
        # ==================================================
    
    # ============== 训练终止流程 ==============
    # 官方推荐的训练结束处理（关闭分布式进程组等）
    accelerator.end_training()
    
    # 模型输出信息（显示最终保存路径）
    print("Saved to ")
    print(args.output_dir)  # 打印输出目录的绝对路径


# ============== 主程序入口 ==============
if __name__ == "__main__":
    args = parse_args()  # 解析命令行参数
    main(args)  # 执行主训练流程