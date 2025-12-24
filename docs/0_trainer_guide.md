# MiniMind Trainer 模块完全指南

本文档位于 `docs/trainer_guide.md`，旨在帮助开发者深入理解 MiniMind 的核心训练模块。

## 1. Trainer 脚本概览

`trainer/` 目录下包含了 MiniMind 所有阶段的训练脚本。每个脚本对应 LLM 全生命周期的一个特定环节。

| 脚本名称 | 功能描述 | 核心作用 |
| :--- | :--- | :--- |
| `train_pretrain.py` | **预训练 (Pre-training)** | 从零开始训练，让模型学习通用知识（词语接龙）。这是 LLM 的地基。 |
| `train_full_sft.py` | **全参监督微调 (Full SFT)** | 在预训练基础上，通过高质量问答对进行全参数微调，教会模型对话指令。 |
| `train_lora.py` | **LoRA 微调** | 使用低秩适应 (LoRA) 技术，仅训练极少量参数（通常<1%），高效注入特定领域知识。 |
| `train_dpo.py` | **直接偏好优化 (DPO)** | RLHF 的一种替代方案。通过比较由人类/模型生成的"好/坏"样本对，让模型对齐人类偏好。 |
| `train_distillation.py`| **模型蒸馏** | 使用教师模型（大模型）指导学生模型（小模型 MiniMind），使其模仿大模型的概率分布。 |
| `train_distill_reason.py`| **推理能力蒸馏** | 专门针对推理任务（Reasoning）的蒸馏脚本，如 DeepSeek-R1 的复现。 |
| `train_ppo.py` | **PPO 强化学习** | 经典的 RLHF 算法，包含 Actor, Critic, Reward, Reference 四个模型，流程最复杂。 |
| `train_grpo.py` | **GRPO 强化学习** | DeepSeek 提出的高效 RL 算法，去除了 Critic 模型，通过 Group 采样计算优势。 |
| `train_spo.py` | **SPO 强化学习** | 简化的偏好优化算法。 |
| `trainer_utils.py` | **通用工具库** | **所有脚本的基石**。包含日志记录、分布式初始化、模型保存、学习率调度等公共函数。 |

---

## 2. 推荐学习路径

从入门到精通，建议按照以下顺序阅读和运行代码：

### 第一阶段：基础构建 (必读)
1.  **`train_pretrain.py`**: 理解 LLM 是如何从海量无监督文本中"诞生"的。重点关注 Dataset 的构建（全量 Loss）。
2.  **`train_full_sft.py`**: 理解如何让模型学会"说话"。重点关注 `loss_mask` 的处理技巧。

### 第二阶段：高效微调 (实用)
3.  **`train_lora.py`**: 学习 PEFT (Parameter-Efficient Fine-Tuning) 技术。观察代码是如何冻结主模型并注入 Adapter 的。

### 第三阶段：价值对齐 (进阶)
4.  **`train_dpo.py`**: 理解目前最流行的对齐算法。代码量少，逻辑清晰，是入门 RLHF/Alignment 的最佳起点。
5.  **`train_grpo.py`**: 学习 DeepSeek 系列所采用的前沿 RL 算法。

---

## 3. 代码执行堆栈与系统指南

MiniMind 的代码设计遵循"去抽象化"原则，调用链路非常直观。以下是典型的执行流：

### 3.1 核心执行链路
```mermaid
graph TD
    A[入口: train_xxx.py] --> B[配置: MiniMindConfig]
    A --> C[初始化: init_distributed_mode]
    A --> D[模型: model_minimind.py]
    A --> E[数据: dataset/lm_dataset.py]
    E --> F[加载器: DataLoader + DistributedSampler]
    A --> G[循环: train_epoch]
    G --> H[前向: model(inputs)]
    G --> I[反向: scaler.scale(loss).backward]
    G --> J[优化: optimizer.step]
```

### 3.2 关键模块解析

#### 1️⃣ 配置中心 (`MiniMindConfig`)
*   **位置**: `model/model_minimind.py`
*   **作用**: 定义模型的所有超参数。
*   **关键参数**:
    *   `hidden_size`: 词向量维度 (512/768)。
    *   `vocab_size`: 词表大小 (6400)。
    *   `use_moe`: 是否开启混合专家模式。

#### 2️⃣ 模型主体 (`MiniMindModel` & `MiniMindForCausalLM`)
*   **位置**: `model/model_minimind.py`
*   **`MiniMindForCausalLM`**: 这是最顶层的封装，包含了各类 PreTrainedModel 的接口。
    *   它包含一个 `MiniMindModel` (骨架) 和一个 `lm_head` (输出层)。
    *   **Tie Weights**: 注意 `self.model.embed_tokens.weight = self.lm_head.weight`，输入和输出共享权重矩阵，这是小模型节省参数的重要技巧。
*   **`MiniMindBlock`**: Transformer 的基本单元。
    *   包含 `Attention` (带 RoPE) 和 `FeedForward` (SwiGLU)。
    *   **RoPE (Rotary Embedding)**: 在 `Attention` 类中通过 `apply_rotary_pos_emb` 实现，通过旋转向量来注入相对位置信息。

#### 3️⃣ 数据管道 (`lm_dataset.py`)
*   **位置**: `dataset/lm_dataset.py`
*   **作用**: 将原始文本 (`jsonl`) 转换为模型可吃的 Tensor。
*   **`__getitem__`**: 核心函数。
    *   **Pretrain**: 截断长文，全部预测。
    *   **SFT**: 解析对话，生成 `loss_mask`，屏蔽提问部分的 Loss。

#### 4️⃣ 工具函数 (`trainer_utils.py`)
*   **位置**: `trainer/trainer_utils.py`
*   **`init_distributed_mode`**: 能够自动识别是单机运行还是 DDP 运行，并设置 `CUDA_VISIBLE_DEVICES` 和 `rank`。
*   **`lm_checkpoint`**: 统一的模型保存与加载逻辑，支持断点续训 (`resume`)。

## 4. 重点难点解析

### 关于混合专家 (MoE)
MiniMind 原生支持 MoE。在 `model_minimind.py` 中：
*   **`MoEGate`**: 一个线性门控网络，决定每个 token 应该去哪个专家（Expert）。
*   **`MOEFeedForward`**: 包含多个 Experts (即普通的 FeedForward)。
*   **Top-K 机制**: 训练时动态路由，只激活 K 个专家，从而在增加总参数量（知识容量）的同时，保持较低的计算量（推理速度）。

### 关于位置编码 (RoPE)
MiniMind 抛弃了传统的绝对位置编码，使用了 RoPE。
*   在 `model_minimind.py` 的 `precompute_freqs_cis` 函数中预计算了频率矩阵。
*   这使得模型具有更好的长度外推能力（训练短，推理长）。
