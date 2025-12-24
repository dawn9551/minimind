# MiniMind 全参监督微调 (SFT) 代码深度剖析

本文档紧接预训练篇，为您深度解读 **`train_full_sft.py`**。
如果说预训练是让模型“读书破万卷”，那么 SFT (Supervised Fine-Tuning) 就是让模型“下笔如有神”，学会如何得体地与人对话。

> **核心差异提前看**：
> SFT 的代码逻辑与 Pretrain 高度相似，**90% 的代码是通用的**。
> 唯二的核心区别在于：
> 1.  **初始状态**：不再随机初始化，而是继承预训练的“遗产”。
> 2.  **学习方式**：不再全文背诵，而是“只学回答，不学提问”（通过 Loss Mask 实现）。

---

## 1. 🎬 剧本入口：`if __name__ == "__main__":`

### 1.1 继承遗产 (Argument Parsing)
SFT 不是从零开始，必须站在巨人的肩膀上。

```python
# L104: --from_weight 'pretrain' 
# 这是 SFT 的灵魂参数。
# 意味着我们要加载之前训练好的 `pretrain_*.pth` 权重，而不是通过随机数生成一个新脑子。
```

### 1.2 数据集切换 (Dataset Factory)
这里发生了关键的“狸猫换太子”。

```python
# L136 (Pretrain 是 PretrainDataset) -> 这里变成了 SFTDataset
train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
# 深入堆栈 -> dataset/lm_dataset.py -> SFTDataset
#   __getitem__:
#       输入: jsonl 中的对话 [{"role": "user", "text": "你好"}, {"role": "assitant", "text": "我是MiniMind"}]
#       处理: 拼接成 "[BOS]用户:你好[EOS][BOS]助手:我是MiniMind[EOS]" 的格式。
#       生成 Loss Mask:
#           "用户:你好" 对应的 mask = 0 (模型不需要预测用户会问什么，这部分不计分)
#           "助手:我是MiniMind" 对应的 mask = 1 (模型必须学会怎么回答，这部分重点计分)
```

---

## 2. 🔄 训练循环：`train_epoch` 函数

让我们再次进入显卡发热的源头（第 23 行）。这里的代码虽然看着眼熟，但暗藏玄机。

### 2.1 不一样的 Loss 计算
这是 SFT 最迷人、也是最核心的代码块。

```python
# L36-40: 正常计算所有 token 的 Loss
loss = loss_fct(...)

# L41: 🎭 戴上面具 (Applying the Mask)
loss = (loss * loss_mask).sum() / loss_mask.sum()
```

**逐行解读**：
1.  **原始 Loss**：模型不仅计算了“回答”的错，也计算了“提问”的错。
2.  **`loss * loss_mask`**：
    *   提问部分的 loss 乘以 0 = 0。
    *   回答部分的 loss 乘以 1 = 原样保留。
3.  **结果**：模型如果在“回答”上犯错，会被狠狠惩罚；但如果在“重复问题”上犯错，毫发无损。这迫使模型集中精力学习如何**输出高质量的回复**。

### 2.2 同样的优化步伐
除了 Loss 的计算方式不同，梯度的反向传播（Backward）和参数更新（Optimizer Step）与预训练**完全一致**。
因为无论是学知识还是学对话，本质都是在调整神经元之间的连接权重。

---

## 3. 🧠 进阶思考：为什么这里要用 "Full" SFT？

脚本名叫 `train_full_sft.py`，这里的 **Full** 是什么意思？

*   **Full (全参数)**：意味着我们更新了模型**所有**的参数（26M 或 104M 个）。
*   **对比 LoRA**：`train_lora.py` 只更新极少量的附加参数（可能只有 1M 个），冻结住原来的大脑。

**对于 MiniMind 这样的小模型，强烈推荐 Full SFT**。
因为小本来就脑容量有限，如果再冻结大部分参数，它可能很难学会复杂的对话逻辑。只有全脑动员，才能发挥最大潜力。

---

## 总结

阅读 `train_full_sft.py` 时，请死死盯着 **Mask** 机制。

*   **Pretrain** 是“填鸭式教育”，给什么背什么。
*   **SFT** 是“启发式教育”，老师（Dataset）指着问题说：“这个不用背（Mask=0），但后面这个答案你要背下来（Mask=1）”。

理解了这一点，你就理解了为什么 ChatGPT 能像人一样对话，而不是像复读机一样续写新闻了。
