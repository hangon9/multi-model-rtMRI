# 项目指导手册：基于 rtMRI 的多模态音素分类

> **写给 AI 协作者：** 本文档是本项目的核心参考资料。在参与任何编码任务前，请完整阅读本手册。本项目具有严格的三阶段路线图，请勿在未明确指示的情况下跳跃阶段或引入未讨论的架构改动。

---

## 1. 项目背景与研究目标

本项目是一个**毕业设计（毕设）**，研究方向为：利用实时磁共振成像（rtMRI）视频与同步音频，进行**帧级音系学特征（Phonological Feature）分类**。

核心研究问题：能否通过对比学习，将音频的声学信息"蒸馏"进视觉编码器，使得推理阶段仅凭 MRI 图像即可准确预测音素的发音特征？

**参考文献**：*Audio–Vision Contrastive Phonology*（Daqi Liu et al., 2024–2025），本项目将其作为基线进行复现并在此之上展开自主研究。

---

## 2. 数据集说明

### 2.1 数据集概览

| 数据集 | 用途 | 备注 |
|---|---|---|
| **USC-annot-16** 主数据集（in-domain），**基线阶段优先跑通** | 80% 训练 / 10% 验证 / 10% 测试 |
| **TIMIT 数据集** | 跨域泛化测试（cross-domain） | 以 USC-annot-16 训练，在 TIMIT 上测试；**训练阶段严禁接触** |

### 2.2 Metadata 格式（CSV）

每一行对应一个视频帧，CSV 列定义如下：

```
subject, task, frame_idx, image_path, audio_path,
Silence, Stop, Nasal, Fricative, Approximant, Vowel,
Labial, Dental, Alveolar, Postalveolar, Palatal, Velar, Glottal,
Front, Central, Back,
Voiced, Voiceless
```

**示例行：**
```csv
sub009,bvt,0,data\USC-annot-16\sub009\images\bvt\frame_0000.png,data\USC-annot-16\sub009\audios\bvt\sub009_bvt_audio.wav,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
```

### 2.3 标签结构（关键：三个独立分类头）

> ⚠️ **这 18 列并非跨全部列的独热编码（one-hot）！** 它们是分属三个独立分类头的标签，每个头内部才构成独热编码。

三个分类头对应的列如下：

| 分类头 | 类别数 | 对应 CSV 列 |
|---|---|---|
| **Manner（发音方式）** | 6 | `Silence, Stop, Nasal, Fricative, Approximant, Vowel` |
| **Place（发音部位）** | 11 | `Silence, Labial, Dental, Alveolar, Postalveolar, Palatal, Velar, Glottal, Front, Central, Back` |
| **Voicing（清浊音）** | 3 | `Silence, Voiced, Voiceless` |

> **说明：** `Front, Central, Back` 三列属于元音发音部位（vowel place）标注，纳入 Place 分类头，使该头共 11 个类别。

**类别不平衡问题**：数据中 Silence 约占 29%，部分小类别仅占 0.几%。原文使用了类别加权（class-balanced learnable weighting scheme）应对此问题，复现时须保留。

**互斥性优化**：三个分类头的预测结果之间存在先验互斥关系（例如，当 Silence=1 时，其他两个头也应预测 Silence），后续可利用此约束过滤后处理结果。

---

## 3. 基线模型架构（Baseline — 当前实现目标）

> **任何阶段一（基线）的代码，都必须严格还原此 pipeline，不得提前引入 Stage 2/3 的改动。**

### 3.1 总体 Pipeline

```
训练阶段：
  MRI 图像帧 ──────► ViT Encoder ──► (B, 65, 768) ──► 线性投影 ──► (B, 31, 768)
                                                                         │
                                                                    Flatten
                                                                         │
                                                                    (B, 23808)
                                                                    /         \
                                              Classification Heads         Cosine Loss ◄── 来自音频
                                                    │
                                              Cross-Entropy Loss ◄── Ground Truth

  音频片段 ──────► Wav2Vec2 Encoder ──► (B, 31, 768) ──► Flatten ──► (B, 23808)

推理阶段：
  仅使用 MRI 图像帧，不使用音频。
```

**联合训练损失函数：**

```
L = L_cls + λ · L_cos     (λ = 0.1)
```

### 3.2 视觉编码器（Image Encoder）—— ViT

| 参数 | 值 |
|---|---|
| 实现库 | MONAI（`monai.networks.nets.ViT`） |
| 输入分辨率 | 128×128（灰度图，复制为 3 通道） |
| Patch 大小 | 16×16 → 196 个 patch token + 1 个 [CLS] token |
| 特征维度 | 768 |
| Transformer 层数 | 12 |
| 注意力头数 | 12 |
| FFN 维度 | 3072 |
| Dropout | 0.1 |
| 权重初始化 | **随机初始化，端到端训练**（非预训练权重） |
| 对比学习输出 | 使用所有 patch 级 token 输出（非仅 CLS），即 shape `(B, 197, 768)` → 投影到 `(B, 31, 768)` |
| 分类任务输出 | 仅使用 [CLS] token |

**MRI 预处理：**
- 重采样至 **15 fps**
- 每帧 resize 至 128×128，灰度图复制为 3 通道
- 每帧对应约 **66.67 ms** 的语音（以该帧时间戳为中心截取时间窗口）

### 3.3 语音编码器（Speech Encoder）—— Wav2Vec2

| 参数 | 值 |
|---|---|
| 模型 | `facebook/wav2vec2-base-960h`（HuggingFace） |
| 训练策略 | **冻结参数（frozen）**，不参与反向传播 |
| 输入 | 16kHz，pad 到固定长度 |
| 输出 | `(B, 31, 768)` 的 frame-level 声学特征 |

### 3.4 投影与对比学习

- ViT 输出的 patch token 序列通过一个**可学习的线性层**，将时间维度从 197 投影至 31（与音频帧数对齐）
- 随后两个模态的特征都通过**独立的 MLP** 投影，得到 shape `[B, T, D]`（T=31, D=768）
- Flatten 为 `[B, T·D]`，即 `(B, 23808)` 的向量
- 对同一时间步的图像-音频对施加 **Cosine Embedding Loss**（正样本鼓励高相似度）

### 3.5 分类头

- 三个**独立**的分类头（manner / place / voicing），各自为一个线性层
- 输入：Flatten 后的视觉特征 `(B, 23808)`
- 损失：**Class-balanced Cross-Entropy**（类别加权，缓解不平衡问题）
- 多任务训练时同时训练三个分类头，单任务时仅训练对应头，其他头不更新

### 3.6 训练超参数

| 参数 | 值 |
|---|---|
| Epochs | 30 |
| Batch Size | 32 |
| 学习率 | 1e-4 |
| Weight Decay | 5e-4 |
| Optimizer | AdamW |
| GPU（参考） | NVIDIA RTX A100 40GB |
| 对比损失权重 λ | 0.1 |

### 3.7 评估指标

- 帧级（frame-level）Precision、Recall、**Macro-averaged F1-score**
- 三个分类头分别评估，并记录每个类别的细粒度指标

---

## 4. 数据集划分方案

### 4.1 In-domain（USC-annot-16 / Horation 16）

除了正常划分测试集，使用交叉验证以外，训练集和验证集也要支持以下三种泛化划分：

| 方案 | 说明 | 泛化难度 |
|---|---|---|
| **Seen Speaker, Unseen Task** | 训练见过目标说话人，未见过该说话人的对应任务 | 低 |
| **Unseen Speaker, Seen Task** | 训练未见过目标说话人，但见过目标任务 | 中 |
| **Unseen Speaker, Unseen Task** | 训练既未见说话人，也未见对应任务 | 高（最严格） |

在代码中体现为不训练某个subject或task

### 4.2 Cross-domain（TIMIT 数据集）

- 训练集：USC-annot-16（与 in-domain 相同）
- 测试集：TIMIT 数据集
- **TIMIT 训练阶段严禁接触**，仅在最终评估时用作跨域泛化测试集，用于衡量模型在不同采集设备/说话人分布下的鲁棒性。

---

## 5. 三阶段研究路线图

### Stage 1 — 基线复现（当前任务）⭐

**目标：** 严格复现论文中的 contrastive learning 方案，跑通完整训练/评估流程。  
**数据集：** 首先在 **USC-annot-16** 上跑通，随后支持以下所有评估划分：

**基线需支持的五种评估配置（缺一不可）：**

| 配置 | 训练集 | 测试集 | 说明 |
|---|---|---|---|
| In-domain ①  | USC-annot-16（Seen Speaker 部分） | Unseen Task | 见过说话人，未见任务 |
| In-domain ②  | USC-annot-16（Seen Task 部分） | Unseen Speaker | 见过任务，未见说话人 |
| In-domain ③  | USC-annot-16（排除目标组合） | Unseen Speaker + Unseen Task | 最严格泛化，对两者都未见 |
| Cross-domain | USC-annot-16（全量） | TIMIT 数据集 | 跨数据集泛化，TIMIT 训练不可见 |

> 三种 in-domain 划分方案来自导师此前的分割文章脚本，复现时参考使用。  
> 每次运行需在配置中明确指定当前使用哪种划分，以确保结果可复现。

**完成标准：** 三个分类头均可正常收敛，指标与论文数量级一致（参见第 6 节参考指标）。  
**截止时间：** 2025年5月25日起 1~2 周内。  
**下一步：** 跑通后安排会议汇报结果，导师确认后进入 Stage 1.5。

---

### Stage 1.5 — 对比损失替换：Cosine Loss → InfoNCE

**目标：** 在基线架构不变的前提下，将对比学习损失从 **Cosine Embedding Loss** 替换为 **InfoNCE Loss**，验证其对分类性能的影响。

**InfoNCE 核心逻辑：**
- **正样本对**：同一帧的 MRI 图像与其对应音频片段 → 特征空间距离尽可能近
- **负样本对**：当前帧图像与 batch 内其他帧的音频 → 特征空间距离尽可能远
- InfoNCE 通过 softmax 对 batch 内所有负样本归一化，比 Cosine Loss 提供更强的对比信号

**实施要求：**
- 保持 Stage 1 所有其他组件不变（ViT、Wav2Vec2、分类头、CE Loss）
- 仅替换 `losses.py` 中的对比损失实现
- 在所有五种评估配置下对比 Cosine Loss 与 InfoNCE 的结果，记录差异
- 本阶段结果作为 Stage 2 多帧改进的新基线参照

**损失函数更新为：**
```
L = L_cls + λ · L_InfoNCE     (λ 待调，初始仍为 0.1)
```

### Stage 2 — 多帧输入改进

**目标：** 将输入从单帧图像改为多帧序列，提升时序建模能力。  
**前置条件：** Stage 1.5（InfoNCE）已完成并有对比结果。  
**可选视频编码器：** ViViT 或 VideoMAE  
**特征融合方式：** Cross-Attention 或 对比学习  
**核心要求：** 效果须超过 Stage 1/1.5 基线，作为毕设核心方法。  
**注意：** 具体架构设计由学生自主决定，可参考现有文献。

### Stage 3 — 添加分割掩码模态

**目标：** 引入发音器官的语义分割掩码作为额外输入。  
**分割内容：** 舌头、上嘴唇、下嘴唇、上颚（由导师提供）  
**研究假设：** 显式的发音器官位置信息可直接提升音素分类效果。

### 可选扩展 — 缺失模态鲁棒性（待后续讨论）

- 训练时随机丢弃音频（audio dropout）
- 教师-学生知识蒸馏（Knowledge Distillation）  
> 该方向仅在基线稳定后讨论，当前不作为优先项。

---

## 6. 参考性能指标（来自原文）

以下为论文报告的 **Contrastive 模型**（本项目要复现的基线）的结果，供验收参考：

**Manner of Articulation（发音方式）：**
| 类别 | Precision | Recall | F1 |
|---|---|---|---|
| Silence | 0.91 | 0.87 | 0.89 |
| Stop | 0.85 | 0.88 | 0.86 |
| Nasal | 0.85 | 0.80 | 0.82 |
| Fricative | 0.80 | 0.85 | 0.82 |
| Approximant | 0.65 | 0.70 | 0.67 |
| Vowel | 0.85 | 0.79 | 0.82 |
| **AVG** | **0.82** | **0.81** | **0.81** |

**Place of Articulation（发音部位）：**
| 类别 | Precision | Recall | F1 |
|---|---|---|---|
| Silence | 0.95 | 0.90 | 0.92 |
| Labial | 0.92 | 0.90 | 0.91 |
| ... | ... | ... | ... |
| **AVG** | **0.80** | **0.76** | **0.78** |

**Voicing（清浊音）：**
| 类别 | Precision | Recall | F1 |
|---|---|---|---|
| Silence | 0.95 | 0.91 | 0.93 |
| Voiceless | 0.85 | 0.80 | 0.82 |
| Voiced | 0.90 | 0.88 | 0.89 |
| **AVG** | **0.90** | **0.86** | **0.88** |

---

## 7. 代码规范与注意事项

### 7.1 必须遵守

- [ ] 推理阶段**只使用 ViT 视觉分支**，音频分支仅在训练时启用
- [ ] Wav2Vec2 参数在 Stage 1 全程**冻结**
- [ ] 三个分类头**独立**，不共享参数
- [ ] 损失函数 = `L_cls（类别加权）+ 0.1 × L_cos`
- [ ] 评估指标统一使用**Macro F1**，分头报告
- [ ] Cross-domain 数据集**绝对不能出现在训练集/验证集**
- [ ] 音频窗口需以目标帧时间戳为**中心**截取

### 7.2 常见陷阱

- **ViT 输入**：MRI 是灰度图，需复制为 3 通道后输入，而非直接单通道输入
- **标签读取**：读取 CSV 时必须按分类头分组提取标签列，不要将全部 18 列拼成一个 one-hot 向量
- **Place 头共 11 列**：`Silence, Labial, Dental, Alveolar, Postalveolar, Palatal, Velar, Glottal, Front, Central, Back`，`Front/Central/Back` 是元音发音部位，**必须纳入 Place 头，不可忽略**
- **时间对齐**：音频窗口是以帧时间戳为中心的固定长度窗口，不是从 0 到当前时刻
- **TIMIT 数据集隔离**：任何训练/验证代码路径中不得出现 TIMIT 数据；仅 `evaluate.py` 的跨域测试模式可加载

### 7.3 推荐项目结构

```
project/
├── data/
│   └── USC-annot-16/           # 原始数据（不修改）
├── dataset.py                  # Dataset 类，负责 CSV 读取与三头标签提取
├── models/
│   ├── vit_encoder.py          # MONAI ViT 封装
│   ├── wav2vec_encoder.py      # Wav2Vec2 封装（冻结）
│   └── baseline.py             # 完整基线模型（对比学习 + 三分类头）
├── losses.py                   # Class-balanced CE + Cosine Embedding Loss
├── train.py                    # 训练主脚本
├── evaluate.py                 # 验证/测试脚本，输出 per-class F1
├── configs/
│   └── baseline.yaml           # 超参数配置
└── AGENT_GUIDE.md              # 本文档
```

---

## 8. 快速参考

| 问题 | 答案 |
|---|---|
| 推理时用哪个模态？ | 仅 MRI 图像（ViT） |
| Wav2Vec2 要微调吗？ | Stage 1/1.5 不微调，权重冻结 |
| 标签是多标签分类吗？ | 否，三个头各自是单标签分类（每帧属于每个头的一个类） |
| Place 头有多少类？ | 11 类：Silence + 7个辅音部位 + Front/Central/Back（元音部位） |
| Silence 会同时出现在三个头吗？ | 是，Silence 帧在所有三个头的标签均为 Silence |
| 对比学习正样本对怎么定义？ | 同一帧的 MRI 图像与其对应音频片段 |
| 对比学习用的是哪种 loss？ | Stage 1：Cosine Embedding Loss；Stage 1.5：替换为 InfoNCE Loss |
| 批次内负样本对？ | 当前帧图像与 batch 内**其他帧**的音频构成负样本对 |
| in-domain 有几种划分？ | 3 种（Seen Spk / Unseen Task；Unseen Spk / Seen Task；Unseen Spk+Task） |
| cross-domain 怎么设置？ | USC-annot-16 训练，TIMIT 仅用于测试，训练代码不可读取 TIMIT |

---

*最后更新：2025年6月12日（v2：修正数据集名称、Place头扩至11类、路线图补充评估划分与InfoNCE阶段）*
