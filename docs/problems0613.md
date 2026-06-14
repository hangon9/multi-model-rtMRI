---

## 📋 完整 Pipeline 问题诊断报告

### **1️⃣ train.py — 训练脚本（问题最集中）**

| # | 问题类型 | 具体问题 | 后果 |
|---|---------|---------|------|
| 🔴 | **A. 顶层级代码在 import 时执行** | L14-16: `with open(CONFIG_PATH) as f: config = yaml.safe_load(f)` 写在文件作用域，不是在 `main()` 内。其他模块 import train.py 时会立刻执行。 | 潜在副作用，且该 `config` 变量与 `main()` 里重加载的 `config` 冲突 |
| 🔴 | **B. `batch["label"]` vs `batch["labels"]` 键名不匹配** | `train_one_epoch` L177: `label = batch["label"].to(device)`，但 `USCAnnot16Dataset.__getitem__` 返回的是 `batch["labels"]`（复数，见 USCAnnot16Loader.py L248）。 | 💥 **运行时 KeyError，数据链路直接断裂** |
| 🔴 | **C. 标签格式不匹配** | `train_one_epoch` 期望 `label` 是 `(B,)` 的 long tensor（单任务 CE 输入），但 `USCAnnot16Dataset` 返回 `labels` 是 `(B, 18)` 的 float tensor（全部 18 列多标签）。 | 💥 **CrossEntropyLoss 无法接收 18 维 float 输入，类型和维度都错** |
| 🔴 | **D. `make_train_test_split` 用 dict 当对象用** | L106-110: `config.group_col`、`config.test_size`、`config.random_seed` — 但 `config` 是个 dict，不是 object。 | 💥 **AttributeError 崩溃** |
| 🔴 | **E. `get_fold_indices` 同上** | L122: `config.group_col`、`config.n_splits` — 同样 dict 当对象用 | 💥 **AttributeError 崩溃** |
| 🟠 | **F. `create_dataloader` 参数混乱** | ① 参数 `batch_size=16` 被 L148 的 `batch_size = cfg_train.get(...)` 覆盖；② 同时写了 `train=True` 硬编码（L157）和 `train=train` 关键字参数（L163），重复；③ 不传 `shuffle` 到 `DataLoader`，永远 `shuffle=False` | 验证集不应 shuffle，但训练集也不 shuffle |
| 🟠 | **G. `main()` 中 fold 循环无效** | `for fold, (train_idx, val_idx) in enumerate(...)：` 内只创建了 loader，训练循环在 fold 循环外面，只用最后一个 fold。 | Cross-validation 没实现 |
| 🟠 | **H. `test_loader` 和 `val_loader` 创建后从未使用** | L196-198、L224-226 创建了 loader，但后面训练循环里用的是 `train_loader`（只有最后一个 fold 的） | 无验证/测试步骤 |
| 🟠 | **I. `args.run_name` 不存在** | L192: `logger = TrainingLogger(..., run_name=args.run_name)`，但 `argparse` 没定义 `--run-name` 参数 | 💥 **AttributeError** |
| 🟠 | **J. 导入路径问题** | L12: `from data.USCAnnot16Loaderold import ...` 使用了 old 版本；但项目中同时有 USCAnnot16Loader.py（新版本）和 USCAnnot16Loaderold.py（旧版）。train.py 引用的是旧版，但旧版 `__init__` 参数签名完全不同（需要 `data_root`, `dataframe_path` 等）。 | 新旧两版 Dataset 签名不一致，参数传错 |
| ⚪ | **K. dataloader 未传 `shuffle` 参数** | 训练集应 `shuffle=True` 但代码硬编码 `shuffle=False` | 训练不收敛 |

---

### **2️⃣ baseline_config.yaml — 配置文件**

| # | 问题类型 | 具体问题 | 后果 |
|---|---------|---------|------|
| 🔴 | **A. `classification_task` 为空字符串** | L5: `classification_task:   # manner \| place \| voicing` — 值是 Python `None`（YAML 空值）。后续代码 `config["data"]["classification_task"]` 会得到 `None`。`BuildLoss.__init__` 中 `self.classification_task = classification_task or ""` 将其变为 `""`。 | 触发多任务模式，但数据/标签是单任务格式，不匹配 |
| 🟠 | **B. `phonemic_table` 路径与实际不符** | L3: `phonemic_table: utils/TIMIT_MRI_Get_Phone_Alignment/Phonemic_Table.xlsx`，但实际文件在 Phonemic_Table.xlsx | 💥 文件不存在 |
| 🟠 | **C. `use_class_weights: true` 但从未计算 class weights** | L49: 标注启用类别加权，但代码中没有计算或加载权重的逻辑。`BuildLoss` 的 `class_weights` 参数默认 `None`。 | 类别不平衡问题未处理 |
| ⚪ | **D. `paths` 字段层级 YAML 解析问题** | L53-55: `paths:` 在 YAML 中层级与其他顶级字段并列，但在 `main()` 中 `config['paths']['log_dir']` 能拿到，没问题。但注意不显式配置时 `/` 可能存在。 | 可正常访问 |

---

### **3️⃣ USCAnnot16Loader.py — Dataset（新版）**

| # | 问题类型 | 具体问题 | 后果 |
|---|---------|---------|------|
| 🔴 | **A. `self.data_root` 未定义** | L66: `_resolve_path` 方法用到 `self.data_root`，但 `__init__` 中从未定义该属性。 | 💥 `AttributeError: 'USCAnnot16Dataset' object has no attribute 'data_root'` |
| 🔴 | **B. 返回的 labels 是 18 维多标签向量** | L248: `labels = row[self.label_columns].values` → shape `(18,)` float32 tensor。但训练代码需要的是单任务 `(B,)` 的类别索引 long tensor。 | 标签格式完全不对齐 |
| 🟠 | **C. 标签包含 18 列，但论文只有三个独立头** | 18 列并非一个多标签问题，而是三个独立分类头。Dataset 没有把标签拆成 `{ "manner": (B,), "place": (B,), "voicing": (B,) }` 三个分量。 | Loss 计算的标签索引不对 |
| ⚪ | **D. `cache_audio` 在 `__init__` 中初始化但在参数中未传递** | 参数 `cache_audio=True` 但未透传给实例——实际上已用了`self.cache_audio` | 可工作 |
| ⚪ | **E. ImageNet Normalization 是否合适？** | L85-87: 使用 ImageNet 的 mean/std 归一化。MRI 图像分布与自然图像不同，可能不一定是最优选择。 | 小问题，不影响运行 |

---

### **4️⃣ USCAnnot16Loaderold.py — Dataset（旧版）**

| # | 问题类型 | 具体问题 | 后果 |
|---|---------|---------|------|
| 🔴 | **A. `__init__` 签名与 train.py 调用不匹配** | train.py L154-163 按新版 Dataset 传参，但 import 的是旧版。旧版签名是 `(data_root, dataframe_path, subjects, tasks, ...)`，新版是 `(dataframe, untrained_subjects, untrained_tasks, ...)`。 | 💥 **传参位置错乱，TypeError 或静默错误** |
| 🟠 | **B. 与新版 Dataset 功能重复** | 两个文件实现类似功能但接口不同，容易混淆。新版在功能上更完善（支持直接从 DataFrame 过滤 subjects/tasks）但未被 train.py 使用。 | 代码维护负担 |

---

### **5️⃣ contrastive_model.py — 对比模型**

| # | 问题类型 | 具体问题 | 后果 |
|---|---------|---------|------|
| 🔴 | **A. 构造参数 `classification_classification_task` 命名错误** | L16: 参数名 `classification_classification_task=""` — 单词重复。但实际传入的是 `classification_task=classification_task`（L213 train.py），会被当作 `**kwargs` 匹配。 | 💥 实际上是位置参数的 `**kwargs` 匹配不成功？看下 `model = AudioVisionContrastiveModel(num_classes=..., classification_task=classification_task)` — 但 `classification_task` 是第三个位置参数匹配不到的，因为前两个是 `visual_tokens` 和 `target_tokens`。Python 会把 `classification_task=` 作为关键字参数匹配到 `classification_classification_task` 吗？**不会！** 因为参数名不匹配。 | 💥 **TypeError: unexpected keyword argument 'classification_task'** |
| 🟠 | **B. `self.classifier` 调用方式** | L45-48: `ClassificationHead(..., num_classes=num_classes, classification_task=classification_task)` — 这里当 `classification_task=""` 时 `num_classes` 被忽略（设计如此），但当前 `classification_task` 是 `None`（从 config 读取后变 `""`），所以进入多任务分支，但连 `num_classes` 都不用传。 | 运行时不影响，但参数冗余 |
| 🟠 | **C. `forward` 中 `classification_task=None` 的默认值逻辑** | L63: `active_classification_task = self.classification_task if classification_task is None else classification_task` — 如果 `self.classification_task=""`（多任务），且调用时没传 `classification_task`，则 `active=''`。然后在 ClassificationHead 的 forward 中触发多任务分支。 | 逻辑正确但路径复杂 |

---

### **6️⃣ classifier.py — 分类头**

| # | 问题类型 | 具体问题 | 后果 |
|---|---------|---------|------|
| 🟠 | **A. `forward` 中 single-task 分支** | L148: `return self.heads[active_classification_task](x)` — 当 `self.classification_task=""`（多任务初始化）但 `active_classification_task` 是单任务名时执行这个分支。这要求多任务模式下也有对应 head。 | 逻辑正确但容易误解 |
| ⚪ | **B. `SingleClassificationHead` 只有 `LayerNorm→Dropout→Linear`** | 论文描述中是 "linear multilayer perceptron with a softmax activation function" — 当前实现没有隐层，相当于线性分类器。论文可能期望更深一点的 MLP。 | 影响性能但不影响运行 |
| ⚪ | **C. `classification_task=""` 分支缺少独立 `num_classes` 参数** | 当 `classification_task=""` 时，`num_classes` 参数被忽略，所有三个 head 硬编码为 6/11/3。 | 灵活性问题 |

---

### **7️⃣ loss_factory.py — Loss 工厂**

| # | 问题类型 | 具体问题 | 后果 |
|---|---------|---------|------|
| 🔴 | **A. 从未计算/传入 class weights** | `class_weights=None` 永远是默认值，尽管 config 中 `use_class_weights: true`。 | 类别不平衡未缓解 |
| 🟠 | **B. 多任务 loss 分支期望 `labels` 是 dict** | L34-41: 当 `active_classification_task=""` 时，要求 `labels` 是 `{"manner": tensor, "place": tensor, "voicing": tensor}` 格式。但 Dataset 返回的是 `(B,18)` 的 float tensor。 | 多任务模式不可用 |
| ⚪ | **C. 单任务 loss 分支也会尝试从 dict 中提取 labels** | L43-50: 当 `logits` 是 dict 时尝试从其中提取；当 `labels` 是 dict 时也提取。但实际单任务下两者都是 tensor。 | 逻辑冗余但无害 |

---

### **8️⃣ vit_encoder.py — ViT 编码器**

| # | 问题类型 | 具体问题 | 后果 |
|---|---------|---------|------|
| 🟠 | **A. MONAI ViT 输出形状待验证** | 128×128 / 16×16 → 64 patches + 1 CLS = 65 tokens，shape `(B, 65, 768)`。但需要实际运行确认 MONAI 的 ViT 在 `classification=False` 时的确切输出格式。 | 如果输出不一致，后续 projection 维度 65→31 会出错 |
| ⚪ | **B. 训练策略** | ViT 随机初始化（论文要求），不加载预训练权重 ✓ | 无 |

---

### **9️⃣ contrast_losses.py — 对比损失**

| # | 问题类型 | 具体问题 | 后果 |
|---|---------|---------|------|
| ⚪ | **A. `CosineContrastLoss` 使用 `CosineEmbeddingLoss`** | 正样本对 target=1，要求两个向量方向一致 ✓ | 可工作 |
| ⚪ | **B. `NTXentLoss` 可用但未连接** | `contrast_loss_name="infonce"` 在 config 中存在但未启用 | Stage 1.5 可用 |

---

## 📊 **数据链路三段检查**

我按数据流向画一个"能否跑通"的判断：

```
 ① metadata.csv 生成
    └─ `data/annot_16_prepare.py` → 生成 `DataFrame-annot-16.csv`
    └─ ✅ 功能完整，有 CLI 入口

 ② Dataset 加载
    └─ `train.py` L12: 导入旧版 `USCAnnot16Loaderold`
    └─ L154: 按新版签名传参 → ❌ TypeError 参数错位
    └─ `USCAnnot16Loader.py`（新版）:
        ├─ `_resolve_path()` 缺 `self.data_root` → ❌ AttributeError
        └─ 返回 `labels` shape (B,18) float → ❌ 与 CE Loss 不兼容

 ③ Model forward
    └─ `contrastive_model.py`:
        ├─ 参数名 `classification_classification_task` 写错 → ❌ TypeError
        └─ 后续逻辑能通

 ④ Loss
    └─ `logits` (dict/tensor) vs `labels` (18-dim) 不匹配 → ❌
```

**结论：当前代码在任何合理输入下都无法完整跑通一个 batch 的 forward+backward。**

---

## 🎯 **解决方案建议（按文件）**

### train.py
1. **删除顶层级 L14-16 的 config 加载**，全部移到 `main()`
2. `batch["label"]` → `batch["labels"]`（与 Dataset 对齐），或者修改 Dataset 返回 `label`
3. 修改标签从 18 维 one-hot 提取对应的 task 标签索引：
   ```python
   # 在 Dataset 中根据 task 参数返回正确的标签
   manner_map = {"Silence":0, "Stop":1, "Nasal":2, "Fricative":3, "Approximant":4, "Vowel":5}
   label = torch.tensor(manner_map[row["Silence":"Vowel"].idxmax()], dtype=torch.long)
   ```
4. `make_train_test_split` 和 `get_fold_indices` 改用 `config["data"]["..."]` 方式读取
5. 将 `create_dataloader` 的 `shuffle` 参数正确传递
6. 把 fold 循环内的训练提取出来
7. 添加验证/测试步骤
8. 导入新版 `USCAnnot16Dataset`（从 `data.USCAnnot16Loader` 而非 `...Loaderold`）
9. 给 `argparse` 加 `--run-name`

### baseline_config.yaml
1. 给 `classification_task` 设置默认值（如 `manner`）
2. 修正 `phonemic_table` 路径
3. 添加 `class_weights: []` 字段或自动计算逻辑

### USCAnnot16Loader.py
1. 补上 `self.data_root = ...`（当前缺失）
2. `__getitem__` 根据传入的 `task` 参数返回对应头的类别索引，而不是 18 维向量

### contrastive_model.py
1. 修复 `classification_classification_task` → `classification_task`

### loss_factory.py
1. 添加 class weight 计算（基于训练集标签分布）

---

## 🚦 **修复优先级**

| 优先级 | 修复 | 阻塞 |
|-------|------|------|
| 🔴 P0 | train.py 键名 `label→labels` + 标签格式 18-dim → 单任务索引 | 完全阻塞 |
| 🔴 P0 | contrastive_model.py 参数名 `classification_classification_task` | 完全阻塞 |
| 🔴 P0 | `make_train_test_split/get_fold_indices` 用 dict 而非 object | 完全阻塞 |
| 🟠 P1 | `USCAnnot16Loader._resolve_path` 缺 `self.data_root` | 半阻塞 |
| 🟠 P1 | 导入 `USCAnnot16Loader` 而非 `...Loaderold` | 半阻塞 |
| 🟠 P1 | `classification_task` 从 config 读取改为 `manner` | 半阻塞 |
| 🟡 P2 | fold 循环真正实现 | 逻辑问题 |
| 🟡 P2 | 添加 val/test 评估 | 功能缺失 |
| ⚪ P3 | class weights 计算 | 优化项 |

需不需要我帮你针对这些问题的任意一个，给出具体的修复代码？