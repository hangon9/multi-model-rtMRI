"""img_mask_oracle 流水线的本地验证（假数据、CPU、秒级）。

覆盖计划里的 T1–T4 与 T6（T5 的「过拟合冒烟」由 T4 的单步反传 + 优化器分组断言代替）：
- T1 路径派生 / 帧号重写的失败语义；
- T2 mask 与图像真的读到 i-2..i+2 这五张不同帧；
- T3 图像与 mask 共用同一组仿射参数，one-hot 互斥且只有 0/1；
- T4 baseline 权重可载入、mask 通道置零时输出与 baseline 逐元素相等、mask_conv 有梯度且
  在 ``freeze_layers>=1`` 下仍可训练；
- T6 极小 ckpt 走完整 ``run_inference``，实验名同时含 ``img`` 与 ``mask`` 时判为
  ``image_mask`` 而不是 ``image``。

用法::

    conda activate rtmri
    python _tmp_verify/test_img_mask.py
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.USCAnnot16Loader import USCAnnot16Dataset  # noqa: E402
from src.eval.inference import (  # noqa: E402
    MODEL_FAMILY_ALIASES,
    _resolve_model_family,
    run_inference,
)
from src.models.img_mask_model import (  # noqa: E402
    ImageMaskMultiheadClassifier,
    ResNet50MaskEncoder,
)
from src.models.img_only_model import ImageMultiheadClassifier  # noqa: E402
from train_img_mask import build_optimizer, resolve_mask_model_params  # noqa: E402

#: 假图像边长；小到 ResNet50 也能在 CPU 上秒级跑完
FAKE_IMG_SIZE = 32
#: 假数据帧数（每句），足够让 window_frames=5 的窗口有内部帧
FAKE_FRAMES = 10
#: 假数据的两名被试与两个任务，用于构造出非空的 unseen_* 三个测试集
SUBJECTS = ("sub001", "sub002")
TASKS = ("bvt", "topic1")
#: 18 个标签列的顺序（与 `data/annot_16_prepare.py` 的 META_COLUMNS 之后保持一致）
LABEL_COLUMNS = [
    "Silence", "Stop", "Nasal", "Fricative", "Approximant", "Vowel",
    "Labial", "Dental", "Alveolar", "Postalveolar", "Palatal", "Velar",
    "Glottal", "Front", "Central", "Back", "Voiced", "Voiceless",
]


# ---------------------------------------------------------------------------
# 假数据构造
# ---------------------------------------------------------------------------

def _write_image(path: Path, gray_value: int) -> None:
    """写一张整幅灰度恒为 ``gray_value`` 的假图像。

    Args:
        path: 目标路径（父目录需已存在）。
        gray_value: 0–255 的灰度值；测试用它编码「这是第几帧」。

    Returns:
        None。副作用：写 PNG 文件。
    """
    Image.fromarray(
        np.full((FAKE_IMG_SIZE, FAKE_IMG_SIZE), gray_value, dtype=np.uint8)
    ).save(path)


def _write_mask(path: Path, organ_value: int) -> None:
    """写一张左上角 4×4 方块为 ``organ_value``、其余为背景 0 的假 mask。

    Args:
        path: 目标路径（父目录需已存在）。
        organ_value: 方块内的灰度值，即 ``data.mask_values`` 里的某个器官值。

    Returns:
        None。副作用：写 PNG 文件。
    """
    arr = np.zeros((FAKE_IMG_SIZE, FAKE_IMG_SIZE), dtype=np.uint8)
    arr[0:4, 0:4] = organ_value
    Image.fromarray(arr).save(path)


def _build_fake_annot16(root: Path) -> Path:
    """按 annot-16 的目录形状生成假数据 + DataFrame CSV。

    Args:
        root: 假数据集根目录（会被创建）。

    Returns:
        生成的 CSV 路径。

    Note:
        每帧图像灰度值 = ``frame_idx + 1``（用来识别窗口里读到的是哪几帧），mask 方块灰度值
        = ``frame_idx % 4 + 1``（用来识别 mask 读到的是哪几帧，且覆盖全部 4 个器官值）。
    """
    rows = []
    for subject in SUBJECTS:
        for task in TASKS:
            image_dir = root / subject / "images" / task
            mask_dir = root / subject / "masks" / task
            image_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            for frame in range(FAKE_FRAMES):
                stem = f"{subject}_{task}_{frame:04d}"
                _write_image(image_dir / f"{stem}_image.png", frame + 1)
                _write_mask(mask_dir / f"{stem}_mask.png", frame % 4 + 1)
                row = {
                    "subject": subject,
                    "task": task,
                    "frame_idx": frame,
                    "image_path": str(image_dir / f"{stem}_image.png"),
                    # mask 模式下数据集不读音频，给一个不存在的路径即可
                    "audio_path": str(root / subject / "audios" / f"{subject}_{task}.wav"),
                }
                row.update({col: 0 for col in LABEL_COLUMNS})
                row["Silence"] = 1   # 全帧标 Silence：只需标签列存在，指标不是本测试的重点
                rows.append(row)

    csv_path = root / "DataFrame-annot-16.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _make_config(root: Path, **overrides) -> dict:
    """构造最小的 mask 流水线配置（CPU、无预训练权重、32×32 输入）。

    Args:
        root: 假数据集根目录。
        **overrides: 按顶层键覆盖（例如 ``data_augment`` 的字典）。

    Returns:
        可直接交给 ``create_dataloader`` / ``run_inference`` 的配置字典。
    """
    config = {
        "experiment_name": "img_mask_oracle_ResNet50",
        "data": {
            "dataset": "USC-annot-16",
            "root": str(root),
            "classification_task": "",
            "unseen_speakers": ["sub002"],
            "unseen_task_types": ["topic"],
            "n_splits": 2,
            "train_fold": [],
            "fps": 15,
            "audio_sample_rate": 16000,
            "window_frames": 1,
            "seed": 42,
            "use_mask": True,
            "mask_values": [1, 2, 3, 4],
        },
        "model": {
            "image_encoder": {
                "model_name": "ResNet50",
                "pretrained": False,
                "img_size": FAKE_IMG_SIZE,
                "freeze_layers": 0,
                "mask_channels": 4,
            },
            "classifier": {"clf_hidden_dim": 16},
            "image_temporal": {"temporal_type": "none"},
        },
        "train": {
            "batch_size": 4,
            "num_workers": 0,
            "lr_image_encoder": 1e-4,
            "lr_image_temporal": 1e-4,
            "lr_classifier": 1e-4,
            "lr_image_mask": 1e-3,
            "weight_decay": 0.001,
            "data_augment": False,
        },
        "loss": {"use_class_weights": False, "bce_pos_weight": False},
        "paths": {"checkpoint_dir": str(root / "ckpt"), "log_dir": str(root / "logs")},
    }
    for key, value in overrides.items():
        config[key] = value
    return config


# ---------------------------------------------------------------------------
# T1：路径派生与帧号重写
# ---------------------------------------------------------------------------

def test_mask_path_from_image() -> None:
    """``_mask_path_from_image`` 在 annot-16 与 TIMIT 两种命名上都要给出预期路径。"""
    annot = USCAnnot16Dataset._mask_path_from_image(
        "data/USC-annot-16/sub009/images/bvt/sub009_bvt_0007_image.png"
    )
    assert annot.replace("\\", "/") == (
        "data/USC-annot-16/sub009/masks/bvt/sub009_bvt_0007_mask.png"
    ), f"annot-16 派生结果不对：{annot}"

    timit = USCAnnot16Dataset._mask_path_from_image(
        "data/USC-TIMIT/images/f1/usctimit_mri_f1_001_005/usctimit_mri_f1_001_005_7.png"
    )
    assert timit.replace("\\", "/") == (
        "data/USC-TIMIT/masks/f1/usctimit_mri_f1_001_005/usctimit_mri_f1_001_005_7_mask.png"
    ), f"TIMIT 派生结果不对：{timit}"

    # 路径里没有 images 目录时必须响亮失败，而不是猜一个路径
    try:
        USCAnnot16Dataset._mask_path_from_image("data/no_images_here/x_0007.png")
    except ValueError as exc:
        assert "images" in str(exc), f"错误信息应提示缺 images 目录：{exc}"
    else:
        raise AssertionError("没有 images 目录时必须抛 ValueError")
    print("[PASS] T1 mask 路径派生（annot-16 / TIMIT / 错误路径）")


def test_rewrite_frame_strictness() -> None:
    """帧号重写必须恰好匹配一次：匹配不上或匹配多处都抛错。"""
    base = USCAnnot16Dataset
    ok = base._rewrite_frame(
        "/a/sub001_bvt_0003_image.png", 7, base._IMG_RE, base._IMG_FMT
    )
    assert ok.endswith("sub001_bvt_0007_image.png"), f"重写结果不对：{ok}"

    for bad in ("/a/image.png", "/a/sub001_bvt_0003_image.jpg"):
        try:
            base._rewrite_frame(bad, 7, base._IMG_RE, base._IMG_FMT)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} 匹配不上时必须抛 ValueError")
    print("[PASS] T1 帧号重写：匹配失败/不匹配即报错")


# ---------------------------------------------------------------------------
# T2 / T3：窗口取帧与仿射一致性
# ---------------------------------------------------------------------------

def _make_dataset(config: dict, dataframe, train: bool = False) -> USCAnnot16Dataset:
    """按配置构造一个 mask 模式的数据集（不经过 DataLoader，便于直接索引）。"""
    cfg_data = config["data"]
    cfg_img = config["model"]["image_encoder"]
    return USCAnnot16Dataset(
        dataframe,
        image_size=cfg_img["img_size"],
        target_sample_rate=cfg_data["audio_sample_rate"],
        fps=cfg_data["fps"],
        window_frames=cfg_data["window_frames"],
        train=train,
        data_augment=cfg_data.get("data_augment", False) if train else False,
        use_mask=True,
        mask_values=cfg_data["mask_values"],
    )


def test_mask_window_reads_neighbours(csv_path: Path, root: Path) -> None:
    """window_frames=5 时图像与 mask 必须读到同样的 i-2..i+2 五张不同帧。"""
    import pandas as pd

    config = _make_config(root, data={**_make_config(root)["data"], "window_frames": 5})
    df = pd.read_csv(csv_path)
    dataset = _make_dataset(config, df)

    sample = dataset[0]
    assert sample["image"].shape == (5, 3 + 4, FAKE_IMG_SIZE, FAKE_IMG_SIZE), (
        f"窗口形状应为 [5,7,H,W]，实际 {tuple(sample['image'].shape)}"
    )

    # 图像通道 0 的均值 = (帧号+1)/255，反推窗口帧号
    image_frames = [
        int(round(float(sample["image"][i, 0].mean()) * 255)) - 1 for i in range(5)
    ]
    assert image_frames == [0, 1, 2, 3, 4], f"图像窗口帧号不对：{image_frames}"

    # mask 的 4 个 one-hot 通道里，哪个为 1 编码了 (帧号 % 4)
    mask_organs = [
        int(torch.argmax(sample["image"][i, 3:].sum(dim=(1, 2)))) for i in range(5)
    ]
    assert mask_organs == [f % 4 for f in image_frames], (
        f"mask 窗口帧号与图像不一致：图像={image_frames}、mask 通道={mask_organs}"
    )
    print("[PASS] T2 五帧窗口：图像与 mask 都读到 i-2..i+2 的不同帧")


def test_onehot_and_affine_consistency(root: Path) -> None:
    """one-hot 互斥且只有 0/1；图像与 mask 经同一组仿射后相对位移一致。"""
    import pandas as pd

    # 只有一句一帧的最小 DataFrame，直接指定图像/mask 的几何关系
    subject, task = SUBJECTS[0], TASKS[0]
    image_dir = root / subject / "images" / task
    mask_dir = root / subject / "masks" / task
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # 图像：中间偏左放一个白色方块；mask：同一位置标成器官 2
    img = np.zeros((FAKE_IMG_SIZE, FAKE_IMG_SIZE), dtype=np.uint8)
    img[8:14, 6:12] = 255
    msk = np.zeros((FAKE_IMG_SIZE, FAKE_IMG_SIZE), dtype=np.uint8)
    msk[8:14, 6:12] = 2
    stem = f"{subject}_{task}_0000"
    Image.fromarray(img).save(image_dir / f"{stem}_image.png")
    Image.fromarray(msk).save(mask_dir / f"{stem}_mask.png")

    config = _make_config(root)
    config["data"]["data_augment"] = {"Random_Affine": True, "random_time_shift": 0}
    df = pd.DataFrame([{
        "subject": subject,
        "task": task,
        "frame_idx": 0,
        "image_path": str(image_dir / f"{stem}_image.png"),
        # mask 模式下数据集不读音频
        "audio_path": str(root / subject / "audios" / f"{subject}_{task}.wav"),
        **{col: (1 if col == "Silence" else 0) for col in LABEL_COLUMNS},
    }])

    dataset = _make_dataset(config, df, train=True)
    # 固定一组仿射参数：只要图像与 mask 共用它，两者的方块位移就必须相同
    dataset._sample_window_affine = lambda: (0.0, [3, -2], 1.0, [0.0, 0.0])

    sample = dataset[0]
    gray = sample["image"][0]          # 灰度通道 0
    onehot = sample["image"][3:]       # 4 个器官通道

    # one-hot 性质：取值只有 0/1，每个像素最多一个器官，且背景像素四通道全 0
    assert set(torch.unique(onehot).tolist()) <= {0.0, 1.0}, (
        f"one-hot 出现了非 0/1 值：{torch.unique(onehot).tolist()}"
    )
    assert float(onehot.sum(dim=0).max()) <= 1.0, "一个像素不能同时属于两个器官"

    organ = onehot[1]   # 灰度值 2 → 类别 id 2 → 通道索引 1
    assert float(organ.sum()) > 0, "仿射后器官方块消失了，无法比较位置"

    gray_ys, gray_xs = torch.nonzero(gray > 0.5, as_tuple=True)
    organ_ys, organ_xs = torch.nonzero(organ > 0.5, as_tuple=True)
    dy = abs(float(gray_ys.float().mean()) - float(organ_ys.float().mean()))
    dx = abs(float(gray_xs.float().mean()) - float(organ_xs.float().mean()))
    assert dy <= 1.0 and dx <= 1.0, (
        f"图像与 mask 的方块位置不一致（dy={dy:.2f}, dx={dx:.2f}），"
        "说明两者没有共用同一组仿射参数"
    )
    print("[PASS] T3 one-hot 互斥 + 图像/mask 共用仿射（方块位移一致）")


# ---------------------------------------------------------------------------
# T4：模型等价性、梯度、冻结行为
# ---------------------------------------------------------------------------

def _build_baseline_and_mask_models(freeze_layers: int = 0):
    """构造同配置的 baseline 与 mask 模型（无预训练权重、32×32、单帧）。"""
    common = dict(
        num_classes=18,
        img_size=FAKE_IMG_SIZE,
        clf_hidden_dim=16,
        dropout=0.0,
        classification_task="",
        model_name="ResNet50",
        pretrained=False,
        freeze_layers=freeze_layers,
        temporal_type="none",
    )
    baseline = ImageMultiheadClassifier(**common).eval()
    mask_model = ImageMaskMultiheadClassifier(
        mask_channels=4, **common
    ).eval()
    return baseline, mask_model


def test_weight_equivalence() -> None:
    """baseline 权重 strict=False 载入后，任意 mask 下输出与 baseline 逐元素相等。"""
    baseline, mask_model = _build_baseline_and_mask_models()
    missing, unexpected = mask_model.load_state_dict(
        baseline.state_dict(), strict=False
    )
    assert set(missing) == {"image_encoder.mask_conv.weight"}, (
        f"唯一允许缺失的参数是 mask_conv，实际缺 {sorted(missing)}"
    )
    assert not unexpected, f"baseline 权重里出现了 mask 模型没有的键：{sorted(unexpected)}"

    torch.manual_seed(0)
    batch = 2
    gray = torch.rand(batch, 3, FAKE_IMG_SIZE, FAKE_IMG_SIZE)
    mask = torch.zeros(batch, 4, FAKE_IMG_SIZE, FAKE_IMG_SIZE)
    mask[:, 2, 4:9, 4:9] = 1.0   # 任意 mask：零初始化的 mask_conv 让它不产生贡献

    with torch.no_grad():
        base_out = baseline(image=gray)
        mask_out = mask_model(image=torch.cat([gray, mask], dim=1))

    # logits 字典里除四个头外还有 all_logits/classification_task 两个非张量键，只比四个头
    for task in ("manner", "place", "voicing", "vowel_backness"):
        assert torch.equal(base_out["logits"][task], mask_out["logits"][task]), (
            f"{task} 头的 logits 与 baseline 不相等：mask 通道零初始化被破坏了"
        )
    assert torch.equal(base_out["pooled_embedding"], mask_out["pooled_embedding"]), (
        "pooled_embedding 与 baseline 不相等"
    )
    print("[PASS] T4 等价性：mask 通道置零 → 输出与 baseline 逐元素相等")


def test_mask_grad_and_freezing() -> None:
    """单步反传后 mask_conv 梯度非零；freeze_layers=1 时它仍可训练、stem 被冻结。"""
    torch.manual_seed(0)
    _, model = _build_baseline_and_mask_models(freeze_layers=1)
    model.train()

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.1
    )
    before = model.image_encoder.mask_conv.weight.detach().clone()

    image = torch.rand(2, 1, 3 + 4, FAKE_IMG_SIZE, FAKE_IMG_SIZE)
    labels = {"manner": torch.tensor([1, 2])}
    logits = model(image=image)["logits"]
    loss = torch.nn.functional.cross_entropy(logits["manner"], labels["manner"])
    loss.backward()

    grad = model.image_encoder.mask_conv.weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0, "mask_conv 没有梯度"

    optimizer.step()
    after = model.image_encoder.mask_conv.weight.detach()
    assert not torch.equal(before, after), "一步更新后 mask_conv 权重没有变化"

    # freeze_layers=1 冻结的是 stem；mask_conv 在 super().__init__() 之后创建，必须仍可训
    assert model.image_encoder.resnet.conv1.weight.requires_grad is False, (
        "freeze_layers=1 应冻结 conv1"
    )
    assert model.image_encoder.mask_conv.weight.requires_grad is True, (
        "mask_conv 必须保持可训练，否则 mask 输入永远学不到东西"
    )

    # freeze_layers=-1 与 mask 输入冲突，必须早失败
    try:
        ResNet50MaskEncoder(pretrained=False, freeze_layers=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("freeze_layers=-1 时必须抛 ValueError")
    print("[PASS] T4 mask_conv 有梯度、可更新，freeze_layers=1 下仍可训练")


def test_optimizer_and_params(root: Path) -> None:
    """优化器要单独给 mask_conv 一个学习率组，模型实参要带上 mask_channels。"""
    config = _make_config(root)
    params = resolve_mask_model_params(config, "")
    assert params["mask_channels"] == 4, f"mask_channels 未透传：{params['mask_channels']}"

    model = ImageMaskMultiheadClassifier(**params)
    optimizer = build_optimizer(model, config)
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert "image_mask_decay" in groups, f"缺少 image_mask 组：{sorted(groups)}"

    mask_params = {
        id(p) for p in model.image_encoder.mask_conv.parameters()
    }
    grouped = {id(p) for p in groups["image_mask_decay"]["params"]}
    assert mask_params == grouped, "mask_conv 参数没有落进 image_mask 组"
    assert groups["image_mask_decay"]["lr"] == 1e-3, (
        f"image_mask 组学习率应取 train.lr_image_mask，实际 {groups['image_mask_decay']['lr']}"
    )
    print("[PASS] T4 优化器分组：mask_conv 独立成组、学习率取自 lr_image_mask")


# ---------------------------------------------------------------------------
# TIMIT：mask 覆盖审计（本地只有部分被试带 mask）
# ---------------------------------------------------------------------------

def _write_timit_sentence(
    root: Path, subject: str, stem: str, n_frames: int, with_mask: bool
) -> None:
    """写一句假 USC-TIMIT 数据（labels + images + audios，可选 masks）。

    Args:
        root: 假数据集根目录（不存在时自动创建）。
        subject: 被试目录名（如 ``f1``），同时作为 ``masks/<subject>/`` 的层级。
        stem: 句子 stem，决定图像文件名 ``{stem}_{i}.png``。
        n_frames: 帧数。
        with_mask: 是否写 ``masks/<subject>/<stem>/<stem>_<i>_mask.png``。

    Returns:
        None。副作用：建目录 + 写标签 CSV / PNG / 静音 wav。

    Note:
        mask 层级必须与真实数据一致（``masks/<subject>/<stem>/``）：基类的
        ``_mask_path_from_image`` 只把最靠右的 ``images`` 换成 ``masks``，不增减层级。
    """
    import wave

    labels_dir = root / "labels" / subject
    image_dir = root / "images" / subject / stem
    audio_dir = root / "audios" / subject
    mask_dir = root / "masks" / subject / stem
    for path in (labels_dir, image_dir, audio_dir):
        path.mkdir(parents=True, exist_ok=True)
    if with_mask:
        mask_dir.mkdir(parents=True, exist_ok=True)

    # 标签：每帧都标 Silence，只需保证列存在（本测试只关心 mask 覆盖检查）
    with (labels_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", *LABEL_COLUMNS])
        for frame in range(n_frames):
            writer.writerow(
                [f"{stem}_{frame}.png", *[1 if c == "Silence" else 0 for c in LABEL_COLUMNS]]
            )

    for frame in range(n_frames):
        _write_image(image_dir / f"{stem}_{frame}.png", frame + 1)
        if with_mask:
            _write_mask(mask_dir / f"{stem}_{frame}_mask.png", 1)

    with wave.open(str(audio_dir / f"{stem}.wav"), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b"\x00\x00" * int(round(n_frames * 16000 / 15)))


def test_timit_mask_audit(root: Path) -> None:
    """缺 mask 的被试必须在构造期报错；只取有 mask 的被试时能正常读样本。"""
    from data.timit_prepare import build_timit_dataframe
    from data.USCTIMITLoader import USCTIMITDataset

    timit_root = root / "USC-TIMIT"
    _write_timit_sentence(timit_root, "f1", "usctimit_mri_f1_001_005", 6, with_mask=True)
    _write_timit_sentence(timit_root, "f2", "usctimit_mri_f2_001_005", 6, with_mask=False)
    df = build_timit_dataframe(timit_root, verbose=False)

    def _make(dataframe, use_mask: bool):
        return USCTIMITDataset(
            dataframe,
            image_size=FAKE_IMG_SIZE,
            window_frames=1,
            audio_cache_size=1,
            use_mask=use_mask,
            mask_values=[1, 2, 3, 4],
        )

    # 只取有 mask 的被试：应该正常，且样本形状为 [3+4, H, W]
    masked_df = df[df["subject"] == "f1"].reset_index(drop=True)
    dataset = _make(masked_df, use_mask=True)
    sample = dataset[0]
    assert sample["image"].shape == (3 + 4, FAKE_IMG_SIZE, FAKE_IMG_SIZE), (
        f"T=1 的 mask 样本形状应为 [7,H,W]，实际 {tuple(sample['image'].shape)}"
    )

    # 带上没 mask 的 f2：必须报错，且消息里能看出是哪个被试缺
    try:
        _make(df, use_mask=True)
    except ValueError as exc:
        assert "f2" in str(exc), f"错误信息应指出缺 mask 的被试：{exc}"
    else:
        raise AssertionError("缺 mask 时必须抛 ValueError")

    # 关闭 mask 后同一份 df 应能正常构造（旧行为不受影响）
    plain = _make(df, use_mask=False)
    assert len(plain) == len(df), "use_mask=False 时不应过滤任何行"
    print("[PASS] TIMIT mask 覆盖审计：缺 mask 报错 / 只取有 mask 的被试可读")


# ---------------------------------------------------------------------------
# T6：模型族判定与端到端推理
# ---------------------------------------------------------------------------

def test_family_resolution(root: Path) -> None:
    """实验名同时含 img 与 mask 时必须判成 image_mask，而不是 image。"""
    config = _make_config(root)
    family = _resolve_model_family(config, Path("checkpoints/best_model.pt"))
    assert family == "image_mask", f"应判为 image_mask，实际 {family}"

    # 别名表要暴露给 evaluate 的 --model choices
    assert MODEL_FAMILY_ALIASES.get("image_mask") == "image_mask", (
        "MODEL_FAMILY_ALIASES 缺少 image_mask"
    )
    print("[PASS] T6 模型族判定：含 mask 的实验名不被误判成 image")


def test_run_inference_smoke(csv_path: Path, root: Path) -> None:
    """极小 ckpt 走完整 run_inference：mask 族可用，且 mask_mode 生效。"""
    import pandas as pd

    config = _make_config(root)
    params = resolve_mask_model_params(config, "")
    model = ImageMaskMultiheadClassifier(**params)
    ckpt_path = root / "best_model.pt"
    torch.save(
        {
            "fold": 1,
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "best_val_loss": 1.0,
            "select_metric": "f1_mean",
            "best_select_value": 0.0,
            "config": config,
        },
        ckpt_path,
    )

    n_expected = len(pd.read_csv(csv_path).query("subject == 'sub002' and task == 'bvt'"))
    results, meta = run_inference(ckpt_path, device="cpu", mode="unseen_speaker")
    assert meta["model_family"] == "image_mask", f"meta 模型族不对：{meta['model_family']}"
    assert meta["mask_mode"] == "gt", f"默认 mask_mode 应为 gt，实际 {meta['mask_mode']}"
    assert results["manner"]["preds"].shape == (n_expected,), (
        f"unseen_speaker 的 manner 预测数应为 {n_expected}，"
        f"实际 {results['manner']['preds'].shape}"
    )

    # E2 诊断路径：mask 置零也要能跑通（同一份 ckpt）
    zero_results, zero_meta = run_inference(
        ckpt_path, device="cpu", mode="unseen_speaker", mask_mode="zero"
    )
    assert zero_meta["mask_mode"] == "zero", f"mask_mode 未生效：{zero_meta['mask_mode']}"
    assert zero_results["manner"]["preds"].shape == (n_expected,), "mask 置零后形状不一致"

    # 非法取值必须报错，而不是静默按 gt 处理
    try:
        run_inference(ckpt_path, device="cpu", mode="unseen_speaker", mask_mode="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("非法 mask_mode 必须抛 ValueError")
    print("[PASS] T6 端到端 run_inference：gt / zero 两条路径 + 非法取值报错")


def main() -> int:
    """跑全部断言，任一失败即返回非零退出码。"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "USC-annot-16"
        csv_path = _build_fake_annot16(root)

        test_mask_path_from_image()
        test_rewrite_frame_strictness()
        test_mask_window_reads_neighbours(csv_path, root)
        test_onehot_and_affine_consistency(root)
        test_weight_equivalence()
        test_mask_grad_and_freezing()
        test_optimizer_and_params(root)
        test_timit_mask_audit(root)
        test_family_resolution(root)
        test_run_inference_smoke(csv_path, root)

    print("\n[test_img_mask] 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
