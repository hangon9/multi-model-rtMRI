"""T3：学生 checkpoint 与评估链的兼容性测试。

验证三件事：

1. ``train_distill._build_checkpoint`` 产出的 ``model_state_dict`` 里不含投影头参数；
2. 同一份 state_dict 能被评估链 ``src.eval.inference._load_state`` 以 image 族严格加载；
3. 蒸馏实验名能被正确判成 image 族（否则评估链会按 contrastive 宽松加载并拿到随机权重）。

不需要真实数据与教师 ckpt：全部用极小学生模型 + 内存里的假投影头。

运行：``python _tmp_verify/test_student_ckpt_eval.py``。
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.inference import (  # noqa: E402
    _EXPERIMENT_TO_FAMILY,
    _load_state,
    _resolve_model_family,
)
from src.losses.distill_loss import DistillLoss  # noqa: E402
from src.models.img_only_model import ImageMultiheadClassifier  # noqa: E402
from train_distill import _build_checkpoint  # noqa: E402

#: 与 train_img_baseline.resolve_model_params 完全同构的极小学生参数
STUDENT_PARAMS = {
    "num_classes": 18,
    "img_size": 32,
    "patch_size": 16,
    "hidden_size": 32,
    "mlp_dim": 64,
    "clf_hidden_dim": 16,
    "num_layers": 1,
    "num_heads": 4,
    "dropout": 0.0,
    "classification_task": "",
    "model_name": "vit",
    "pretrained": False,
    "freeze_layers": 0,
    "temporal_type": "none",
    "conformer_layers": 1,
    "conformer_heads": 4,
    "conv_kernel_size": 3,
}


def _build_student():
    """构建一个极小学生模型（无预训练权重，不联网）。"""
    return ImageMultiheadClassifier(**STUDENT_PARAMS)


def test_checkpoint_separates_projector_from_model_state():
    """``model_state_dict`` 只含学生；投影头单独存放。"""
    model = _build_student()
    distill = DistillLoss(mode="logit_feat", student_dim=32, teacher_dim=32)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(distill.projector.parameters()), lr=1e-3
    )
    fake_teacher = SimpleNamespace(
        meta={"checkpoint": "fake.pt", "path_source": "fold_file", "fold": 1}
    )

    ckpt = _build_checkpoint(
        fold_id=1,
        epoch=0,
        model=model,
        distill=distill,
        optimizer=optimizer,
        best_val_loss=0.5,
        teacher=fake_teacher,
        config={"data": {}},
        checkpoint_type="fold_best",
    )

    assert ckpt["projector_state_dict"] is not None
    assert ckpt["checkpoint_type"] == "fold_best"
    assert ckpt["teacher_info"] == fake_teacher.meta
    assert not any(k.startswith("projector") for k in ckpt["model_state_dict"])
    # 学生侧键集合必须与模型自身完全一致（多一个就会让严格加载失败）
    assert set(ckpt["model_state_dict"]) == set(model.state_dict())


def test_model_state_dict_loads_strictly_in_eval_chain():
    """评估链对 image 族是严格加载；学生 state_dict 必须能原样通过。"""
    model = _build_student()
    distill = DistillLoss(mode="logit_feat", student_dim=32, teacher_dim=32)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(distill.projector.parameters()), lr=1e-3
    )
    ckpt = _build_checkpoint(
        fold_id=2,
        epoch=3,
        model=model,
        distill=distill,
        optimizer=optimizer,
        best_val_loss=0.42,
        teacher=SimpleNamespace(meta={}),
        config={"data": {}},
        checkpoint_type=None,
    )

    rebuilt = _build_student()
    # 严格加载：多键/缺键都会抛 RuntimeError
    _load_state(rebuilt, ckpt["model_state_dict"], "image")

    for (name_a, param_a), (name_b, param_b) in zip(
        model.state_dict().items(), rebuilt.state_dict().items()
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b), name_a

    # 顺带确认 checkpoint 结构里没有多余键混进 model_state_dict
    assert "checkpoint_type" not in ckpt, "全局最优 ckpt 保持与基线一致的字段集合"


def test_experiment_names_resolve_to_image_family():
    """蒸馏实验名必须被判成 image 族，否则评估会按错模型族加载。"""
    for name in ("kd_img_ResNet50_logit", "kd_img_ResNet50_logit_feat"):
        assert _EXPERIMENT_TO_FAMILY.get(name) == "image", name
        family = _resolve_model_family(
            {"experiment_name": name}, Path("checkpoints") / "best_model.pt"
        )
        assert family == "image", (name, family)


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"[OK] {name}")
    print("T3 全部通过")
