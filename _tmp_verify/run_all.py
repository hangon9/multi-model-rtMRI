"""一键跑完仓库的本地测试（形状 / 数值 / 兼容性 / 冒烟过拟合 / TIMIT 跨域评估）。

这些测试只做本地验证，不涉及真实训练，也不会联网：
- 音频骨干用桩件替换，图像编码器用 ``vit``（无预训练权重）；
- teacher / student checkpoint 都是内存里现场生成的极小 ckpt；
- USC-TIMIT 相关测试用 ``tempfile`` 造假数据，不读真实语料。

用法::

    conda activate rtmri
    python _tmp_verify/run_all.py

每个模块都以子进程运行，任一失败即整体返回非零退出码，便于在 CI/HPC 登录节点上直接用。
真实训练（含真实数据与 wav2vec2 权重）请勿在本机跑，交给 HPC。
"""

import subprocess
import sys
from pathlib import Path

#: 仓库根目录；子进程以它为工作目录，保证相对导入（``_tmp_verify.xxx``）成立
REPO_ROOT = Path(__file__).resolve().parents[1]
#: 按计划顺序排列的测试脚本（先教师、再损失、最后端到端与跨域评估）
TEST_SCRIPTS = (
    "test_teacher_resolve.py",
    "test_teacher_load.py",
    "test_data_inherit.py",
    "test_distill_loss.py",
    "test_student_ckpt_eval.py",
    "check_bce_scale_numpy.py",
    "test_overfit_smoke.py",
    "test_train_loop_integration.py",
    # F1 选模 / 验证集 shuffle：共享模块 + 各流水线的接入点 + eval 侧的画图
    "test_f1_module.py",
    "test_f1_select_pipelines.py",
    "test_heads_summary.py",
    "test_contrast_f1_select.py",
    # USC-TIMIT 跨域评估：先是数据层，再是评估链路（均基于假数据）
    "test_timit_prepare.py",
    "test_timit_loader.py",
    "test_eval_timit_smoke.py",
    # img_mask_oracle：mask 数据层 / 模型 / 评估接入（假数据，CPU）
    "test_img_mask.py",
)


def main() -> int:
    """逐个运行测试脚本并汇总结果。

    Returns:
        0 表示全部通过；否则返回失败个数。
    """
    failed = []
    for script in TEST_SCRIPTS:
        path = REPO_ROOT / "_tmp_verify" / script
        print(f"\n{'=' * 70}\n[run_all] {script}\n{'=' * 70}", flush=True)
        # -X utf8：Windows 控制台默认 GBK，中文断言信息会编码失败
        result = subprocess.run(
            [sys.executable, "-X", "utf8", str(path)],
            cwd=str(REPO_ROOT),
        )
        if result.returncode != 0:
            failed.append(script)

    print(f"\n{'=' * 70}")
    if failed:
        print(f"[run_all] 失败 {len(failed)} 个：{', '.join(failed)}")
        return len(failed)
    print(f"[run_all] 全部 {len(TEST_SCRIPTS)} 个脚本通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
