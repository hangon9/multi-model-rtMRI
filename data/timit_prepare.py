"""USC-TIMIT 帧级 DataFrame 构建器（评估专用）。

职责：把 USC-TIMIT 的「逐句标签 CSV + 图像序列 + 同步 wav」汇总成一张与
``USCAnnot16Dataset`` 列约定一致的 DataFrame（``subject`` / ``task`` / ``frame_idx`` /
``image_path`` / ``audio_path`` + 18 个标签列），供 TIMIT 测试集推理使用。

边界（非职责）：

- 不参与训练：本模块只造测试集表，训练链路仍然只用 USC-annot-16；
- 不缓存 CSV：约 30 万行量级、一次目录扫描即可重建（秒级），因此没有「缓存过期」
  问题（USC-annot-16 的 ``DataFrame-annot-16.csv`` 缓存是为了省掉 TextGrid 解析）；
- 不复用 ``data/USCAnnot16Loader.py`` 的取帧逻辑，也**不 import 它**：TIMIT 的图像
  命名是 ``{stem}_{i}.png``（帧号不补零、后面的帧号没有固定宽度），基类的
  ``_(\\d{4})_image\\.png$`` 正则在 TIMIT 上匹配失败并静默原样返回，会把多帧窗口退化成
  「中心帧复制 N 次」。取帧细节由 ``data/USCTIMITLoader.py`` 负责，两边解耦。

两条与「多帧窗口会不会读到坏帧」直接相关的守则：

1. 图像存在性按 **文件名**（标签 CSV 里的 ``Filename``）与目录列表比对，不按帧号集合
   比对——帧号相同但文件名不同（目录改名、CSV 写错）时，按帧号判断会放过一条磁盘上
   并不存在的路径，等到 dataloader 里才炸。
2. 保留帧之间出现**句中空洞**（中间某帧没图像）时整句跳过：多帧窗口无条件读
   ``center_idx ± half`` 的图像，句尾缺帧无害（窗口上界由 DataFrame 最大帧号兜住），
   句中缺帧则必然触发 ``FileNotFoundError``。实测真实数据没有句中空洞，本条只是防线。

线程/生命周期：纯函数式，一次调用完成扫描并返回新建的 DataFrame；无全局状态、不写文件
（除 stdout 报告）。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

#: 18 个帧级标签列，顺序与 USC-annot-16 的 ``DataFrame-annot-16.csv`` 完全一致。
#: 该顺序是硬约束：``USCAnnot16Dataset`` 用 ``vals[0:6].argmax()`` 等方式按下标取值。
DEFAULT_LABEL_COLUMNS: tuple[str, ...] = (
    "Silence", "Stop", "Nasal", "Fricative", "Approximant", "Vowel",
    "Labial", "Dental", "Alveolar", "Postalveolar", "Palatal", "Velar", "Glottal",
    "Front", "Central", "Back",
    "Voiced", "Voiceless",
)

#: 标签 CSV 中记录图像文件名的那一列（TIMIT 侧表头为 ``Filename``）
FILENAME_COLUMN = "Filename"

#: 图像文件名 → 帧号，例如 ``usctimit_mri_f1_001_005_137.png`` → ``137``。
#: 必须锚定结尾并且只取最后一组数字，否则句名里的 ``001_005`` 会被误当帧号。
_FRAME_PATTERN = re.compile(r"_(\d+)\.png$")

#: 句子 stem → (subject, task)，例如 ``usctimit_mri_f1_001_005`` → (``f1``, ``001_005``)。
#: ``subject`` 用贪婪性更弱的一段（无下划线），``task`` 吃掉剩余全部。
_STEM_PATTERN = re.compile(r"^usctimit_mri_(?P<subject>[^_]+)_(?P<task>.+)$")

#: wav 时长与「图像帧数 / fps」的允许偏差，单位帧。
#: 实测各被试有效帧率 15.05–15.15，25s 的句子约 0.25s（≈4 帧）漂移，故取 5 帧；
#: 而真正的时间轴错位在数据上表现为 1.5s 以上（≥22 帧），不会被该阈值放过。
_DURATION_TOLERANCE_FRAMES = 5.0

#: 句子被整句跳过的原因标签。
#: 注意：图像目录不存在时优先记 ``SKIP_MISSING_IMAGE_DIR``，即使该句的 wav 也缺
#: （真实数据里 ``m3/usctimit_mri_m3_331_335`` 就是图像与 wav 双缺）。
SKIP_MISSING_IMAGE_DIR = "missing_image_dir"
SKIP_MISSING_AUDIO = "missing_audio"
SKIP_NO_IMAGE_FRAME = "no_image_frame"
SKIP_FRAME_GAP = "mid_sequence_gap"


@dataclass
class SentenceStats:
    """单个句子（一段 MRI 录制）的扫描结果。

    Attributes:
        subject: 被试目录名，例如 ``f1``；来自 ``labels/<subject>/`` 的目录名。
        task: 句子在 subject 内的任务标识，例如 ``001_005``。
        n_labels: 标签 CSV 的数据行数（含之后被丢弃的行）。
        n_images: 图像目录下的 ``*.png`` 数量；句子被跳过时为 0。
        n_kept: 最终进入 DataFrame 的帧数。
        n_missing_images: 标签里点名、但图像目录里没有该文件名的帧数（该帧被丢弃）。
        n_extra_images: 图像目录里有、标签里没有点名的帧数（不进入 DataFrame）。
        n_unparsable_images: 图像目录里文件名不含帧号的文件数（同样不进入 DataFrame）。
        n_all_zero: 18 个标签全为 0 的行数；``drop_unlabeled`` 时被丢弃。
        n_duplicate_rows: 标签 CSV 里重复点名同一张图的行使数（重复行被丢弃）。
        duration: wav 时长（秒）；``check_audio_duration=False``、读取失败或无需核对时为 None。
        duration_diff_frames: ``duration`` 与 ``n_images / fps`` 之差，单位帧；不可算时为 None。
        skipped: 整句跳过的原因（``SKIP_*`` 常量）；未跳过时为 None。
    """

    subject: str
    task: str
    n_labels: int = 0
    n_images: int = 0
    n_kept: int = 0
    n_missing_images: int = 0
    n_extra_images: int = 0
    n_unparsable_images: int = 0
    n_all_zero: int = 0
    n_duplicate_rows: int = 0
    duration: float | None = None
    duration_diff_frames: float | None = None
    skipped: str | None = None


@dataclass
class TimitBuildStats:
    """整次扫描的汇总统计，供日志与测试断言使用。

    Attributes:
        sentences: 逐句统计，顺序为 ``(subject, task)`` 升序。
        fps: 本次扫描用于时长核对的名义帧率（只影响 ``duration_diff_frames``）。
    """

    sentences: list[SentenceStats] = field(default_factory=list)
    fps: float = 15.0

    @property
    def n_sentences_used(self) -> int:
        """成功进入 DataFrame 的句子数。"""
        return sum(1 for s in self.sentences if s.skipped is None)

    @property
    def n_sentences_skipped(self) -> dict[str, int]:
        """按 ``SKIP_*`` 原因分组的整句跳过计数；未出现的原因为 0（不写进字典）。"""
        counts: dict[str, int] = {}
        for stats in self.sentences:
            if stats.skipped is not None:
                counts[stats.skipped] = counts.get(stats.skipped, 0) + 1
        return counts

    @property
    def n_frames_kept(self) -> int:
        """最终进入 DataFrame 的总帧数。"""
        return sum(s.n_kept for s in self.sentences)

    @property
    def n_frames_missing_images(self) -> int:
        """因图像文件缺失而丢弃的标签帧总数。"""
        return sum(s.n_missing_images for s in self.sentences)

    @property
    def n_images_extra(self) -> int:
        """图像目录里存在、但标签里没有因而永不采样的帧总数。"""
        return sum(s.n_extra_images for s in self.sentences)

    @property
    def n_frames_all_zero(self) -> int:
        """因标签全 0 而丢弃的帧总数。"""
        return sum(s.n_all_zero for s in self.sentences)

    @property
    def n_duplicate_rows(self) -> int:
        """因重复点名同一张图而丢弃的行总数。"""
        return sum(s.n_duplicate_rows for s in self.sentences)

    @property
    def n_duration_probe_failed(self) -> int:
        """可用句子中 wav 头信息读不出来的句数（时长核对静默失效时要能看出来）。"""
        return sum(
            1 for s in self.sentences if s.skipped is None and s.duration is None
        )

    @property
    def duration_warnings(self) -> list[SentenceStats]:
        """音频时长与帧数推算时长偏差超阈值的句子（潜在时间轴错位）。"""
        return [
            s
            for s in self.sentences
            if s.duration_diff_frames is not None
            and abs(s.duration_diff_frames) > _DURATION_TOLERANCE_FRAMES
        ]


def _match_frame_index(filename: str) -> int | None:
    """从图像文件名解析帧号，解析不出时返回 None。

    Args:
        filename: 图像文件名，例如 ``usctimit_mri_f1_001_005_137.png``。

    Returns:
        帧号（非负整数，图像序列内从 0 开始）；文件名不以 ``_<数字>.png`` 结尾时返回 None。

    Note:
        这里返回 None 而不是抛错，是因为目录里可能混入无关 png（由调用方计入
        ``n_unparsable_images``）；真正致命的是「路径重写后拼出一条不存在的路径」，
        那个由 ``USCTIMITDataset._frame_path`` 响亮报错。
    """
    match = _FRAME_PATTERN.search(str(filename))
    return int(match.group(1)) if match is not None else None


def _list_image_files(image_dir: Path) -> tuple[dict[str, int], int]:
    """列出图像目录中的文件名与其帧号。

    Args:
        image_dir: 单句的图像目录，例如 ``images/f1/usctimit_mri_f1_001_005/``。

    Returns:
        ``({文件名: 帧号}, 无法解析帧号的文件数)``。非 ``.png`` 文件与子目录不计入任何一项。

    Note:
        按**文件名**而不是帧号集合记录：「标签里的 Filename 是否真实存在」这一判断必须
        精确到文件名，同号不同名 / 同名不同号都属于不一致，按帧号比较会漏掉后者。
    """
    files: dict[str, int] = {}
    unparsable = 0
    for name in os.listdir(image_dir):
        if not name.endswith(".png"):
            continue
        frame_idx = _match_frame_index(name)
        if frame_idx is None:
            unparsable += 1  # 命名异常但仍是 png：计入报告，不参与取帧
            continue
        files[name] = frame_idx
    return files, unparsable


def _build_record(
    subject: str,
    task: str,
    frame_idx: int,
    image_path: str,
    audio_path: str,
    label_values,
    label_columns: tuple[str, ...],
) -> dict:
    """拼一条帧级记录（列名与顺序由调用方统一后续排序保证）。

    Args:
        subject: 被试目录名。
        task: 句子任务标识。
        frame_idx: 帧号（取自图像文件名，不用 CSV 行号）。
        image_path: 该帧图像路径。
        audio_path: 该句同步音频路径。
        label_values: 该行的 18 个标签值（``numpy`` 行向量，按 ``label_columns`` 顺序）。
        label_columns: 标签列名与顺序。

    Returns:
        单个样本的字段字典。
    """
    record = {
        "subject": subject,
        "task": task,
        "frame_idx": frame_idx,
        "image_path": image_path,
        "audio_path": audio_path,
    }
    for col_idx, col in enumerate(label_columns):
        record[col] = label_values[col_idx]
    return record


def _find_frame_gap(
    records: list[dict], available_frames: set[int]
) -> tuple[int, int] | None:
    """检查一批待入表记录是否存在「句中缺图」。

    Args:
        records: 单句的待入表记录（每条含 ``frame_idx``），可能在句首/句尾缺帧。
        available_frames: 该句图像目录里真实存在的帧号集合。

    Returns:
        保留帧的 ``[min, max]`` 区间内存在「无图像帧」时返回 ``(空洞起始帧号, 空洞结束帧号)``，
        否则返回 None。句首/句尾的连续缺失不算空洞：多帧窗口的边界帧会被
        ``USCAnnot16Dataset`` 按 DataFrame 最大帧号丢弃，不会读到不存在的图像。

    Note:
        判据是「图像是否存在」而不是「是否有记录」：标签全 0 而被丢掉的帧不会产生记录，
        但图像仍存在、窗口照样能读到，因此不算空洞。
    """
    frames = sorted(int(r["frame_idx"]) for r in records)
    if len(frames) < 2:
        return None
    missing_inside = [
        frame
        for frame in range(frames[0], frames[-1] + 1)
        if frame not in available_frames
    ]
    if not missing_inside:
        return None
    return missing_inside[0], missing_inside[-1]


def _probe_duration(audio_path: Path) -> float | None:
    """读取 wav 时长（秒），只读文件头、不解码波形。

    Args:
        audio_path: 单句的同步音频路径。

    Returns:
        时长（秒）；文件读不到或后端不支持时返回 None。返回 None 的句数由
        :attr:`TimitBuildStats.n_duration_probe_failed` 统计并在报告里打印，
        这样「核对静默失效」与「核对通过」不会混为一谈。
    """
    try:
        import torchaudio

        info = torchaudio.info(str(audio_path))
        if not info.sample_rate:
            return None
        return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        return None


def _iter_label_files(root: Path, subjects: list[str] | None) -> list[tuple[str, Path]]:
    """枚举待扫描的标签 CSV。

    Args:
        root: USC-TIMIT 根目录，内含 ``labels/`` / ``images/`` / ``audios/``。
        subjects: 只扫描这些被试目录（例如 ``["f1"]``）；None 表示 ``labels/`` 下全部。

    Returns:
        ``[(subject, csv_path), ...]``，按 ``(subject, 文件名)`` 升序。

    Raises:
        FileNotFoundError: ``labels/`` 不存在时抛出；空目录返回空列表。
        ValueError: 指定的 ``subjects`` 中有目录不存在时抛出（拼错被试名必须立刻报错）。
    """
    labels_root = root / "labels"
    if not labels_root.is_dir():
        raise FileNotFoundError(f"USC-TIMIT labels 目录不存在：{labels_root}")

    available = sorted(p.name for p in labels_root.iterdir() if p.is_dir())
    if subjects is None:
        wanted = available
    else:
        missing = [s for s in subjects if s not in available]
        if missing:
            raise ValueError(
                f"USC-TIMIT 中不存在这些被试目录：{missing}；现有 {available}。"
            )
        wanted = list(subjects)

    pairs: list[tuple[str, Path]] = []
    for subject in wanted:
        for csv_path in sorted((labels_root / subject).glob("*.csv")):
            pairs.append((subject, csv_path))
    return pairs


def scan_timit(
    root: str | Path,
    label_columns: tuple[str, ...] = DEFAULT_LABEL_COLUMNS,
    subjects: list[str] | None = None,
    drop_unlabeled: bool = True,
    fps: float = 15.0,
    check_audio_duration: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, TimitBuildStats]:
    """扫描 USC-TIMIT 目录，返回帧级 DataFrame 与完整统计。

    帧数以标签 CSV 为准、图像目录为约束：

    - 标签点名但图像缺失的帧丢弃（实测只在句尾）；
    - 图像里有但标签没点名的帧直接不采样（不会成为样本，也不会被邻帧窗口读到，
      因为 ``USCAnnot16Dataset`` 的窗口上界取自 DataFrame 的最大帧号）；
    - 保留帧之间出现句中空洞时**整句跳过**（多帧窗口会无条件读 ``center ± half``，
      句中缺图必炸；句尾缺图无害）。

    Args:
        root: USC-TIMIT 根目录，须含 ``labels/<subject>/*.csv``、``images/<subject>/<stem>/``、
            ``audios/<subject>/<stem>.wav``。
        label_columns: 标签列名与顺序，默认 :data:`DEFAULT_LABEL_COLUMNS`；必须与
            ``USCAnnot16Dataset`` 的下标取值假设一致。
        subjects: 只扫描这些被试；None 表示全部。拼错会抛 ValueError。
        drop_unlabeled: True 时丢弃 18 个标签全为 0 的行。``argmax([0]*6) == 0`` 会被
            下游解释成 Silence，静默抬高 silence 占比，故默认丢弃。
        fps: 名义帧率，只用于「图像帧数 / fps vs wav 时长」的诊断；不影响帧号与切窗。
        check_audio_duration: 是否读 wav 文件头做时长核对；False 时跳过（大批量重复
            调用可省掉一次全量 stat/info）。
        verbose: 是否打印逐句异常与分组汇总。

    Returns:
        ``(dataframe, stats)``。DataFrame 列序固定为 ``subject, task, frame_idx,
        image_path, audio_path, <label_columns...>``，按 ``(subject, task, frame_idx)``
        升序；``image_path`` / ``audio_path`` 为字符串（Windows 与 HPC 通用）。

    Raises:
        FileNotFoundError: ``root/labels`` 不存在。
        ValueError: 指定被试不存在、标签 CSV 缺列、或图像文件名无法解析帧号。

    Note:
        只有 DataFrame 里的帧会被 ``USCTIMITDataset`` 采到，因此「多余图像帧」既不会成为
        样本，也不会被多帧窗口当作邻帧读到（基类的窗口上界取自 DataFrame 的最大帧号）。
    """
    root = Path(root)
    label_columns = tuple(label_columns)
    stats = TimitBuildStats(fps=float(fps))
    records: list[dict] = []

    for subject, csv_path in _iter_label_files(root, subjects):
        stem = csv_path.stem
        match = _STEM_PATTERN.match(stem)
        # 前缀不符时退化为「整段 stem 当 task」，保持可评估但不做无根据的猜测
        task = match.group("task") if match else stem
        image_dir = root / "images" / subject / stem
        audio_path = root / "audios" / subject / f"{stem}.wav"

        sent = SentenceStats(subject=subject, task=task)

        # 句子级前置检查：缺图目录 / 缺 wav 直接整句跳过，避免逐帧刷 300 次失败
        if not image_dir.is_dir():
            sent.skipped = SKIP_MISSING_IMAGE_DIR
            stats.sentences.append(sent)
            continue
        if not audio_path.is_file():
            sent.skipped = SKIP_MISSING_AUDIO
            stats.sentences.append(sent)
            continue

        table = pd.read_csv(csv_path)
        missing_cols = [
            col for col in (FILENAME_COLUMN, *label_columns) if col not in table.columns
        ]
        if missing_cols:
            raise ValueError(f"{csv_path} 缺少必需列：{missing_cols}")
        sent.n_labels = len(table)

        image_files, sent.n_unparsable_images = _list_image_files(image_dir)
        sent.n_images = len(image_files)
        if not image_files:
            sent.skipped = SKIP_NO_IMAGE_FRAME
            stats.sentences.append(sent)
            continue

        # 存在性按文件名比对：同帧号但文件名不同也是不符，按帧号比会放过不存在的路径
        filenames = [str(name) for name in table[FILENAME_COLUMN]]
        sent.n_missing_images = sum(1 for name in filenames if name not in image_files)
        sent.n_extra_images = len(set(image_files) - set(filenames))

        # 逐行组记录：只保留「图像文件名真实存在、且（可选）标签非全 0」的帧
        label_values = table.loc[:, list(label_columns)].to_numpy()
        pending: list[dict] = []
        seen_names: set[str] = set()
        for row_idx, filename in enumerate(filenames):
            if filename not in image_files:
                continue  # 该帧没有对应图像，无法评估，丢弃（只计数）
            if filename in seen_names:
                sent.n_duplicate_rows += 1
                continue  # 同一张图被重复标注：保留首次出现，避免同一帧被算两次
            seen_names.add(filename)
            if drop_unlabeled and not label_values[row_idx].any():
                sent.n_all_zero += 1
                continue
            pending.append(
                _build_record(
                    subject=subject,
                    task=task,
                    frame_idx=image_files[filename],
                    image_path=str(image_dir / filename),
                    audio_path=str(audio_path),
                    label_values=label_values[row_idx],
                    label_columns=label_columns,
                )
            )

        # 句中空洞守卫：保留帧的 [min, max] 区间里只要有帧缺图，多帧窗口就会读到不存在的
        # 文件。句尾缺帧不算空洞（窗口上界由 DataFrame 最大帧号兜住），故只看区间内部。
        gap = _find_frame_gap(pending, set(image_files.values()))
        if gap is not None:
            print(
                f"[TIMIT] WARNING: 跳过 {subject}/{task}：帧 {gap[0]}..{gap[1]} 之间存在缺图的帧，"
                "多帧窗口会读到不存在的图像（实测数据无此情况，请核对数据完整性）"
            )
            sent.skipped = SKIP_FRAME_GAP
            stats.sentences.append(sent)
            continue

        records.extend(pending)
        sent.n_kept = len(pending)

        # 音频时长核对：图像序列按 ≈15fps 与 wav 同步，偏差过大意味着时间轴错位，需人工看
        if check_audio_duration:
            sent.duration = _probe_duration(audio_path)
            if sent.duration is not None and sent.n_images:
                expected = sent.n_images / float(fps)
                sent.duration_diff_frames = (sent.duration - expected) * float(fps)

        stats.sentences.append(sent)

    frame_columns = ["subject", "task", "frame_idx", "image_path", "audio_path"]
    dataframe = pd.DataFrame.from_records(records, columns=[*frame_columns, *label_columns])
    if not dataframe.empty:
        dataframe = dataframe.sort_values(
            ["subject", "task", "frame_idx"], kind="mergesort"
        ).reset_index(drop=True)

    if verbose:
        _print_report(dataframe, stats)

    if dataframe.empty:
        raise RuntimeError(
            f"USC-TIMIT 在 {root} 下没有可评估的帧：全部句子都被跳过（原因见上面的报告）。"
        )
    return dataframe, stats


def _print_report(dataframe: pd.DataFrame, stats: TimitBuildStats) -> None:
    """打印扫描报告：逐句异常、按被试汇总、全局合计。

    Args:
        dataframe: :func:`scan_timit` 产出的帧级表（只读取 ``subject`` 列做交叉校验）。
        stats: 同一次扫描的统计对象。

    Returns:
        None。只写 stdout。

    Note:
        逐句正常（无缺失 / 无多余 / 无全 0）的句子不单独打印，避免 800+ 行噪声；
        异常句子与跳过句子一定打印，因为它们直接影响可评估帧数与对齐正确性。
    """
    skipped = stats.n_sentences_skipped
    for sent in stats.sentences:
        if sent.skipped is not None:
            print(f"[TIMIT] 跳过 {sent.subject}/{sent.task}：{sent.skipped}")
            continue
        if (
            sent.n_missing_images
            or sent.n_extra_images
            or sent.n_all_zero
            or sent.n_duplicate_rows
        ):
            print(
                f"[TIMIT] {sent.subject}/{sent.task}：标签 {sent.n_labels} 行，"
                f"图像 {sent.n_images} 帧，入表 {sent.n_kept} 帧，"
                f"缺图 {sent.n_missing_images} 帧，多余图 {sent.n_extra_images} 帧，"
                f"全 0 标签 {sent.n_all_zero} 行，重复行 {sent.n_duplicate_rows}"
            )

    # 按被试汇总：跨被试的差异（缺失帧、多余帧、时长偏差）是域偏移分析的第一手信息
    print("[TIMIT] 按被试汇总：")
    print(
        f"  {'subject':8}{'sent':>6}{'frames':>9}{'no_img':>8}{'extra':>7}"
        f"{'all_zero':>9}{'skip':>6}{'dur_warn':>9}"
    )
    for subject in sorted({s.subject for s in stats.sentences}):
        subset = [s for s in stats.sentences if s.subject == subject]
        warn = sum(
            1
            for s in subset
            if s.duration_diff_frames is not None
            and abs(s.duration_diff_frames) > _DURATION_TOLERANCE_FRAMES
        )
        print(
            f"  {subject:8}{sum(1 for s in subset if s.skipped is None):>6}"
            f"{sum(s.n_kept for s in subset):>9}"
            f"{sum(s.n_missing_images for s in subset):>8}"
            f"{sum(s.n_extra_images for s in subset):>7}"
            f"{sum(s.n_all_zero for s in subset):>9}"
            f"{sum(1 for s in subset if s.skipped is not None):>6}"
            f"{warn:>9}"
        )

    print(
        f"[TIMIT] 合计：句子 {stats.n_sentences_used} 句可用 / "
        f"{sum(skipped.values())} 句跳过 {skipped or {}}，"
        f"帧 {stats.n_frames_kept}（DataFrame {len(dataframe)} 行），"
        f"缺图帧 {stats.n_frames_missing_images}，多余图帧 {stats.n_images_extra}，"
        f"全 0 标签帧 {stats.n_frames_all_zero}，重复行 {stats.n_duplicate_rows}"
    )

    if stats.n_duration_probe_failed:
        print(
            f"[TIMIT] WARNING: {stats.n_duration_probe_failed} 句的 wav 头信息读不出来，"
            "时长核对未生效（请检查 torchaudio 后端）"
        )

    warnings = stats.duration_warnings
    for sent in warnings[:10]:
        print(
            f"[TIMIT] WARNING: {sent.subject}/{sent.task} 的 wav 时长 "
            f"{sent.duration:.2f}s 与 {sent.n_images} 帧 / {stats.fps:g}fps 推算值偏差 "
            f"{sent.duration_diff_frames:+.1f} 帧，可能存在音频时间轴错位"
        )
    if len(warnings) > 10:
        print(f"[TIMIT] WARNING: 另有 {len(warnings) - 10} 句存在同类时长偏差（已省略明细）")


def build_timit_dataframe(
    root: str | Path,
    label_columns: tuple[str, ...] = DEFAULT_LABEL_COLUMNS,
    subjects: list[str] | None = None,
    drop_unlabeled: bool = True,
    fps: float = 15.0,
    check_audio_duration: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """构建 USC-TIMIT 测试集 DataFrame（:func:`scan_timit` 的免统计入口）。

    Args:
        root: USC-TIMIT 根目录。
        label_columns: 标签列名与顺序。
        subjects: 只保留这些被试；None 表示全部。
        drop_unlabeled: 是否丢弃标签全 0 的帧。
        fps: 名义帧率，仅用于音频时长核对。
        check_audio_duration: 是否做音频时长核对。
        verbose: 是否打印扫描报告。

    Returns:
        帧级 DataFrame，列序与排序见 :func:`scan_timit`。

    Raises:
        RuntimeError: 没有任何可评估帧时抛出。
        ValueError: 被试目录不存在 / 标签 CSV 缺列 / 文件名无法解析帧号。
    """
    dataframe, _ = scan_timit(
        root,
        label_columns=label_columns,
        subjects=subjects,
        drop_unlabeled=drop_unlabeled,
        fps=fps,
        check_audio_duration=check_audio_duration,
        verbose=verbose,
    )
    return dataframe
