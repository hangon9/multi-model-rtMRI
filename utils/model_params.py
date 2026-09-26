"""模型参数日志工具。

训练脚本在构建模型前，先把「模型实际接收到的参数」整理成字典，再用本模块的函数写入
training.log。日志里记录的是代码里最终生效的取值（含代码里的默认值），而不仅仅是
config 文件里写了什么，方便复现实验、排查配置键名写错导致的静默回退。

职责边界：本模块只做「字典 → 日志行」的格式化输出，不读 config、不构建模型、不校验
取值合法性。因此「日志里的值 == 模型构造时的实参」由调用方保证：训练脚本必须把同一份
参数字典既用于写日志、又用于构造模型（见各 train_*.py 里的 resolve_model_params）。

层次与约束：属于 utils 层的纯函数工具，无全局状态、无线程约束；唯一的副作用是通过
调用方传入的 logger 写日志。中文文案直接写入日志，与项目其它注释语言保持一致。
"""


def flatten_params(params: dict, prefix: str = ""):
    """把（可能嵌套的）参数字典展开成 (点号路径, 值) 序列。

    嵌套字典用点号连接成路径（{"model": {"img_size": 128}} → ("model.img_size", 128)），
    便于和 config 文件的层级逐项对照。

    Args:
        params: 待展开的参数字典；允许任意层嵌套，叶子值可以是任意类型（含 None）。
        prefix: 递归用的路径前缀，调用方无需传入。

    Yields:
        (path, value) 二元组。顺序为字典的插入顺序（Python 3.7+ 保证），因此调用方可以
        通过调整字典的构造顺序来控制日志行的排列顺序。
    """
    # 逐项展开：值仍是字典时继续递归，其它类型直接作为叶子产出
    for key, value in params.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            yield from flatten_params(value, path)
        else:
            yield path, value


def format_model_params(params: dict, indent: str = "  ") -> list:
    """把参数字典格式化为对齐的文本行，便于阅读。

    Args:
        params: 参数字典，通常来自各 train_*.py 的 resolve_model_params。
        indent: 每行的前缀缩进，默认两个空格。

    Returns:
        文本行列表，可直接逐行交给 logger.info。参数字典为空时返回
        [f"{indent}(无可打印参数)"] 而不是空列表——日志里始终留有痕迹，
        便于区分「没有参数可打印」和「这一段落根本没被执行」。
    """
    items = list(flatten_params(params))
    if not items:
        return [f"{indent}(无可打印参数)"]

    # 以最长路径决定填充宽度，让冒号在视觉上对齐；纯显示用途，不改变任何取值
    width = max(len(path) for path, _ in items)
    return [f"{indent}{path.ljust(width)} : {value}" for path, value in items]


def log_model_params(
    logger,
    model_name: str,
    params: dict,
    title: str = "模型实际接收参数",
) -> None:
    """把模型实际接收到的参数打印到日志。

    Args:
        logger: 提供 info(str) 方法的日志器，通常是 utils.logger.TrainingLogger。
        model_name: 写进标题的模型类名（如 "ImageMultiheadClassifier"），用于在多次
            实验或多模型对比时快速定位日志段落。
        params: 模型实际接收的参数；必须已补齐代码默认值，由调用方解析后传入。
        title: 标题文案，默认「模型实际接收参数」。

    Side effects:
        向 logger 写入 1 + N 行文本（N 为展开后的参数个数）；无返回值、不修改入参。

    Note:
        取值不做类型转换：None 按 Python 的 "None" 打印（例如某个可选项未启用），
        浮点数按原始精度输出，以免日志与真实取值不符。
    """
    logger.info(f"[{model_name}] {title}:")
    for line in format_model_params(params):
        logger.info(line)
