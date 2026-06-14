##############################################################################
## The MIT License (MIT)
##
## Copyright (c) 2016 Kyler Brown
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
##############################################################################

#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
from pathlib import Path
import codecs
import argparse


Entry = namedtuple(
    "Entry",
    [
        "start",
        "stop",
        "name",
        "tier"
    ]
)


def _looks_like_textgrid(text):
    """
    判断解码后的文本是否像 Praat TextGrid 文件。

    这个函数主要用于处理没有 BOM 的 UTF-16 LE / UTF-16 BE 文件。
    如果直接 decode 成错误编码，可能不会立刻报错，但文本内容会不正常。
    """
    sample = text[:5000]

    markers = [
        "TextGrid",
        "ooTextFile",
        "xmin",
        "xmax",
        "tiers",
        "item",
        "intervals",
        "IntervalTier",
        "TextTier"
    ]

    return any(marker in sample for marker in markers)


def _decode_textgrid_bytes(raw_bytes, filename=None):
    """
    使用多种编码尝试解码 TextGrid 文件。

    支持的常见编码包括：
        - utf-8-sig
        - utf-8
        - utf-16
        - utf-16-le
        - utf-16-be
        - utf-32
        - utf-32-le
        - utf-32-be
        - gb18030
        - big5
        - cp1252
        - latin-1

    返回：
        text, used_encoding
    """

    bom_encodings = []

    # 注意：UTF-32 的 BOM 判断必须放在 UTF-16 前面。
    if raw_bytes.startswith(codecs.BOM_UTF32_LE):
        bom_encodings.append("utf-32-le")
    elif raw_bytes.startswith(codecs.BOM_UTF32_BE):
        bom_encodings.append("utf-32-be")
    elif raw_bytes.startswith(codecs.BOM_UTF16_LE):
        bom_encodings.append("utf-16-le")
    elif raw_bytes.startswith(codecs.BOM_UTF16_BE):
        bom_encodings.append("utf-16-be")
    elif raw_bytes.startswith(codecs.BOM_UTF8):
        bom_encodings.append("utf-8-sig")

    fallback_encodings = [
        "utf-8-sig",
        "utf-8",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "utf-32",
        "utf-32-le",
        "utf-32-be",
        "gb18030",
        "big5",
        "cp1252",
        "latin-1",
    ]

    # 去重，同时保持顺序
    encodings = []
    for enc in bom_encodings + fallback_encodings:
        if enc not in encodings:
            encodings.append(enc)

    decoded_candidates = []
    errors = []

    for enc in encodings:
        try:
            text = raw_bytes.decode(enc, errors="strict")
            text = text.lstrip("\ufeff")

            if _looks_like_textgrid(text):
                return text, enc

            decoded_candidates.append((enc, text))

        except UnicodeDecodeError as e:
            errors.append((enc, str(e)))

    # 如果没有任何解码结果明显像 TextGrid，
    # 但某些编码可以成功解码，则返回第一个可解码结果。
    if decoded_candidates:
        enc, text = decoded_candidates[0]
        return text, enc

    error_message = "Could not decode TextGrid file"

    if filename is not None:
        error_message += f": {filename}"

    if errors:
        error_message += "\nTried encodings:\n"
        error_message += "\n".join(
            f"  - {enc}: {err}"
            for enc, err in errors
        )

    raise UnicodeDecodeError(
        "unknown",
        raw_bytes,
        0,
        min(1, len(raw_bytes)),
        error_message
    )


def _read_textgrid_file(filename):
    """
    读取 TextGrid 文件，并自动处理不同编码。

    返回格式与原来的 _read(f) 保持一致：
        [
            line.strip(),
            line.strip(),
            ...
        ]
    """
    raw_bytes = Path(filename).read_bytes()
    text, used_encoding = _decode_textgrid_bytes(
        raw_bytes,
        filename=filename
    )

    return [line.strip() for line in text.splitlines()]


def _read(f):
    """
    保留原来的 _read(f) 接口，避免其他代码依赖它时报错。

    但 read_textgrid() 现在不再直接使用这个函数，
    而是使用 _read_textgrid_file(filename)，从而支持多种编码。
    """
    return [x.strip() for x in f.readlines()]


def _get_float_val(string):
    """
    从一行 TextGrid 文本中取出最后一个字段，并转成 float。

    例如：
        xmin = 0.123
    返回：
        0.123
    """
    return float(string.split()[-1])


def _get_str_val(string):
    """
    从一行 TextGrid 文本中取出双引号中的字符串。

    例如：
        text = "AH0"
    返回：
        AH0
    """
    parts = string.split('"')

    if len(parts) >= 3:
        return parts[-2]

    # 兼容一些没有双引号的异常格式
    if "=" in string:
        return string.split("=", 1)[1].strip()

    return ""


def _find_tiers(interval_lines, tier_lines, tiers):
    """
    根据 interval 所在行号判断其属于哪个 tier。

    interval_lines:
        intervals [1]:
        intervals [2]:
        ...

    tier_lines:
        name = "phones"
        name = "words"
        ...

    tiers:
        phones
        words
        ...
    """
    if not tier_lines:
        return [None for _ in interval_lines]

    tier_pairs = list(zip(tier_lines, tiers))
    result = []

    current_tier = tier_pairs[0][1]
    next_index = 1

    for interval_line in interval_lines:
        while (
            next_index < len(tier_pairs)
            and interval_line > tier_pairs[next_index][0]
        ):
            current_tier = tier_pairs[next_index][1]
            next_index += 1

        result.append(current_tier)

    return result


def _build_entry(i, content, tier):
    """
    根据 TextGrid 中 intervals 或 points 的起始行构造 Entry。

    Long TextGrid interval 格式通常是：
        intervals [1]:
            xmin = 0
            xmax = 0.1
            text = "AH0"

    Long TextGrid point 格式通常是：
        points [1]:
            number = 0.1
            mark = "x"
    """
    start = _get_float_val(content[i + 1])

    if content[i].startswith("intervals "):
        offset = 1
        stop = _get_float_val(content[i + 1 + offset])
        label = _get_str_val(content[i + 2 + offset])
    else:
        # TextTier point object
        offset = 0
        stop = start
        label = _get_str_val(content[i + 2 + offset])

    return Entry(
        start=start,
        stop=stop,
        name=label,
        tier=tier
    )


def _parse_long_textgrid(content, tierName="phones"):
    """
    解析 Praat long TextGrid 格式。
    """
    tier_lines = []
    tiers = []

    interval_lines = []

    for i, line in enumerate(content):
        if line.startswith("name ="):
            tier_lines.append(i)
            tiers.append(_get_str_val(line))

        elif line.startswith("intervals ") or line.startswith("points "):
            interval_lines.append(i)

    interval_tiers = _find_tiers(
        interval_lines=interval_lines,
        tier_lines=tier_lines,
        tiers=tiers
    )

    entries = []

    for line_index, tier in zip(interval_lines, interval_tiers):
        if tierName is None or tier == tierName:
            try:
                entry = _build_entry(
                    line_index,
                    content,
                    tier
                )
                entries.append(entry)
            except Exception:
                # 如果某个 interval 格式异常，跳过该 interval
                # 避免整个 TextGrid 解析失败
                continue

    return entries


def _parse_short_textgrid(content, tierName="phones"):
    """
    尝试解析 Praat short TextGrid 格式。

    说明：
    short TextGrid 没有 xmin = / xmax = / text = 这样的字段名，
    结构更依赖行顺序。这个解析器覆盖常见 IntervalTier short 格式。

    如果你的文件主要是 long TextGrid，可以不用关心这个函数。
    """

    entries = []
    i = 0
    current_tier = None

    while i < len(content):
        line = content[i]

        if line == '"IntervalTier"' or line == '"TextTier"':
            tier_type = line.strip('"')

            if i + 1 < len(content):
                current_tier = content[i + 1].strip('"')
            else:
                current_tier = None

            i += 1
            continue

        if current_tier is not None and (tierName is None or current_tier == tierName):
            # 尝试匹配 short interval:
            # xmin
            # xmax
            # "label"
            try:
                start = float(content[i])
                stop = float(content[i + 1])
                label_line = content[i + 2]

                # label 通常带引号
                if label_line.startswith('"') and label_line.endswith('"'):
                    label = label_line.strip('"')
                    entries.append(
                        Entry(
                            start=start,
                            stop=stop,
                            name=label,
                            tier=current_tier
                        )
                    )
                    i += 3
                    continue

            except Exception:
                pass

        i += 1

    return entries


def read_textgrid(filename, tierName="phones"):
    """
    Reads a TextGrid file into a list of Entry objects.

    Each Entry has the following fields:
        - start
        - stop
        - name
        - tier

    参数：
        filename:
            TextGrid 文件路径。

        tierName:
            要读取的 tier 名称，默认是 'phones'。
            如果传入 None，则读取所有 tier。

    返回：
        [
            Entry(start=..., stop=..., name=..., tier=...),
            ...
        ]

    这个版本支持多种文件编码，包括 UTF-16 BE。
    """

    content = _read_textgrid_file(filename)

    if not content:
        return []

    # 优先尝试 long TextGrid 格式
    entries = _parse_long_textgrid(
        content,
        tierName=tierName
    )

    # 如果 long 格式没有解析出结果，再尝试 short 格式
    if not entries:
        entries = _parse_short_textgrid(
            content,
            tierName=tierName
        )

    return entries


def write_csv(
    textgrid_list,
    filename=None,
    sep=",",
    header=True,
    save_gaps=False,
    meta=True
):
    """
    Writes a list of TextGrid Entry objects to a CSV file.

    如果 filename 为 None，则打印到标准输出。
    """
    columns = list(Entry._fields)

    f = None

    if filename:
        f = open(filename, "w", encoding="utf-8", newline="")

    try:
        if header:
            hline = sep.join(columns)

            if filename:
                f.write(hline + "\n")
            else:
                print(hline)

        for entry in textgrid_list:
            if not entry.name and not save_gaps:
                continue

            row = sep.join(
                str(x)
                for x in list(entry)
            )

            if filename:
                f.write(row + "\n")
            else:
                print(row)

    finally:
        if f is not None:
            f.flush()
            f.close()

    if filename and meta:
        with open(filename + ".meta", "w", encoding="utf-8") as metaf:
            metaf.write("---\nunits: s\ndatatype: 1002\n")


def textgrid2csv():
    """
    Command line tool:
        python textgrid.py input.TextGrid -o output.csv
    """
    parser = argparse.ArgumentParser(
        description="convert a TextGrid file to a CSV."
    )

    parser.add_argument(
        "TextGrid",
        default="F:\\schoolworks\\FAU\\ss26\\mt\\USC-annot-16\\sub009\\alignments\\bvt\\sub009_bvt_alignment.textgrid",
        help="a TextGrid file to process"
    )

    parser.add_argument(
        "-o",
        "--output",
        help="optional output file"
    )

    parser.add_argument(
        "--sep",
        help="separator to use in CSV output",
        default=","
    )

    parser.add_argument(
        "--noheader",
        help="no header for the CSV",
        action="store_false"
    )

    parser.add_argument(
        "--savegaps",
        help="preserves intervals with no label",
        action="store_true"
    )

    parser.add_argument(
        "--tier",
        help="tier name to read, default: phones",
        default="phones"
    )

    args = parser.parse_args()

    tgrid = read_textgrid(
        args.TextGrid,
        tierName=args.tier
    )

    write_csv(
        tgrid,
        filename=args.output,
        sep=args.sep,
        header=args.noheader,
        save_gaps=args.savegaps
    )


if __name__ == "__main__":
    textgrid2csv()