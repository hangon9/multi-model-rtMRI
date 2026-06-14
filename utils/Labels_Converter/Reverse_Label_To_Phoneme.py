# -*- coding: utf-8 -*-
"""
将已经转换好的 label.csv 反向映射为音素（phoneme）。

思路借鉴 Get_Phonemic_Labels.py：
1. 原脚本是“音素 -> 特征标签”；
2. 本脚本做“特征标签 -> 音素候选”；
3. 由于 Phonemic_Table 中多个音素可能共享同一组特征，因此反向映射通常不是一一对应；
4. 默认输出一个“主音素” (按表中出现顺序取第一个) ，同时保留所有候选音素，便于后续人工检查或二次规则筛选。

用法示例：
python Reverse_Label_To_Phoneme.py \
    --label_csv sub009_bvt_label.csv \
    --phoneme_table Phonemic_Table.csv \
    --output_csv sub009_bvt_phoneme.csv

可选参数：
--mode first   : 输出第一个候选音素（默认）
--mode concat  : 将所有候选音素用“/”拼接到 Phoneme 列
--mode strict  : 若存在多个候选，则 Phoneme 列写为 AMBIGUOUS

批量处理目录示例：
python Reverse_Label_To_Phoneme.py \
    --input_dir labels_annot_16 \
    --output_dir phoneme_output \
    --phoneme_table Phonemic_Table.csv

会递归查找 input_dir 下所有 csv 文件，并将结果保存到 output_dir 中对应的相对路径位置。
"""

import argparse
from pathlib import Path
import pandas as pd


def load_table(table_path: str) -> pd.DataFrame:
    """读取音素表，支持 .csv / .xlsx。"""
    path = Path(table_path)
    suffix = path.suffix.lower()

    if suffix == '.csv':
        df = pd.read_csv(path)
    elif suffix in ['.xlsx', '.xls']:
        if suffix == '.xlsx':
            df = pd.read_excel(path, engine='openpyxl')
        else:
            df = pd.read_excel(path, engine='xlrd')
    else:
        raise ValueError(f'不支持的音素表格式: {suffix}')

    # 清理空列 / 空行 / 多余列名
    df.columns = [str(c).strip() for c in df.columns]
    unnamed_cols = [c for c in df.columns if c.startswith('Unnamed') or c == '']
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols, errors='ignore')
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')

    # 自动识别第一列是否为音素列
    first_col = df.columns[0]
    if first_col.lower() not in ['phoneme', 'phone']:
        df = df.rename(columns={first_col: 'Phoneme'})
    else:
        df = df.rename(columns={first_col: 'Phoneme'})

    df['Phoneme'] = df['Phoneme'].astype(str).str.strip().str.upper()

    feature_cols = [c for c in df.columns if c != 'Phoneme']
    df[feature_cols] = df[feature_cols].fillna(0).astype(int)
    return df[['Phoneme'] + feature_cols]


def load_label_csv(label_csv: str) -> pd.DataFrame:
    """读取 label.csv，并清理空列。"""
    df = pd.read_csv(label_csv)
    df.columns = [str(c).strip() for c in df.columns]
    unnamed_cols = [c for c in df.columns if c.startswith('Unnamed') or c == '']
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols, errors='ignore')
    df = df.dropna(axis=1, how='all')
    return df


def build_reverse_map(phoneme_table: pd.DataFrame, feature_cols: list) -> dict:
    """
    构建 {特征元组: [音素1, 音素2, ...]} 的反向映射。
    注意：这是多对一映射反转后形成的一对多关系。
    """
    reverse_map = {}
    for _, row in phoneme_table.iterrows():
        key = tuple(int(row[c]) for c in feature_cols)
        reverse_map.setdefault(key, []).append(row['Phoneme'])
    return reverse_map


def select_phoneme(candidates, mode='first'):
    """根据输出模式选择主音素。"""
    if not candidates:
        return 'UNK'
    if mode == 'first':
        return candidates[0]
    elif mode == 'concat':
        return '/'.join(candidates)
    elif mode == 'strict':
        return candidates[0] if len(candidates) == 1 else 'AMBIGUOUS'
    else:
        raise ValueError(f'不支持的 mode: {mode}')


def build_output_path(input_file: Path, input_root: Path, output_root: Path) -> Path:
    """根据输入文件相对路径构建输出文件路径，并把后缀改为 _phoneme.csv。"""
    relative_path = input_file.resolve().relative_to(input_root.resolve())
    relative_output = output_root.joinpath(relative_path)
    return relative_output.with_name(relative_output.stem + '_phoneme.csv')


def is_within(path: Path, potential_parent: Path) -> bool:
    """判断 path 是否位于 potential_parent 目录下。"""
    try:
        path.resolve().relative_to(potential_parent.resolve())
        return True
    except ValueError:
        return False


def reverse_label_to_phoneme(label_csv: str,
                             phoneme_table_path: str,
                             output_csv: str = None,
                             mode: str = 'first') -> pd.DataFrame:
    # 读取数据
    table = load_table(phoneme_table_path)
    labels = load_label_csv(label_csv)

    # 第一列默认为帧名列，后续列与表中的特征列对齐
    frame_col = labels.columns[0]
    label_feature_cols = list(labels.columns[1:])
    table_feature_cols = list(table.columns[1:])

    # 检查列是否一致（顺序也要一致，避免元组错位）
    if label_feature_cols != table_feature_cols:
        missing_in_label = [c for c in table_feature_cols if c not in label_feature_cols]
        extra_in_label = [c for c in label_feature_cols if c not in table_feature_cols]
        raise ValueError(
            'label.csv 的特征列与音素表不一致。\n'
            f'音素表特征列: {table_feature_cols}\n'
            f'label 特征列: {label_feature_cols}\n'
            f'label 缺少列: {missing_in_label}\n'
            f'label 多出列: {extra_in_label}'
        )

    # 构建反向映射
    reverse_map = build_reverse_map(table, table_feature_cols)

    # 逐行映射
    phonemes = []
    candidates_col = []
    num_candidates = []
    status_col = []

    for _, row in labels.iterrows():
        key = tuple(int(row[c]) for c in table_feature_cols)
        candidates = reverse_map.get(key, [])

        phoneme = select_phoneme(candidates, mode=mode)
        phonemes.append(phoneme)
        candidates_col.append('/'.join(candidates) if candidates else '')
        num_candidates.append(len(candidates))

        if len(candidates) == 0:
            status_col.append('NO_MATCH')
        elif len(candidates) == 1:
            status_col.append('UNIQUE')
        else:
            status_col.append('AMBIGUOUS')

    # 输出“与 label.csv 相似”的逐帧表格
    out_df = pd.DataFrame({
        frame_col: labels[frame_col],
        'Phoneme': phonemes,
        'Candidates': candidates_col,
        'NumCandidates': num_candidates,
        'Status': status_col,
    })

    if output_csv is None:
        output_csv = str(Path(label_csv).with_name(Path(label_csv).stem + '_phoneme.csv'))

    out_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    return out_df


def reverse_label_folder(input_dir: str,
                         output_dir: str,
                         phoneme_table_path: str,
                         mode: str = 'first') -> pd.DataFrame:
    """递归处理输入目录下的所有 csv 文件，并将输出保存到目标目录。"""
    input_root = Path(input_dir)
    output_root = Path(output_dir)

    if not input_root.is_dir():
        raise FileNotFoundError(f'输入目录不存在或不是目录: {input_root}')
    if input_root.resolve() == output_root.resolve():
        raise ValueError('input_dir 和 output_dir 不能相同。')

    output_root.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_root.rglob('*.csv'))
    results = []

    for csv_file in csv_files:
        if is_within(csv_file, output_root):
            continue

        output_path = build_output_path(csv_file, input_root, output_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            out_df = reverse_label_to_phoneme(
                label_csv=str(csv_file),
                phoneme_table_path=phoneme_table_path,
                output_csv=str(output_path),
                mode=mode,
            )
            results.append({
                'InputFile': str(csv_file),
                'OutputFile': str(output_path),
                'Status': 'OK',
                'Rows': len(out_df),
                'Error': '',
            })
        except Exception as exc:
            results.append({
                'InputFile': str(csv_file),
                'OutputFile': str(output_path),
                'Status': 'FAILED',
                'Rows': 0,
                'Error': str(exc),
            })
            print(f'跳过 {csv_file}: {exc}')

    summary_df = pd.DataFrame(results)
    summary_path = output_root / 'batch_reverse_summary.csv'
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    return summary_df


def main():
    parser = argparse.ArgumentParser(description='将 label.csv 反向映射为音素表。')
    parser.add_argument('--label_csv', default=None, help='输入的 label.csv 文件路径')
    parser.add_argument('--input_dir', default=r'datasets/labels_TIMIT', help='递归处理的输入目录路径')
    parser.add_argument('--output_dir', default=r'datasets/phoneme_TIMIT', help='批量处理时的输出目录路径')
    parser.add_argument('--phoneme_table', default=r'utils/TIMIT_MRI_Get_Phone_Alignment/Phonemic_Table.xlsx', help='音素表路径（csv/xlsx/xls）')
    parser.add_argument('--output_csv', default=None, help='输出 csv 文件路径')
    parser.add_argument('--mode', choices=['first', 'concat', 'strict'], default='first',
                        help='主音素输出策略：first/concat/strict')
    args = parser.parse_args()

    if args.input_dir and args.label_csv:
        raise ValueError('label_csv 和 input_dir 只能选择一种模式。')

    if args.input_dir:
        if not args.output_dir:
            raise ValueError('批量处理目录时必须提供 --output_dir。')

        summary_df = reverse_label_folder(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            phoneme_table_path=args.phoneme_table,
            mode=args.mode,
        )

        print('批量转换完成。')
        print(f'处理文件数: {len(summary_df)}')
        print('状态统计:')
        print(summary_df['Status'].value_counts(dropna=False).to_string())
        print(f'汇总文件: {Path(args.output_dir) / "batch_reverse_summary.csv"}')
        return

    if not args.label_csv:
        raise ValueError('必须提供 --label_csv 或 --input_dir。')

    out_df = reverse_label_to_phoneme(
        label_csv=args.label_csv,
        phoneme_table_path=args.phoneme_table,
        output_csv=args.output_csv,
        mode=args.mode,
    )

    print('转换完成。')
    print(f'输出行数: {len(out_df)}')
    print('状态统计:')
    print(out_df['Status'].value_counts(dropna=False).to_string())
    print(f'输出文件: {args.output_csv if args.output_csv else Path(args.label_csv).with_name(Path(args.label_csv).stem + "_phoneme.csv")}')


if __name__ == '__main__':
    main()
