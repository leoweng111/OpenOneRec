"""视频推荐（Pretrain 阶段）数据处理。

作用：把每个用户"最近历史视频序列 + 下一批目标视频序列"转成一条 pretrain
    segments 样本（无 prompt、纯 SID 拼接）。目的是让 LLM 学会在给定
    history SID 之后自然续写 target SID —— 这是把"推荐"当作序列建模的核心 loss。

输入：
    metadata parquet    列: uid, split, hist_video_pid, target_video_pid
    pid2sid parquet     列: pid, sid（sid 是长度 3 的 int 数组，形如 [340, 6566, 5603]）

输出（一条样本）：
    source     'RecIF_VideoRec_Pretrain'
    uuid       随机
    segments   [{"type":"text","text": HIST_SIDS + TARGET_SIDS}]
                其中每个 SID 被格式化为
                '<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>'
                即 5 个 Itemic Token（两个特殊 + 三个 codebook token）。

流程：
    1) 读入 pid2sid → dict
    2) 只保留 split == 0（训练集）
    3) 每条 row：取最近 HIST_MAX_LEN 个 hist pid + 前 TARGET_MAX_LEN 个 target pid
       → 全部翻译成 SID 拼成一条长字符串
    4) 打包成 segments 格式的 dataframe，写到 train.parquet
下游：`data/scripts/split_data.py` 会把该 parquet 分片给 pretrain dataloader。
"""

import pandas as pd
import argparse
import json
import uuid
from pathlib import Path
from tqdm import tqdm

# ============== Configuration ==============
SID_FORMAT = '<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>'
HIST_MAX_LEN = 512
TARGET_MAX_LEN = 10


# ============== Core Functions ==============
def pids_to_sids(pids, pid2sid: dict) -> str:
    """Convert a list of pids to SID string."""
    if pids is None or (isinstance(pids, float) and pd.isna(pids)):
        return ""
    sids = []
    for pid in pids:
        if pid in pid2sid:
            code = pid2sid[pid]
            sid = SID_FORMAT.format(c0=code[0], c1=code[1], c2=code[2])
            sids.append(sid)
    return ''.join(sids)


def build_segments(hist_sids: str, target_sids: str) -> str:
    """Build segments format JSON string for pretrain."""
    text = f"{hist_sids}{target_sids}"
    segments = [{"type": "text", "text": text}]
    return json.dumps(segments, ensure_ascii=False)


def process_row(row, pid2sid: dict) -> dict:
    """Process a single row of data."""
    hist_pids = row['hist_video_pid']
    target_pids = row['target_video_pid']

    # Check data validity
    if hist_pids is None or (isinstance(hist_pids, float) and pd.isna(hist_pids)):
        return None
    if target_pids is None or (isinstance(target_pids, float) and pd.isna(target_pids)):
        return None

    # Truncate and convert to SID
    hist_sids = pids_to_sids(hist_pids[-HIST_MAX_LEN:], pid2sid)
    target_sids = pids_to_sids(target_pids[:TARGET_MAX_LEN], pid2sid)

    if not hist_sids or not target_sids:
        return None

    return {
        'source': 'RecIF_VideoRec_Pretrain',
        'uuid': str(uuid.uuid4()),
        'segments': build_segments(hist_sids, target_sids),
        'metadata': json.dumps({'uid': int(row['uid'])}, ensure_ascii=False)
    }


# ============== Main Function ==============
def main():
    parser = argparse.ArgumentParser(description="Video Recommendation Pretrain Data Processing")
    parser.add_argument('--input', type=str, required=True, help='Input metadata parquet path')
    parser.add_argument('--pid2sid', type=str, required=True, help='pid2sid mapping parquet path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load pid2sid mapping
    print(f"Loading pid2sid from {args.pid2sid}...")
    df_pid2sid = pd.read_parquet(args.pid2sid)
    pid2sid = dict(zip(df_pid2sid['pid'], df_pid2sid['sid']))
    print(f"  Loaded {len(pid2sid):,} mappings")

    # 2. Load metadata
    print(f"Loading metadata from {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df):,} rows")

    # 3. Process data (train only, split=0)
    print("Processing...")
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if row['split'] != 0:
            continue
        result = process_row(row, pid2sid)
        if result:
            results.append(result)

    # 4. Save results
    df_train = pd.DataFrame(results)
    train_path = output_dir / 'train.parquet'
    df_train.to_parquet(train_path, index=False)

    print(f"Saved: {train_path} ({len(df_train):,} rows)")
    print("Done!")


if __name__ == "__main__":
    main()
