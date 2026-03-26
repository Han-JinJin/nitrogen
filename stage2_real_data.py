"""
Stage 2: 从真实视频 + NitroGen 标注数据生成训练数据
读取视频帧 + parquet 标注 → 裁剪控制器区域 → 拼接帧对 → 保存为图片 + 标签

处理 VFR (可变帧率) 视频: 使用线性映射将 parquet 行索引映射到实际视频帧
内存优化: 逐帧读取视频, 即时保存裁剪结果, 不在内存中缓存全部帧
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from stage2_layout_config import BUTTON_NAMES, GRID_SIZE

# ========== 配置 ==========
SEGMENTS_DIR = r"D:\Python\videos\segments"
EXTRACTED_DIR = r"D:\Python\videos\extracted_data\extracted_data"
OUTPUT_DIR = r"D:\Python\videos\stage2_data"
IMG_H, IMG_W = 256, 512       # 两帧拼接: 256×256 × 2
FRAME_STEP = 6                 # 每隔 N 个标注行采样一次
PADDING_RATIO = 0.15           # bbox 外扩比例
PARQUET_FRAMES = 1200          # parquet 每 chunk 的标注行数 (60fps × 20s)

# 验证集划分: 每个视频的最后 VAL_CHUNKS_PER_VIDEO 个 chunk 作为验证集
VAL_CHUNKS_PER_VIDEO = 2

# 仅使用有 actions_processed.parquet 的视频
USABLE_VIDEOS = ["_R_RT_z0srE", "07TXwbcRdvM", "5gl7G68tAgs", "6j5X8VAMwUs"]


def _pos_to_grid_idx(x, y, grid_size=GRID_SIZE):
    """连续坐标 [-1, 1] → 展平网格索引 [0, grid_size²-1]"""
    col = int(np.clip(np.round((x + 1) / 2 * (grid_size - 1)), 0, grid_size - 1))
    row = int(np.clip(np.round((y + 1) / 2 * (grid_size - 1)), 0, grid_size - 1))
    return row * grid_size + col


def find_segment_for_chunk(video_id, chunk_idx):
    """chunk_idx (0-indexed) → 对应的 segment 视频文件路径"""
    seg_dir = Path(SEGMENTS_DIR) / video_id / "segments"
    if not seg_dir.is_dir():
        seg_dir = Path(SEGMENTS_DIR) / video_id

    seg_num = chunk_idx + 1  # segments 从 1 开始编号
    pattern = f"*_seg{seg_num:04d}_*.mp4"
    matches = list(seg_dir.glob(pattern))
    return str(matches[0]) if matches else None


def crop_controller_region(frame, bbox, padding=PADDING_RATIO):
    """从帧中裁剪控制器叠加层区域 (带 padding)"""
    h, w = frame.shape[:2]
    bx, by, bw, bh = bbox

    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    x1 = max(0, bx - pad_x)
    y1 = max(0, by - pad_y)
    x2 = min(w, bx + bw + pad_x)
    y2 = min(h, by + bh + pad_y)

    return frame[y1:y2, x1:x2]


def process_chunk(video_id, chunk_dir, chunk_idx, img_dir, counter):
    """处理一个 chunk, 即时保存图片, 返回 labels 列表和更新后的 counter

    内存优化: 逐帧读取视频, 只保留当前帧和上一帧
    """
    # 读取 metadata
    meta_path = os.path.join(chunk_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    bbox = meta["bbox_controller_overlay"]  # [x, y, w, h]

    # 读取 parquet
    parquet_path = os.path.join(chunk_dir, "actions_processed.parquet")
    if not os.path.exists(parquet_path):
        return [], counter
    df = pd.read_parquet(parquet_path)
    n_parquet = len(df)

    # 找到对应视频文件
    video_path = find_segment_for_chunk(video_id, chunk_idx)
    if video_path is None:
        print(f"  Warning: segment not found for {video_id} chunk {chunk_idx}")
        return [], counter

    # 获取视频帧数 (不读入全部帧)
    cap = cv2.VideoCapture(video_path)
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_video < 2:
        cap.release()
        return [], counter

    # 预计算需要的采样点: ann_t = FRAME_STEP, 2*FRAME_STEP, ...
    # 以及每个采样点需要的视频帧索引
    sample_points = []
    for ann_t in range(FRAME_STEP, n_parquet, FRAME_STEP):
        vid_t = round(ann_t * (n_video - 1) / (PARQUET_FRAMES - 1))
        vid_prev = round((ann_t - 1) * (n_video - 1) / (PARQUET_FRAMES - 1))
        vid_t = min(vid_t, n_video - 1)
        vid_prev = min(vid_prev, n_video - 1)
        if vid_t < n_video:
            sample_points.append((ann_t, vid_prev, vid_t))

    if not sample_points:
        cap.release()
        return [], counter

    # 收集所有需要的视频帧索引
    needed_frames = set()
    for _, vp, vt in sample_points:
        needed_frames.add(vp)
        needed_frames.add(vt)

    # 顺序读取视频, 只保留需要的帧
    frame_cache = {}
    frame_idx = 0
    max_needed = max(needed_frames)

    while frame_idx <= max_needed:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in needed_frames:
            # 立即裁剪并缩放, 减少内存占用
            crop = crop_controller_region(frame, bbox)
            if crop.size > 0:
                crop = cv2.resize(crop, (IMG_W // 2, IMG_H))
                frame_cache[frame_idx] = crop
        frame_idx += 1

    cap.release()

    # 生成帧对并保存
    labels = []
    for ann_t, vid_prev, vid_t in sample_points:
        if vid_prev not in frame_cache or vid_t not in frame_cache:
            continue

        crop_prev = frame_cache[vid_prev]
        crop_curr = frame_cache[vid_t]

        # 横向拼接: (256, 512, 3) — [prev | curr]
        combined = np.concatenate([crop_prev, crop_curr], axis=1)

        # 保存图片
        fname = f"{counter:06d}.jpg"
        cv2.imwrite(
            os.path.join(img_dir, fname), combined,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )

        # 读取标签
        row = df.iloc[ann_t]
        buttons = [int(row[btn]) for btn in BUTTON_NAMES]
        j_left = np.array(row["j_left"], dtype=np.float32)
        j_right = np.array(row["j_right"], dtype=np.float32)

        left_idx = _pos_to_grid_idx(float(j_left[0]), float(j_left[1]))
        right_idx = _pos_to_grid_idx(float(j_right[0]), float(j_right[1]))

        labels.append({
            "buttons": buttons,
            "left_stick": [float(j_left[0]), float(j_left[1])],
            "right_stick": [float(j_right[0]), float(j_right[1])],
            "left_grid_idx": left_idx,
            "right_grid_idx": right_idx,
        })
        counter += 1

    # 释放帧缓存
    frame_cache.clear()

    return labels, counter


def save_labels(labels, split_dir):
    """保存 npz 标签文件"""
    left_idx = np.array([l["left_grid_idx"] for l in labels], dtype=np.int64)
    right_idx = np.array([l["right_grid_idx"] for l in labels], dtype=np.int64)
    btns = np.array([l["buttons"] for l in labels], dtype=np.int64)
    left_stick = np.array([l["left_stick"] for l in labels], dtype=np.float32)
    right_stick = np.array([l["right_stick"] for l in labels], dtype=np.float32)

    np.savez_compressed(
        os.path.join(split_dir, "labels.npz"),
        left_grid_idx=left_idx,
        right_grid_idx=right_idx,
        buttons=btns,
        left_stick=left_stick,
        right_stick=right_stick,
    )


def main():
    print("=" * 60)
    print("Stage 2: Real Data Preparation")
    print(f"  Videos:       {USABLE_VIDEOS}")
    print(f"  Val chunks:   last {VAL_CHUNKS_PER_VIDEO} per video")
    print(f"  Step:         every {FRAME_STEP} annotations")
    print("=" * 60)

    # 准备输出目录
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)

    train_labels = []
    val_labels = []
    train_counter = 0
    val_counter = 0

    for video_id in USABLE_VIDEOS:
        video_extracted = os.path.join(EXTRACTED_DIR, video_id)
        if not os.path.isdir(video_extracted):
            print(f"\n[SKIP] {video_id}: extracted_data not found")
            continue

        chunks = sorted(os.listdir(video_extracted))
        n_chunks = len(chunks)
        # 每个视频的最后 VAL_CHUNKS_PER_VIDEO 个 chunk 作为验证集
        val_start = max(0, n_chunks - VAL_CHUNKS_PER_VIDEO)

        print(f"\n{video_id}: {n_chunks} chunks (train: 0-{val_start-1}, val: {val_start}-{n_chunks-1})")

        for ci, chunk_name in enumerate(tqdm(chunks, desc=f"  {video_id}")):
            chunk_dir = os.path.join(video_extracted, chunk_name)
            if not os.path.isdir(chunk_dir):
                continue

            is_val = ci >= val_start
            split = "val" if is_val else "train"
            img_dir = os.path.join(OUTPUT_DIR, split, "images")
            counter = val_counter if is_val else train_counter

            chunk_idx = int(chunk_name.split("_chunk_")[-1])
            labels, counter = process_chunk(
                video_id, chunk_dir, chunk_idx, img_dir, counter
            )

            if is_val:
                val_labels.extend(labels)
                val_counter = counter
            else:
                train_labels.extend(labels)
                train_counter = counter

    # 保存标签
    print(f"\n{'=' * 60}")
    print(f"Train: {len(train_labels)} samples")
    print(f"Val:   {len(val_labels)} samples")
    print(f"Total: {len(train_labels) + len(val_labels)} samples")

    if train_labels:
        save_labels(train_labels, os.path.join(OUTPUT_DIR, "train"))
        print(f"  Saved train labels.npz")

    if val_labels:
        save_labels(val_labels, os.path.join(OUTPUT_DIR, "val"))
        print(f"  Saved val labels.npz")

    # 可视化样本 (从已保存的文件中读取)
    print(f"\n[Saving visualization samples...]")
    vis_dir = os.path.join(OUTPUT_DIR, "vis_samples")
    os.makedirs(vis_dir, exist_ok=True)

    all_labels = train_labels + val_labels
    n_vis = min(20, len(all_labels))
    indices = np.random.choice(len(all_labels), n_vis, replace=False)

    for j, idx in enumerate(indices):
        lbl = all_labels[idx]
        # 确定图片路径
        if idx < len(train_labels):
            img_path = os.path.join(OUTPUT_DIR, "train", "images", f"{idx:06d}.jpg")
        else:
            val_idx = idx - len(train_labels)
            img_path = os.path.join(OUTPUT_DIR, "val", "images", f"{val_idx:06d}.jpg")

        img = cv2.imread(img_path)
        if img is None:
            continue

        info = (f"L=({lbl['left_stick'][0]:.2f},{lbl['left_stick'][1]:.2f}) "
                f"R=({lbl['right_stick'][0]:.2f},{lbl['right_stick'][1]:.2f}) "
                f"Btn={sum(lbl['buttons'])}")
        cv2.putText(img, info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(vis_dir, f"sample_{j:02d}.jpg"), img)

    print(f"  Saved {n_vis} visualization samples → {vis_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
