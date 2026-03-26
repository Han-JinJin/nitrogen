"""
Stage 2: 推理演示脚本
在真实视频片段上运行模型，对比 ground truth
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import cv2
import json
import numpy as np
import torch
import pandas as pd
from pathlib import Path

from stage2_model import SegFormerActionParser, grid_to_position, NUM_BUTTONS
from stage2_layout_config import BUTTON_NAMES

# ========== 配置 ==========
CHECKPOINT = r"D:\Python\videos\results\stage2_finetune\best.pt"
SEGMENTS_DIR = r"D:\Python\videos\segments"
EXTRACTED_DIR = r"D:\Python\videos\extracted_data\extracted_data"
OUTPUT_DIR = r"D:\Python\videos\results\stage2_demo"

# 使用训练数据中的视频做演示 (chunk 0 是各视频的第一个片段)
DEMO_VIDEOS = ["_R_RT_z0srE", "07TXwbcRdvM", "5gl7G68tAgs", "6j5X8VAMwUs"]
DEMO_CHUNK = 0  # 演示用第一个 chunk

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FRAME_STEP = 6
PARQUET_FRAMES = 1200


def load_model(ckpt_path, device):
    model = SegFormerActionParser().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess_pair(crop_prev, crop_curr):
    """两帧裁剪 -> 模型输入 tensor"""
    prev = cv2.resize(crop_prev, (256, 256))
    curr = cv2.resize(crop_curr, (256, 256))
    combined = np.concatenate([prev, curr], axis=1)  # (256, 512, 3)
    combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    combined = (combined - MEAN) / STD
    return torch.from_numpy(combined).permute(2, 0, 1).unsqueeze(0)


def find_segment_video(video_id, chunk_idx):
    """chunk_idx (0-indexed) → 对应的 segment 视频文件路径"""
    seg_dir = Path(SEGMENTS_DIR) / video_id / "segments"
    if not seg_dir.is_dir():
        seg_dir = Path(SEGMENTS_DIR) / video_id
    seg_num = chunk_idx + 1  # segments 从 1 开始编号
    pattern = f"*_seg{seg_num:04d}_*.mp4"
    matches = list(seg_dir.glob(pattern))
    return str(matches[0]) if matches else None


def run_demo(video_id, chunk_idx, model, device):
    """在单个视频片段上运行推理并对比 ground truth"""
    # 1. 找到标注
    chunk_name = f"{video_id}_chunk_{chunk_idx:04d}"
    chunk_dir = Path(EXTRACTED_DIR) / video_id / chunk_name
    parquet_path = chunk_dir / "actions_processed.parquet"
    meta_path = chunk_dir / "metadata.json"

    if not parquet_path.exists() or not meta_path.exists():
        print(f"    Skipped: annotations not found at {chunk_dir}")
        return None

    # 找视频片段
    video_path = find_segment_video(video_id, chunk_idx)
    if video_path is None or not Path(video_path).exists():
        print(f"    Skipped: video segment not found for {video_id} chunk {chunk_idx}")
        return None

    # 2. 加载标注
    with open(meta_path) as f:
        meta = json.load(f)
    bbox = meta["bbox_controller_overlay"]  # [x, y, w, h]
    bx, by, bw, bh = bbox

    df = pd.read_parquet(parquet_path)

    # 3. 读取视频帧
    cap = cv2.VideoCapture(str(video_path))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # padding
    pad_x = int(bw * 0.15)
    pad_y = int(bh * 0.15)
    x1 = max(0, bx - pad_x)
    y1 = max(0, by - pad_y)
    x2 = bx + bw + pad_x
    y2 = by + bh + pad_y

    # 4. 推理
    results = []
    sample_indices = list(range(FRAME_STEP, PARQUET_FRAMES, FRAME_STEP))

    # 预读帧 (用 VFR mapping)
    needed_vid_frames = set()
    for ann_t in sample_indices:
        vid_t = round(ann_t * (n_video - 1) / (PARQUET_FRAMES - 1))
        vid_t_prev = round((ann_t - 1) * (n_video - 1) / (PARQUET_FRAMES - 1))
        needed_vid_frames.add(vid_t)
        needed_vid_frames.add(vid_t_prev)

    frame_cache = {}
    for fid in range(n_video):
        ret, frame = cap.read()
        if not ret:
            break
        if fid in needed_vid_frames:
            # 裁剪
            h, w = frame.shape[:2]
            cx1 = max(0, x1)
            cy1 = max(0, y1)
            cx2 = min(w, x2)
            cy2 = min(h, y2)
            frame_cache[fid] = frame[cy1:cy2, cx1:cx2].copy()
    cap.release()

    for ann_t in sample_indices:
        if ann_t >= len(df):
            break

        vid_t = round(ann_t * (n_video - 1) / (PARQUET_FRAMES - 1))
        vid_t_prev = round((ann_t - 1) * (n_video - 1) / (PARQUET_FRAMES - 1))

        if vid_t not in frame_cache or vid_t_prev not in frame_cache:
            continue

        # 模型推理
        inp = preprocess_pair(frame_cache[vid_t_prev], frame_cache[vid_t])
        inp = inp.to(device)

        with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch.float16):
            pred = model(inp)

        lx, ly = grid_to_position(pred["seg_left"])
        rx, ry = grid_to_position(pred["seg_right"])
        btn_probs = pred["buttons"].sigmoid().squeeze(0)
        btn_states = (btn_probs > 0.5).int().cpu().tolist()

        # Ground truth
        row = df.iloc[ann_t]
        gt_btns = [int(row[name]) for name in BUTTON_NAMES]
        gt_jl = row["j_left"]
        gt_jr = row["j_right"]

        results.append({
            "ann_t": ann_t,
            "pred_left":  [round(lx.item(), 4), round(ly.item(), 4)],
            "pred_right": [round(rx.item(), 4), round(ry.item(), 4)],
            "pred_btns": btn_states,
            "gt_left":  [round(float(gt_jl[0]), 4), round(float(gt_jl[1]), 4)],
            "gt_right": [round(float(gt_jr[0]), 4), round(float(gt_jr[1]), 4)],
            "gt_btns": gt_btns,
        })

    return results


def compute_metrics(results):
    """计算推理结果的指标"""
    if not results:
        return {}

    pred_lx = np.array([r["pred_left"][0] for r in results])
    pred_ly = np.array([r["pred_left"][1] for r in results])
    gt_lx   = np.array([r["gt_left"][0] for r in results])
    gt_ly   = np.array([r["gt_left"][1] for r in results])

    pred_rx = np.array([r["pred_right"][0] for r in results])
    pred_ry = np.array([r["pred_right"][1] for r in results])
    gt_rx   = np.array([r["gt_right"][0] for r in results])
    gt_ry   = np.array([r["gt_right"][1] for r in results])

    def r2(pred, gt):
        ss_res = ((pred - gt) ** 2).sum()
        ss_tot = ((gt - gt.mean()) ** 2).sum()
        if ss_tot < 1e-6:
            return 0.0
        return 1 - ss_res / ss_tot

    def mae(pred, gt):
        return np.abs(pred - gt).mean()

    # 按键
    pred_all_btns = np.array([r["pred_btns"] for r in results])
    gt_all_btns   = np.array([r["gt_btns"] for r in results])
    btn_acc = (pred_all_btns == gt_all_btns).mean()

    # 按下的按键的 precision/recall
    gt_pressed = gt_all_btns.sum()
    pred_pressed = pred_all_btns.sum()
    tp = ((pred_all_btns == 1) & (gt_all_btns == 1)).sum()
    precision = tp / pred_pressed if pred_pressed > 0 else 0
    recall = tp / gt_pressed if gt_pressed > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "n_samples": len(results),
        "R2_Lx": round(r2(pred_lx, gt_lx), 4),
        "R2_Ly": round(r2(pred_ly, gt_ly), 4),
        "R2_Rx": round(r2(pred_rx, gt_rx), 4),
        "R2_Ry": round(r2(pred_ry, gt_ry), 4),
        "MAE_Lx": round(mae(pred_lx, gt_lx), 4),
        "MAE_Ly": round(mae(pred_ly, gt_ly), 4),
        "MAE_Rx": round(mae(pred_rx, gt_rx), 4),
        "MAE_Ry": round(mae(pred_ry, gt_ry), 4),
        "btn_acc": round(float(btn_acc), 4),
        "btn_F1": round(float(f1), 4),
        "btn_precision": round(float(precision), 4),
        "btn_recall": round(float(recall), 4),
    }


def visualize_comparison(results, video_id, output_dir):
    """生成预测 vs GT 对比可视化图"""
    if not results or len(results) < 10:
        return

    n = min(50, len(results))
    img_h, img_w = 400, 800
    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

    # 左摇杆 X 轴对比
    pred_lx = [r["pred_left"][0] for r in results[:n]]
    gt_lx   = [r["gt_left"][0] for r in results[:n]]

    for i in range(n):
        x = int(20 + i * (img_w - 40) / n)
        # GT: 绿色
        gy = int(100 - gt_lx[i] * 80)
        cv2.circle(img, (x, gy), 3, (0, 180, 0), -1)
        # Pred: 红色
        py = int(100 - pred_lx[i] * 80)
        cv2.circle(img, (x, py), 3, (0, 0, 220), -1)

    cv2.putText(img, "Left X: Green=GT, Red=Pred", (20, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.line(img, (20, 100), (img_w - 20, 100), (200, 200, 200), 1)

    # 左摇杆 Y 轴对比
    pred_ly = [r["pred_left"][1] for r in results[:n]]
    gt_ly   = [r["gt_left"][1] for r in results[:n]]

    for i in range(n):
        x = int(20 + i * (img_w - 40) / n)
        gy = int(280 - gt_ly[i] * 80)
        cv2.circle(img, (x, gy), 3, (0, 180, 0), -1)
        py = int(280 - pred_ly[i] * 80)
        cv2.circle(img, (x, py), 3, (0, 0, 220), -1)

    cv2.putText(img, "Left Y: Green=GT, Red=Pred", (20, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.line(img, (20, 280), (img_w - 20, 280), (200, 200, 200), 1)

    out_path = os.path.join(output_dir, f"{video_id}_comparison.png")
    cv2.imwrite(out_path, img)
    print(f"    Saved: {out_path}")


def main():
    print("=" * 60)
    print("Stage 2: Inference Demo (vs Ground Truth)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. 加载模型
    print(f"\n[1] Loading model: {CHECKPOINT}")
    if not os.path.exists(CHECKPOINT):
        print("ERROR: Checkpoint not found!")
        return
    model = load_model(CHECKPOINT, device)
    print("  Model loaded.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. 逐视频推理
    print("\n[2] Running inference...")
    all_metrics = {}

    for video_id in DEMO_VIDEOS:
        chunk = DEMO_CHUNK
        print(f"\n  Video: {video_id} (chunk={chunk})")

        results = run_demo(video_id, chunk, model, device)
        if results is None:
            continue

        metrics = compute_metrics(results)
        all_metrics[video_id] = metrics

        print(f"    Samples: {metrics['n_samples']}")
        print(f"    R2:  Lx={metrics['R2_Lx']:.3f}  Ly={metrics['R2_Ly']:.3f}  "
              f"Rx={metrics['R2_Rx']:.3f}  Ry={metrics['R2_Ry']:.3f}")
        print(f"    MAE: Lx={metrics['MAE_Lx']:.3f}  Ly={metrics['MAE_Ly']:.3f}  "
              f"Rx={metrics['MAE_Rx']:.3f}  Ry={metrics['MAE_Ry']:.3f}")
        print(f"    Btn: acc={metrics['btn_acc']:.3f}  F1={metrics['btn_F1']:.3f}  "
              f"P={metrics['btn_precision']:.3f}  R={metrics['btn_recall']:.3f}")

        # 保存详细结果
        detail_path = os.path.join(OUTPUT_DIR, f"{video_id}_detail.json")
        with open(detail_path, "w") as f:
            json.dump({"metrics": metrics, "predictions": results[:20]}, f, indent=2)

        # 可视化对比
        visualize_comparison(results, video_id, OUTPUT_DIR)

    # 3. 汇总
    print("\n" + "=" * 60)
    print("Summary:")
    for vid, m in all_metrics.items():
        r2_avg = (m["R2_Lx"] + m["R2_Ly"] + m["R2_Rx"] + m["R2_Ry"]) / 4
        print(f"  {vid}: R2_avg={r2_avg:.3f}  btn_F1={m['btn_F1']:.3f}  "
              f"MAE_L=({m['MAE_Lx']:.3f},{m['MAE_Ly']:.3f})")

    summary_path = os.path.join(OUTPUT_DIR, "demo_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
