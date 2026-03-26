"""
Stage 2: 合成训练数据生成器
在控制器模板上渲染随机摇杆位置和按键状态，合成到游戏帧背景上
生成 SegFormer 训练所需的 (图像, 标签) 对
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import json
import glob
import random
from pathlib import Path
from tqdm import tqdm

from stage2_layout_config import (
    CONTROLLER_FAMILIES, BUTTON_NAMES, NUM_BUTTONS, GRID_SIZE,
    get_all_template_families
)
def _pos_to_grid_idx_np(x, y, grid_size=GRID_SIZE):
    """numpy 版 position_to_grid_idx (避免 import torch)"""
    col = int(np.clip(np.round((x + 1) / 2 * (grid_size - 1)), 0, grid_size - 1))
    row = int(np.clip(np.round((y + 1) / 2 * (grid_size - 1)), 0, grid_size - 1))
    return row * grid_size + col

# ========== 配置 ==========
TEMPLATE_DIR = r"D:\Python\videos\shoubing\overlay_templates"
SEGMENTS_DIR = r"D:\Python\videos\segments"
VIDEO_DIR = r"D:\Python\videos"
OUTPUT_DIR = r"D:\Python\videos\stage2_data_synthetic"
NUM_TRAIN = 8000
NUM_VAL = 1000
IMG_H, IMG_W = 256, 512  # 两帧拼接: 256x256 × 2
FRAME_SIZE = (640, 360)  # 视频帧尺寸
NUM_BG_FRAMES = 500  # 从视频中提取的背景帧数


def extract_background_frames(segments_dir, num_frames=NUM_BG_FRAMES):
    """从 segments/1~11 目录中提取多样化的背景帧"""
    clips = []
    for sub in sorted(Path(segments_dir).iterdir()):
        if sub.is_dir():
            # 支持 sub/segments/*.mp4 嵌套结构
            nested = sub / "segments"
            search_dir = nested if nested.is_dir() else sub
            clips.extend(sorted(search_dir.glob("*.mp4")))
    if not clips:
        print("  Warning: No video clips found, using random backgrounds")
        return []

    frames_per_clip = max(1, num_frames // len(clips))
    backgrounds = []

    for clip in clips:
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            continue
        indices = np.linspace(0, total - 1, frames_per_clip, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                backgrounds.append(frame)
        cap.release()
        if len(backgrounds) >= num_frames:
            break

    print(f"  Extracted {len(backgrounds)} background frames")
    return backgrounds


def load_overlay_templates(template_dir):
    """加载高清叠加层模板 (保留 alpha 通道)"""
    family_map = get_all_template_families()
    templates = []

    for ext in ["*.png", "*.jpg"]:
        for path in glob.glob(os.path.join(template_dir, ext)):
            name = Path(path).stem
            family_name = family_map.get(name)
            if family_name is None:
                continue

            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            # 确保有 alpha 通道
            if len(img.shape) == 2:
                continue
            if img.shape[2] == 3:
                # 对没有 alpha 的图片 (如裁剪模板)，基于黑色背景生成 alpha
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, alpha = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
                img = np.concatenate([img, alpha[:, :, None]], axis=2)

            templates.append({
                "name": name,
                "family": family_name,
                "image": img,  # BGRA
            })

    print(f"  Loaded {len(templates)} overlay templates")
    return templates


def render_joystick(img_bgra, center_px, radius_px, stick_xy):
    """在模板上渲染摇杆指示器 (亮色小圆点)"""
    cx, cy = center_px
    dx = int(stick_xy[0] * radius_px)
    dy = int(stick_xy[1] * radius_px)
    dot_x = cx + dx
    dot_y = cy + dy
    dot_r = max(2, int(radius_px * 0.25))

    # 绘制摇杆点 (白色/亮色半透明圆)
    color = random.choice([
        (255, 255, 255, 220),  # 白色
        (0, 255, 255, 220),    # 青色
        (180, 255, 180, 220),  # 浅绿
        (255, 255, 200, 220),  # 浅黄
    ])
    cv2.circle(img_bgra, (dot_x, dot_y), dot_r, color, -1, cv2.LINE_AA)
    # 外圈
    cv2.circle(img_bgra, (dot_x, dot_y), dot_r + 1,
               (color[0], color[1], color[2], 150), 1, cv2.LINE_AA)


def render_button_press(img_bgra, center_px, radius_px):
    """在模板上渲染按键高亮 (模拟按下状态) — 使用简单 addWeighted"""
    cx, cy = center_px
    r = int(radius_px)
    # 直接在按键区域增亮
    color_bgr = random.choice([(255, 255, 255), (200, 200, 255),
                                (200, 255, 200), (255, 200, 200)])
    cv2.circle(img_bgra, (cx, cy), r, (*color_bgr, 200), -1, cv2.LINE_AA)


def composite_overlay(background, overlay_bgra, x, y, opacity=1.0):
    """将 BGRA 叠加层合成到背景上"""
    oh, ow = overlay_bgra.shape[:2]
    bh, bw = background.shape[:2]

    # 裁剪到帧范围内
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bw, x + ow)
    y2 = min(bh, y + oh)
    if x2 <= x1 or y2 <= y1:
        return background

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    alpha = overlay_bgra[oy1:oy2, ox1:ox2, 3:4].astype(np.float32) / 255.0 * opacity
    bgr = overlay_bgra[oy1:oy2, ox1:ox2, :3].astype(np.float32)

    result = background.copy()
    roi = result[y1:y2, x1:x2].astype(np.float32)
    result[y1:y2, x1:x2] = (roi * (1 - alpha) + bgr * alpha).astype(np.uint8)
    return result


def generate_sample(templates, backgrounds, family_probs=None):
    """
    生成一个合成训练样本

    Returns:
        image: (256, 512, 3) — 两帧横向拼接
        label: dict with keys:
            left_stick: (x, y) in [-1, 1]
            right_stick: (x, y) in [-1, 1]
            buttons: list of 17 binary values
            left_grid_idx: int [0, 120]
            right_grid_idx: int [0, 120]
    """
    # 1. 选择模板
    tmpl = random.choice(templates)
    family = CONTROLLER_FAMILIES[tmpl["family"]]

    # 2. 随机摇杆位置 [-1, 1]
    left_x = np.clip(np.random.normal(0, 0.4), -1, 1)
    left_y = np.clip(np.random.normal(0, 0.4), -1, 1)
    right_x = np.clip(np.random.normal(0, 0.4), -1, 1)
    right_y = np.clip(np.random.normal(0, 0.4), -1, 1)

    # 3. 随机按键状态
    buttons = [0] * NUM_BUTTONS
    for i in range(NUM_BUTTONS):
        if random.random() < 0.15:
            buttons[i] = 1

    # 4. 渲染模板
    def render_overlay(stick_l, stick_r, btns):
        img = tmpl["image"].copy()
        h, w = img.shape[:2]

        # 渲染摇杆
        for stick_name, stick_pos in [("left", stick_l), ("right", stick_r)]:
            stick_cfg = family["joysticks"][stick_name]
            cx = int(stick_cfg["center"][0] * w)
            cy = int(stick_cfg["center"][1] * h)
            r = int(stick_cfg["radius"] * w)
            render_joystick(img, (cx, cy), r, stick_pos)

        # 渲染按键高亮
        for i, btn_name in enumerate(BUTTON_NAMES):
            if btns[i] == 1 and btn_name in family["buttons"]:
                btn_cfg = family["buttons"][btn_name]
                cx = int(btn_cfg["center"][0] * w)
                cy = int(btn_cfg["center"][1] * h)
                r = int(btn_cfg["radius"] * w)
                render_button_press(img, (cx, cy), r)

        return img

    # 帧 t: 当前状态
    overlay_t = render_overlay((left_x, left_y), (right_x, right_y), buttons)

    # 帧 t-1: 稍微不同的摇杆位置 (模拟运动)
    delta = 0.1
    prev_lx = np.clip(left_x + np.random.uniform(-delta, delta), -1, 1)
    prev_ly = np.clip(left_y + np.random.uniform(-delta, delta), -1, 1)
    prev_rx = np.clip(right_x + np.random.uniform(-delta, delta), -1, 1)
    prev_ry = np.clip(right_y + np.random.uniform(-delta, delta), -1, 1)
    # 按键可能有微小变化
    prev_buttons = buttons.copy()
    if random.random() < 0.1:
        idx = random.randint(0, NUM_BUTTONS - 1)
        prev_buttons[idx] = 1 - prev_buttons[idx]
    overlay_t1 = render_overlay((prev_lx, prev_ly), (prev_rx, prev_ry), prev_buttons)

    # 5. 共享背景、位置、缩放 — 两帧使用完全相同的合成上下文
    opacity = random.uniform(0.5, 1.0)
    oh, ow = overlay_t.shape[:2]
    overlay_scale = random.uniform(0.15, 0.45)
    target_w = int(FRAME_SIZE[0] * overlay_scale)
    target_h = int(oh * (target_w / ow))

    if backgrounds:
        bg = random.choice(backgrounds).copy()
        bg = cv2.resize(bg, FRAME_SIZE)
    else:
        bg = np.random.randint(30, 200, (FRAME_SIZE[1], FRAME_SIZE[0], 3),
                               dtype=np.uint8)

    # 共享放置位置
    x = random.randint(0, max(0, FRAME_SIZE[0] - target_w))
    y = random.randint(0, max(0, FRAME_SIZE[1] - target_h))
    pad = random.uniform(0.05, 0.20)
    pad_x = int(target_w * pad)
    pad_y = int(target_h * pad)
    crop_x1 = max(0, x - pad_x)
    crop_y1 = max(0, y - pad_y)
    crop_x2 = min(FRAME_SIZE[0], x + target_w + pad_x)
    crop_y2 = min(FRAME_SIZE[1], y + target_h + pad_y)

    def compose_and_crop(overlay_bgra):
        resized = cv2.resize(overlay_bgra, (target_w, target_h))
        frame = composite_overlay(bg.copy(), resized, x, y, opacity)
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_resized = cv2.resize(crop, (IMG_W // 2, IMG_H))
        return crop_resized

    frame_t = compose_and_crop(overlay_t)
    frame_t1 = compose_and_crop(overlay_t1)

    # 两帧横向拼接: (256, 512, 3)
    combined = np.concatenate([frame_t1, frame_t], axis=1)

    # 7. 数据增强
    combined = augment(combined)

    # 8. 构建标签
    left_idx = _pos_to_grid_idx_np(left_x, left_y)
    right_idx = _pos_to_grid_idx_np(right_x, right_y)

    label = {
        "left_stick": [float(left_x), float(left_y)],
        "right_stick": [float(right_x), float(right_y)],
        "buttons": buttons,
        "left_grid_idx": left_idx,
        "right_grid_idx": right_idx,
    }

    return combined, label


def augment(img):
    """数据增强: 模拟真实视频中的噪声和压缩"""
    # JPEG 压缩
    if random.random() < 0.7:
        quality = random.randint(40, 95)
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # 高斯噪声
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(3, 12),
                                 img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 亮度/对比度
    if random.random() < 0.4:
        alpha = random.uniform(0.8, 1.2)  # 对比度
        beta = random.uniform(-20, 20)     # 亮度
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # 轻微高斯模糊
    if random.random() < 0.2:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    return img


def generate_dataset(split, num_samples, templates, backgrounds, output_dir):
    """生成数据集分割"""
    split_dir = os.path.join(output_dir, split)
    img_dir = os.path.join(split_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    labels = {}
    for i in tqdm(range(num_samples), desc=f"Generating {split}"):
        img, label = generate_sample(templates, backgrounds)
        fname = f"{i:06d}.jpg"
        cv2.imwrite(os.path.join(img_dir, fname), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        labels[fname] = label

    # 保存标签
    label_path = os.path.join(split_dir, "labels.json")
    with open(label_path, "w") as f:
        json.dump(labels, f)
    print(f"  Saved {num_samples} samples to {split_dir}")

    # 同时保存紧凑的 npz 格式
    left_idx = np.array([labels[f"{i:06d}.jpg"]["left_grid_idx"]
                         for i in range(num_samples)], dtype=np.int64)
    right_idx = np.array([labels[f"{i:06d}.jpg"]["right_grid_idx"]
                          for i in range(num_samples)], dtype=np.int64)
    btns = np.array([labels[f"{i:06d}.jpg"]["buttons"]
                     for i in range(num_samples)], dtype=np.int64)
    left_stick = np.array([labels[f"{i:06d}.jpg"]["left_stick"]
                           for i in range(num_samples)], dtype=np.float32)
    right_stick = np.array([labels[f"{i:06d}.jpg"]["right_stick"]
                            for i in range(num_samples)], dtype=np.float32)

    np.savez_compressed(
        os.path.join(split_dir, "labels.npz"),
        left_grid_idx=left_idx,
        right_grid_idx=right_idx,
        buttons=btns,
        left_stick=left_stick,
        right_stick=right_stick,
    )
    print(f"  Saved labels.npz")


def main():
    print("=" * 60)
    print("Stage 2: Synthetic Data Generation")
    print("=" * 60)

    print("\n[1] Loading overlay templates...")
    templates = load_overlay_templates(TEMPLATE_DIR)
    if not templates:
        print("ERROR: No templates found!")
        return

    print(f"\n[2] Extracting background frames...")
    backgrounds = extract_background_frames(SEGMENTS_DIR)

    print(f"\n[3] Generating training set ({NUM_TRAIN} samples)...")
    generate_dataset("train", NUM_TRAIN, templates, backgrounds, OUTPUT_DIR)

    print(f"\n[4] Generating validation set ({NUM_VAL} samples)...")
    generate_dataset("val", NUM_VAL, templates, backgrounds, OUTPUT_DIR)

    # 5. 可视化检查: 保存几个样本
    print("\n[5] Saving visualization samples...")
    vis_dir = os.path.join(OUTPUT_DIR, "vis_samples")
    os.makedirs(vis_dir, exist_ok=True)
    for i in range(10):
        img, label = generate_sample(templates, backgrounds)
        # 标注信息
        info = (f"L=({label['left_stick'][0]:.2f},{label['left_stick'][1]:.2f}) "
                f"R=({label['right_stick'][0]:.2f},{label['right_stick'][1]:.2f}) "
                f"Btn={sum(label['buttons'])}")
        cv2.putText(img, info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(vis_dir, f"sample_{i:02d}.jpg"), img)
    print(f"  Saved to {vis_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
