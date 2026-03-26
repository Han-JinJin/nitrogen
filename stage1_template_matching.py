"""
Stage 1: Template Matching for Controller Overlay Detection
按 NitroGen 论文复现: SIFT + XFeat 特征匹配 + 几何变换

方法1: RootSIFT + FLANN + Homography (经典鲁棒匹配)
方法2: XFeat + LighterGlue (学习特征 + 学习匹配器, xfeat-lighterglue.pt)
方法3: NCC 后备 (仅当前两个都无匹配时)
"""

import cv2
import numpy as np
import os
import glob
import json
import time
import argparse
import pickle
import hashlib
from pathlib import Path
import torch

# ========== 配置 ==========
VIDEO_DIR = r"D:\Python\videos"
TEMPLATE_DIR = r"D:\Python\videos\shoubing\overlay_templates"
OUTPUT_DIR = r"D:\Python\videos\results"
FRAMES_PER_VIDEO = 25
MIN_INLIERS = 20
LOWE_RATIO = 0.75
NCC_THRESHOLD = 0.45
EARLY_STOP_INLIERS = 50


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Controller Overlay Detection")
    parser.add_argument("--video-dir", default=VIDEO_DIR, help="视频目录")
    parser.add_argument("--template-dir", default=TEMPLATE_DIR, help="模板目录")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="输出目录")
    return parser.parse_args()


def extract_frames(video_path, num_frames=25):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def load_templates(template_dir):
    templates = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        for path in glob.glob(os.path.join(template_dir, ext)):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if len(img.shape) == 3 and img.shape[2] == 4:
                alpha = img[:, :, 3:4] / 255.0
                bgr = img[:, :, :3]
                white = np.ones_like(bgr) * 255
                img = (bgr * alpha + white * (1 - alpha)).astype(np.uint8)
            templates.append({"name": Path(path).stem, "path": path, "image": img})
    print(f"  Loaded {len(templates)} templates")
    return templates


def discover_videos(video_dir):
    """扫描视频目录，支持多种结构:
    1. segments/1/*.mp4, segments/2/*.mp4, ...
    2. [name]/segments/*.mp4
    3. 直接 *.mp4
    """
    videos = []
    base = Path(video_dir)

    # 优先: segments/N/segments/*.mp4 结构 (从服务器下载的格式)
    segments_base = base / "segments"
    if segments_base.is_dir():
        for sub in sorted(segments_base.iterdir(), key=lambda p: p.name.zfill(10)):
            if sub.is_dir():
                # 检查 sub/segments/*.mp4 (嵌套结构)
                nested = sub / "segments"
                search_dir = nested if nested.is_dir() else sub
                mp4s = sorted(search_dir.glob("*.mp4"))
                if mp4s:
                    videos.append({
                        "name": sub.name,
                        "first_segment": str(mp4s[0]),
                    })
        if videos:
            return videos

    # 其次: [name]/segments/*.mp4 结构
    for segments_dir in sorted(base.glob("*/segments")):
        mp4s = sorted(segments_dir.glob("*.mp4"))
        if mp4s:
            videos.append({
                "name": segments_dir.parent.name,
                "first_segment": str(mp4s[0]),
            })
    if videos:
        return videos

    # 后备: 直接扫描目录下的 mp4
    for mp4 in sorted(base.glob("*.mp4")):
        videos.append({"name": mp4.stem, "first_segment": str(mp4)})
    return videos


# ========== 模板特征缓存 ==========

def compute_templates_hash(templates):
    """根据模板文件内容计算哈希，判断缓存是否过期"""
    h = hashlib.md5()
    for tmpl in sorted(templates, key=lambda t: t["name"]):
        with open(tmpl["path"], "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


def _kp_to_list(kp_list):
    """OpenCV KeyPoint -> 可序列化 list"""
    return [{"pt": k.pt, "size": k.size, "angle": k.angle,
             "response": k.response, "octave": k.octave,
             "class_id": k.class_id} for k in kp_list]


def _kp_from_list(data):
    """可序列化 list -> OpenCV KeyPoint"""
    return [cv2.KeyPoint(d["pt"][0], d["pt"][1], d["size"],
                         d["angle"], d["response"], d["octave"],
                         d["class_id"]) for d in data]


def save_sift_cache(templates, cache_path):
    data = {}
    for tmpl in templates:
        feats = []
        for f in tmpl.get("sift_features", []):
            feats.append({
                "scale": f["scale"], "kp": _kp_to_list(f["kp"]),
                "des": f["des"], "w": f["w"], "h": f["h"],
            })
        data[tmpl["name"]] = feats
    with open(cache_path, "wb") as fh:
        pickle.dump(data, fh)


def load_sift_cache(templates, cache_path):
    with open(cache_path, "rb") as fh:
        data = pickle.load(fh)
    for tmpl in templates:
        if tmpl["name"] in data:
            feats = []
            for f in data[tmpl["name"]]:
                feats.append({
                    "scale": f["scale"], "kp": _kp_from_list(f["kp"]),
                    "des": f["des"], "w": f["w"], "h": f["h"],
                })
            tmpl["sift_features"] = feats
        else:
            tmpl["sift_features"] = []


def save_xfeat_cache(templates, cache_path):
    data = {}
    for tmpl in templates:
        if "xfeat_features" in tmpl:
            data[tmpl["name"]] = tmpl["xfeat_features"]
    torch.save(data, cache_path)


def load_xfeat_cache(templates, cache_path):
    data = torch.load(cache_path, weights_only=False)
    for tmpl in templates:
        tmpl["xfeat_features"] = data.get(tmpl["name"], [])


def validate_homography(H, tmpl_shape, frame_shape):
    """验证 Homography 投影结果合理性"""
    if H is None:
        return False, None
    h, w = tmpl_shape[:2]
    fh, fw = frame_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    try:
        projected = cv2.perspectiveTransform(corners, H)
    except cv2.error:
        return False, None
    pts = projected.reshape(-1, 2)

    # bbox 检查
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    bw, bh = x_max - x_min, y_max - y_min
    area = bw * bh
    frame_area = fh * fw

    if bw < 15 or bh < 15:
        return False, None
    if area < frame_area * 0.002 or area > frame_area * 0.5:
        return False, None
    # 大部分应在帧内
    if x_max < 0 or y_max < 0 or x_min > fw or y_min > fh:
        return False, None
    # 检查凸性（防止退化投影）
    cross_products = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        p3 = pts[(i + 2) % 4]
        v1 = p2 - p1
        v2 = p3 - p2
        cross_products.append(v1[0] * v2[1] - v1[1] * v2[0])
    if not (all(c > 0 for c in cross_products) or all(c < 0 for c in cross_products)):
        return False, None

    return True, projected


# ========== 方法1: RootSIFT + FLANN + Homography ==========

def rootsift_descriptors(des):
    """RootSIFT: L1 normalize -> sqrt -> L2 normalize"""
    if des is None:
        return None
    des = des.astype(np.float32)
    # L1 normalize
    norms = np.linalg.norm(des, ord=1, axis=1, keepdims=True)
    norms[norms == 0] = 1
    des = des / norms
    # sqrt
    des = np.sqrt(des)
    return des


def precompute_sift(templates, sift):
    """预计算模板多尺度 RootSIFT 特征"""
    target_widths = [48, 80, 128, 192, 256]
    for tmpl in templates:
        tmpl["sift_features"] = []
        h, w = tmpl["image"].shape[:2]
        for tw in target_widths:
            scale = tw / w
            th = int(h * scale)
            if tw < 20 or th < 20:
                continue
            resized = cv2.resize(tmpl["image"], (tw, th))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            if des is not None and len(kp) >= 4:
                rdes = rootsift_descriptors(des)
                tmpl["sift_features"].append({
                    "scale": scale, "kp": kp, "des": rdes,
                    "w": tw, "h": th
                })


def run_sift_matching(all_frames, templates, sift, flann):
    """RootSIFT + FLANN + Homography"""
    # 预计算帧特征
    frame_feats = []
    for frame in all_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        rdes = rootsift_descriptors(des)
        frame_feats.append((kp, rdes))

    best = {"inliers": 0, "template": None, "projected": None,
            "frame": None, "frame_idx": -1}
    total = len(all_frames) * len(templates)
    done = 0

    for f_idx, (f_kp, f_des) in enumerate(frame_feats):
        if f_des is None or len(f_kp) < 4:
            done += len(templates)
            continue
        for tmpl in templates:
            done += 1
            if done % 100 == 0:
                print(f"\r    SIFT: {done}/{total}", end="", flush=True)
            for feat in tmpl["sift_features"]:
                try:
                    matches = flann.knnMatch(feat["des"], f_des, k=2)
                except cv2.error:
                    continue
                good = [m for mp in matches if len(mp) == 2
                        for m in [mp[0]] if m.distance < LOWE_RATIO * mp[1].distance]
                if len(good) < MIN_INLIERS:
                    continue

                src = np.float32([feat["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([f_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                if H is None or mask is None:
                    continue
                inliers = int(mask.sum())
                if inliers <= best["inliers"] or inliers < MIN_INLIERS:
                    continue

                valid, proj = validate_homography(
                    H, (feat["h"], feat["w"]), all_frames[f_idx].shape)
                if valid:
                    best.update({"inliers": inliers, "template": tmpl["name"],
                                 "projected": proj, "frame": all_frames[f_idx],
                                 "frame_idx": f_idx})
                    if best["inliers"] >= EARLY_STOP_INLIERS:
                        print(f"\n    SIFT early stop: {best['inliers']} inliers")
                        return best
    print()
    return best


# ========== 方法2: XFeat + LighterGlue (学习匹配器) ==========

def precompute_xfeat_features(all_frames, templates, xfeat,
                              cache_dir=None, tmpl_hash=None):
    """预计算帧和模板的 XFeat 特征 (避免重复提取)"""
    # 预计算帧特征
    frame_xfeats = []
    for frame in all_frames:
        im = xfeat.parse_input(frame)
        feat = xfeat.detectAndCompute(im, top_k=4096)[0]
        feat['image_size'] = torch.tensor([frame.shape[1], frame.shape[0]])
        frame_xfeats.append(feat)

    # 检查模板 XFeat 缓存
    frame_w = all_frames[0].shape[1]
    xfeat_cache_path = None
    if cache_dir and tmpl_hash:
        xfeat_cache_path = os.path.join(cache_dir, f"xfeat_{tmpl_hash}_{frame_w}.pt")
        if os.path.exists(xfeat_cache_path):
            print(f"    Loading XFeat template cache...")
            load_xfeat_cache(templates, xfeat_cache_path)
            return frame_xfeats

    # 预计算模板多尺度特征
    target_ratios = [0.08, 0.15, 0.25, 0.35]
    for tmpl in templates:
        tmpl["xfeat_features"] = []
        th, tw = tmpl["image"].shape[:2]
        for ratio in target_ratios:
            target_w = int(frame_w * ratio)
            scale = target_w / tw
            target_h = int(th * scale)
            if target_w < 30 or target_h < 30:
                continue
            if target_w >= all_frames[0].shape[1] or target_h >= all_frames[0].shape[0]:
                continue
            resized = cv2.resize(tmpl["image"], (target_w, target_h))
            # XFeat 要求最小尺寸
            if target_w < 32 or target_h < 32:
                continue
            try:
                im = xfeat.parse_input(resized)
                feat = xfeat.detectAndCompute(im, top_k=4096)[0]
            except Exception:
                continue
            if len(feat['keypoints']) < 4:
                continue
            feat['image_size'] = torch.tensor([target_w, target_h])
            tmpl["xfeat_features"].append({
                "feat": feat, "w": target_w, "h": target_h
            })

    # 保存缓存
    if xfeat_cache_path:
        save_xfeat_cache(templates, xfeat_cache_path)
        print(f"    Saved XFeat template cache")

    return frame_xfeats


def run_xfeat_matching(all_frames, templates, xfeat, frame_xfeats):
    """XFeat + LighterGlue (学习特征 + 学习匹配器)"""
    best = {"inliers": 0, "template": None, "projected": None,
            "frame": None, "frame_idx": -1}

    sample_indices = np.linspace(0, len(all_frames) - 1,
                                 min(5, len(all_frames)), dtype=int)
    total = len(sample_indices) * len(templates)
    done = 0

    for f_idx in sample_indices:
        f_feat = frame_xfeats[f_idx]
        for tmpl in templates:
            done += 1
            if done % 10 == 0:
                print(f"\r    XFeat+LighterGlue: {done}/{total}", end="", flush=True)

            for tf in tmpl["xfeat_features"]:
                try:
                    mkpts_0, mkpts_1, _ = xfeat.match_lighterglue(
                        tf["feat"], f_feat, min_conf=0.1)
                except Exception:
                    continue

                if len(mkpts_0) < MIN_INLIERS:
                    continue

                pts0 = mkpts_0.reshape(-1, 1, 2).astype(np.float32)
                pts1 = mkpts_1.reshape(-1, 1, 2).astype(np.float32)

                H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
                if H is None or mask is None:
                    continue
                inliers = int(mask.sum())
                if inliers <= best["inliers"] or inliers < MIN_INLIERS:
                    continue

                valid, proj = validate_homography(
                    H, (tf["h"], tf["w"]), all_frames[f_idx].shape)
                if valid:
                    best.update({"inliers": inliers, "template": tmpl["name"],
                                 "projected": proj, "frame": all_frames[f_idx],
                                 "frame_idx": int(f_idx)})
                    if best["inliers"] >= EARLY_STOP_INLIERS:
                        print(f"\n    XFeat early stop: {best['inliers']} inliers")
                        return best
    print()
    return best


# ========== 方法3: NCC 后备 ==========

def run_ncc_matching(all_frames, templates):
    best = {"score": -1, "template": None, "bbox": None,
            "frame": None, "frame_idx": -1}

    sample_indices = np.linspace(0, len(all_frames) - 1,
                                 min(3, len(all_frames)), dtype=int)
    total = len(sample_indices) * len(templates)
    done = 0

    for f_idx in sample_indices:
        frame = all_frames[f_idx]
        gray_f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge_f = cv2.Canny(gray_f, 50, 150)
        fw, fh = frame.shape[1], frame.shape[0]

        for tmpl in templates:
            done += 1
            if done % 10 == 0:
                print(f"\r    NCC: {done}/{total}", end="", flush=True)

            gray_t = cv2.cvtColor(tmpl["image"], cv2.COLOR_BGR2GRAY)
            edge_t = cv2.Canny(gray_t, 50, 150)
            tw, th = tmpl["image"].shape[1], tmpl["image"].shape[0]

            min_tw = int(fw * 0.05)
            max_tw = int(fw * 0.45)
            step = max(1, (max_tw - min_tw) // 8)

            for target_w in range(min_tw, max_tw + 1, step):
                s = target_w / tw
                target_h = int(th * s)
                if target_h < 10 or target_w >= fw or target_h >= fh:
                    continue
                rg = cv2.resize(gray_t, (target_w, target_h))
                re = cv2.resize(edge_t, (target_w, target_h))
                res_g = cv2.matchTemplate(gray_f, rg, cv2.TM_CCOEFF_NORMED)
                res_e = cv2.matchTemplate(edge_f, re, cv2.TM_CCOEFF_NORMED)
                res = 0.4 * res_g + 0.6 * res_e
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > best["score"]:
                    best.update({
                        "score": max_val, "template": tmpl["name"],
                        "bbox": (max_loc[0], max_loc[1],
                                 max_loc[0] + target_w, max_loc[1] + target_h),
                        "frame": frame, "frame_idx": int(f_idx)
                    })
    print()
    return best


# ========== 主处理逻辑 ==========

def process_video(video_name, video_path, templates, sift, flann, xfeat,
                  cache_dir=None, tmpl_hash=None):
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"  Segment: {video_path}")
    print(f"{'='*60}")

    all_frames = extract_frames(video_path, num_frames=FRAMES_PER_VIDEO)
    if not all_frames:
        return None
    print(f"  Sampled {len(all_frames)} frames")

    # --- 方法1: RootSIFT ---
    print("  [Method 1: RootSIFT + Homography]")
    sift_result = run_sift_matching(all_frames, templates, sift, flann)
    sift_early_stopped = sift_result["inliers"] >= EARLY_STOP_INLIERS
    if sift_result["inliers"] >= MIN_INLIERS:
        pts = sift_result["projected"].reshape(-1, 2)
        bbox = [int(pts[:, 0].min()), int(pts[:, 1].min()),
                int(pts[:, 0].max()), int(pts[:, 1].max())]
        print(f"    -> {sift_result['template']} ({sift_result['inliers']} inliers) bbox={bbox}")
        if sift_early_stopped:
            print(f"    -> Early stop! Skipping XFeat & NCC")
    else:
        print(f"    -> no match ({sift_result['inliers']} best)")

    # --- 方法2: XFeat + LighterGlue (跳过如果 SIFT 早停) ---
    xfeat_result = {"inliers": 0, "template": None, "projected": None,
                    "frame": None, "frame_idx": -1}
    if not sift_early_stopped:
        print("  [Method 2: XFeat + LighterGlue]")
        print("    Pre-computing XFeat features...", flush=True)
        frame_xfeats = precompute_xfeat_features(
            all_frames, templates, xfeat, cache_dir, tmpl_hash)
        xfeat_result = run_xfeat_matching(all_frames, templates, xfeat, frame_xfeats)
        if xfeat_result["inliers"] >= MIN_INLIERS:
            pts = xfeat_result["projected"].reshape(-1, 2)
            bbox = [int(pts[:, 0].min()), int(pts[:, 1].min()),
                    int(pts[:, 0].max()), int(pts[:, 1].max())]
            print(f"    -> {xfeat_result['template']} ({xfeat_result['inliers']} inliers) bbox={bbox}")
            if xfeat_result["inliers"] >= EARLY_STOP_INLIERS:
                print(f"    -> Early stop! Skipping NCC")
        else:
            print(f"    -> no match ({xfeat_result['inliers']} best)")

    # --- 选最优 ---
    feature_best = sift_result if sift_result["inliers"] >= xfeat_result["inliers"] else xfeat_result
    method_name = "SIFT" if sift_result["inliers"] >= xfeat_result["inliers"] else "XFeat"

    result = {"video": video_name, "method": None, "template": None,
              "bbox": None, "score": 0, "frame": None, "matched": False}

    if feature_best["inliers"] >= MIN_INLIERS and feature_best["projected"] is not None:
        pts = feature_best["projected"].reshape(-1, 2)
        bbox = (int(pts[:, 0].min()), int(pts[:, 1].min()),
                int(pts[:, 0].max()), int(pts[:, 1].max()))
        result.update({
            "method": method_name, "template": feature_best["template"],
            "bbox": bbox, "score": feature_best["inliers"],
            "frame": feature_best["frame"], "projected": feature_best["projected"],
            "matched": True,
        })
    else:
        # --- 方法3: NCC 后备 ---
        print("  [Method 3: NCC (fallback)]")
        ncc_result = run_ncc_matching(all_frames, templates)
        if ncc_result["score"] >= NCC_THRESHOLD:
            print(f"    -> {ncc_result['template']} (score={ncc_result['score']:.3f}) "
                  f"bbox={ncc_result['bbox']}")
            result.update({
                "method": "NCC", "template": ncc_result["template"],
                "bbox": ncc_result["bbox"], "score": ncc_result["score"],
                "frame": ncc_result["frame"], "matched": True,
            })
        else:
            print(f"    -> no match (best={ncc_result['score']:.3f})")

    elapsed = time.time() - t0
    if result["matched"]:
        bbox = result["bbox"]
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        print(f"  => RESULT: {result['method']} '{result['template']}' "
              f"bbox=({bbox[0]},{bbox[1]},{bw}x{bh}) score={result['score']} [{elapsed:.1f}s]")
    else:
        print(f"  => NO MATCH [{elapsed:.1f}s]")
    return result


def draw_result(frame, result):
    vis = frame.copy()
    if "projected" in result and result["projected"] is not None:
        pts = result["projected"].reshape(-1, 2).astype(int)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
    elif result["bbox"]:
        x1, y1, x2, y2 = result["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if result["bbox"]:
        bbox = result["bbox"]
        label = f"{result['template']} [{result['method']}]"
        cv2.putText(vis, label, (bbox[0], max(bbox[1] - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    return vis


def main():
    args = parse_args()
    video_dir = args.video_dir
    template_dir = args.template_dir
    output_dir = args.output_dir

    t_start = time.time()
    print("=" * 60)
    print("Stage 1: Controller Overlay Detection")
    print("  RootSIFT + Homography | XFeat + LighterGlue | NCC")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # --- 初始化 ---
    print("\n[1] Initializing...")
    sift = cv2.SIFT_create(nfeatures=2000)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=80)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  XFeat device: {device}")
    xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat',
                           pretrained=True, top_k=4096, trust_repo=True)

    # --- 加载模板 ---
    print("\n[2] Loading templates...")
    templates = load_templates(template_dir)
    if not templates:
        print("ERROR: No templates!")
        return

    tmpl_hash = compute_templates_hash(templates)
    sift_cache_path = os.path.join(cache_dir, f"sift_{tmpl_hash}.pkl")

    if os.path.exists(sift_cache_path):
        print(f"  Loading SIFT cache (hash={tmpl_hash})...")
        load_sift_cache(templates, sift_cache_path)
    else:
        print("  Pre-computing RootSIFT features...")
        precompute_sift(templates, sift)
        save_sift_cache(templates, sift_cache_path)
        print(f"  Saved SIFT cache (hash={tmpl_hash})")

    # --- 发现视频 ---
    videos = discover_videos(video_dir)
    print(f"\n[3] Found {len(videos)} videos")
    for v in videos:
        print(f"    {v['name']}")

    # --- 处理 ---
    results = {}
    for video in videos:
        name = video["name"]
        result = process_video(name, video["first_segment"], templates,
                               sift, flann, xfeat, cache_dir, tmpl_hash)
        if result is None:
            continue
        if result["matched"] and result["frame"] is not None:
            vis = draw_result(result["frame"], result)
            cv2.imwrite(os.path.join(output_dir, f"{name}_match.png"), vis)
        results[name] = {
            "video": name,
            "matched": result["matched"],
            "method": result["method"],
            "template": result["template"],
            "bbox": result["bbox"],
            "score": float(result["score"]) if result["score"] else 0,
        }

    with open(os.path.join(output_dir, "stage1_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    matched = 0
    for p, r in results.items():
        if r["matched"]:
            matched += 1
            bbox = r["bbox"]
            bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            print(f"  {p}: {r['method']} -> {r['template']} "
                  f"({bw}x{bh} @ ({bbox[0]},{bbox[1]})) score={r['score']}")
        else:
            print(f"  {p}: NO MATCH")
    print(f"\nMatched: {matched}/{len(results)} | Time: {total_time:.1f}s")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
