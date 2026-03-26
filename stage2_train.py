"""
Stage 2: SegFormer 训练脚本
FP16 混合精度, 梯度累积, AdamW, Linear LR Decay
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import cv2
import json
import numpy as np
import random
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from stage2_model import (
    SegFormerActionParser, ActionParserLoss,
    grid_to_position, position_to_grid_idx,
    GRID_SIZE, NUM_BUTTONS,
)

# ========== 配置 ==========
# 通过命令行参数选择模式:
#   python stage2_train.py              → 默认: 真实数据训练
#   python stage2_train.py pretrain     → 阶段1: 合成数据预训练
#   python stage2_train.py finetune     → 阶段2: 真实数据微调 (加载预训练权重)

CONFIGS = {
    # 默认/真实数据训练 (无预训练)
    "default": {
        "data_dir": r"D:\Python\videos\stage2_data",
        "save_dir": r"D:\Python\videos\results\stage2_checkpoints",
        "pretrained": None,
        "lr": 1e-4,
        "epochs": 30,
    },
    # 阶段1: 合成数据预训练
    "pretrain": {
        "data_dir": r"D:\Python\videos\stage2_data_synthetic",
        "save_dir": r"D:\Python\videos\results\stage2_pretrain",
        "pretrained": None,
        "lr": 1e-4,
        "epochs": 20,
    },
    # 阶段2: 真实数据微调
    "finetune": {
        "data_dir": r"D:\Python\videos\stage2_data",
        "save_dir": r"D:\Python\videos\results\stage2_finetune",
        "pretrained": r"D:\Python\videos\results\stage2_pretrain\best.pt",
        "lr": 2e-5,             # 微调用更低学习率
        "epochs": 20,
    },
}

# 解析模式
_mode = sys.argv[1] if len(sys.argv) > 1 else "default"
_cfg = CONFIGS.get(_mode, CONFIGS["default"])

CONFIG = {
    "data_dir": _cfg["data_dir"],
    "save_dir": _cfg["save_dir"],
    "pretrained": _cfg["pretrained"],
    "batch_size": 2,            # 4GB VRAM 安全值
    "accumulation_steps": 16,   # 有效 batch = 32
    "lr": _cfg["lr"],
    "weight_decay": 0.1,
    "epochs": _cfg["epochs"],
    "img_h": 256,
    "img_w": 512,
    "num_workers": 0,
    "log_interval": 50,         # 每 N 个 step 打印日志
    "val_interval": 1,          # 每 N 个 epoch 验证
    "save_interval": 5,         # 每 N 个 epoch 保存
    "mode": _mode,
}

# ImageNet 归一化
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def augment_image(img_uint8):
    """训练时数据增强 (输入 uint8 BGR, 输出 uint8 BGR)"""
    img = img_uint8

    # 亮度/对比度抖动
    if random.random() < 0.5:
        alpha = random.uniform(0.8, 1.2)   # 对比度
        beta = random.uniform(-20, 20)      # 亮度
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # 高斯噪声
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(3, 10), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # JPEG 压缩伪影
    if random.random() < 0.4:
        quality = random.randint(50, 90)
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # 轻微高斯模糊
    if random.random() < 0.15:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    return img


class ControllerDataset(Dataset):
    """控制器数据集 (支持真实数据和合成数据, 格式相同)"""

    def __init__(self, data_dir, split="train", augment=False):
        self.img_dir = os.path.join(data_dir, split, "images")
        self.augment = augment
        npz_path = os.path.join(data_dir, split, "labels.npz")

        data = np.load(npz_path)
        self.left_grid_idx = data["left_grid_idx"]
        self.right_grid_idx = data["right_grid_idx"]
        self.buttons = data["buttons"]
        self.left_stick = data["left_stick"]
        self.right_stick = data["right_stick"]

        self.filenames = [f"{i:06d}.jpg" for i in range(len(self.left_grid_idx))]
        print(f"  {split}: {len(self.filenames)} samples (augment={augment})")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((CONFIG["img_h"], CONFIG["img_w"], 3), dtype=np.uint8)

        # 训练时数据增强
        if self.augment:
            img = augment_image(img)

        # BGR -> RGB, HWC -> CHW, [0,1], 归一化
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)

        target = {
            "left_grid_idx": torch.tensor(self.left_grid_idx[idx], dtype=torch.long),
            "right_grid_idx": torch.tensor(self.right_grid_idx[idx], dtype=torch.long),
            "buttons": torch.tensor(self.buttons[idx], dtype=torch.float32),
            "left_stick": torch.tensor(self.left_stick[idx], dtype=torch.float32),
            "right_stick": torch.tensor(self.right_stick[idx], dtype=torch.float32),
        }
        return img, target


def evaluate(model, val_loader, criterion, device):
    """验证集评估"""
    model.eval()
    total_loss = 0
    total_seg_l = 0
    total_seg_r = 0
    total_cls = 0
    n_batches = 0

    all_pred_lx, all_pred_ly = [], []
    all_pred_rx, all_pred_ry = [], []
    all_gt_lx, all_gt_ly = [], []
    all_gt_rx, all_gt_ry = [], []
    all_pred_btn, all_gt_btn = [], []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

            with torch.amp.autocast(device.type, dtype=torch.float16):
                pred = model(images)
                loss, loss_dict = criterion(pred, targets)

            total_loss += loss.item()
            total_seg_l += loss_dict["seg_left"]
            total_seg_r += loss_dict["seg_right"]
            total_cls += loss_dict["cls"]
            n_batches += 1

            # 摇杆位置
            lx, ly = grid_to_position(pred["seg_left"])
            rx, ry = grid_to_position(pred["seg_right"])
            all_pred_lx.append(lx.cpu())
            all_pred_ly.append(ly.cpu())
            all_pred_rx.append(rx.cpu())
            all_pred_ry.append(ry.cpu())
            all_gt_lx.append(targets["left_stick"][:, 0].cpu())
            all_gt_ly.append(targets["left_stick"][:, 1].cpu())
            all_gt_rx.append(targets["right_stick"][:, 0].cpu())
            all_gt_ry.append(targets["right_stick"][:, 1].cpu())

            # 按键
            btn_pred = (pred["buttons"].sigmoid() > 0.5).float()
            all_pred_btn.append(btn_pred.cpu())
            all_gt_btn.append(targets["buttons"].cpu())

    # 汇总
    pred_lx = torch.cat(all_pred_lx)
    pred_ly = torch.cat(all_pred_ly)
    gt_lx = torch.cat(all_gt_lx)
    gt_ly = torch.cat(all_gt_ly)
    pred_rx = torch.cat(all_pred_rx)
    pred_ry = torch.cat(all_pred_ry)
    gt_rx = torch.cat(all_gt_rx)
    gt_ry = torch.cat(all_gt_ry)

    # R² (摇杆) — 方差过小时返回 0 (无意义)
    def r2_score(pred, gt):
        ss_res = ((pred - gt) ** 2).sum()
        ss_tot = ((gt - gt.mean()) ** 2).sum()
        if ss_tot.item() < 1e-6:
            return 0.0  # 方差接近零, R² 无意义
        return 1 - (ss_res / ss_tot).item()

    r2_lx = r2_score(pred_lx, gt_lx)
    r2_ly = r2_score(pred_ly, gt_ly)
    r2_rx = r2_score(pred_rx, gt_rx)
    r2_ry = r2_score(pred_ry, gt_ry)
    r2_avg = (r2_lx + r2_ly + r2_rx + r2_ry) / 4

    # 按键准确率
    pred_btn = torch.cat(all_pred_btn)
    gt_btn = torch.cat(all_gt_btn)
    btn_acc = (pred_btn == gt_btn).float().mean().item()

    # 逐按键 F1 (简化: 仅报告整体)
    tp = ((pred_btn == 1) & (gt_btn == 1)).float().sum().item()
    fp = ((pred_btn == 1) & (gt_btn == 0)).float().sum().item()
    fn = ((pred_btn == 0) & (gt_btn == 1)).float().sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "loss": total_loss / n_batches,
        "seg_left": total_seg_l / n_batches,
        "seg_right": total_seg_r / n_batches,
        "cls": total_cls / n_batches,
        "r2_lx": r2_lx, "r2_ly": r2_ly,
        "r2_rx": r2_rx, "r2_ry": r2_ry,
        "r2_avg": r2_avg,
        "btn_acc": btn_acc,
        "btn_f1": f1,
    }


def main():
    print("=" * 60)
    print(f"Stage 2: SegFormer Training [{CONFIG['mode']}]")
    print(f"  Data:       {CONFIG['data_dir']}")
    print(f"  Save:       {CONFIG['save_dir']}")
    print(f"  Pretrained: {CONFIG['pretrained'] or 'None'}")
    print(f"  LR:         {CONFIG['lr']}")
    print(f"  Epochs:     {CONFIG['epochs']}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    # 1. 数据集
    print("\n[1] Loading datasets...")
    train_dataset = ControllerDataset(CONFIG["data_dir"], "train", augment=True)
    val_dataset = ControllerDataset(CONFIG["data_dir"], "val", augment=False)
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"],
        shuffle=True, num_workers=CONFIG["num_workers"],
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )

    # 2. 模型
    print("\n[2] Creating model...")
    model = SegFormerActionParser().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # 加载预训练权重
    if CONFIG["pretrained"] and os.path.exists(CONFIG["pretrained"]):
        print(f"  Loading pretrained: {CONFIG['pretrained']}")
        state_dict = torch.load(CONFIG["pretrained"], weights_only=True)
        model.load_state_dict(state_dict)
        print("  Pretrained weights loaded!")

    # 3. 优化器 + 调度器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )
    steps_per_epoch = len(train_loader) // CONFIG["accumulation_steps"]
    total_steps = CONFIG["epochs"] * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=total_steps
    )
    scaler = torch.amp.GradScaler(device.type)
    criterion = ActionParserLoss(seg_weight=1.0, cls_weight=1.0)

    print(f"  Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}")
    print(f"  Effective batch size: {CONFIG['batch_size'] * CONFIG['accumulation_steps']}")

    # 4. 训练循环
    print("\n[3] Training...")
    best_r2 = -999
    t_start = time.time()

    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_loss = 0
        epoch_steps = 0
        optimizer.zero_grad()

        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

            with torch.amp.autocast(device.type, dtype=torch.float16):
                pred = model(images)
                loss, loss_dict = criterion(pred, targets)
                loss = loss / CONFIG["accumulation_steps"]

            scaler.scale(loss).backward()

            if (step + 1) % CONFIG["accumulation_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                epoch_steps += 1

            epoch_loss += loss.item() * CONFIG["accumulation_steps"]

            if (step + 1) % (CONFIG["log_interval"] * CONFIG["accumulation_steps"]) == 0:
                avg = epoch_loss / (step + 1)
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t_start
                print(f"  E{epoch+1} step {epoch_steps}/{steps_per_epoch} "
                      f"loss={avg:.4f} (seg_L={loss_dict['seg_left']:.3f} "
                      f"seg_R={loss_dict['seg_right']:.3f} "
                      f"cls={loss_dict['cls']:.3f}) "
                      f"lr={lr:.6f} [{elapsed:.0f}s]")

        avg_train_loss = epoch_loss / len(train_loader)

        # 验证
        if (epoch + 1) % CONFIG["val_interval"] == 0:
            metrics = evaluate(model, val_loader, criterion, device)
            is_best = metrics["r2_avg"] > best_r2
            if is_best:
                best_r2 = metrics["r2_avg"]
                torch.save(model.state_dict(),
                           os.path.join(CONFIG["save_dir"], "best.pt"))

            elapsed = time.time() - t_start
            print(f"  E{epoch+1} val: loss={metrics['loss']:.4f} "
                  f"R2_avg={metrics['r2_avg']:.4f} "
                  f"(Lx={metrics['r2_lx']:.3f} Ly={metrics['r2_ly']:.3f} "
                  f"Rx={metrics['r2_rx']:.3f} Ry={metrics['r2_ry']:.3f}) "
                  f"btn_acc={metrics['btn_acc']:.4f} F1={metrics['btn_f1']:.3f} "
                  f"{'*BEST*' if is_best else ''} [{elapsed:.0f}s]")

        # 保存定期 checkpoint
        if (epoch + 1) % CONFIG["save_interval"] == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_r2": best_r2,
            }, os.path.join(CONFIG["save_dir"], f"checkpoint_e{epoch+1:02d}.pt"))

    # 最终验证
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    # 加载最佳模型
    model.load_state_dict(torch.load(
        os.path.join(CONFIG["save_dir"], "best.pt"), weights_only=True))
    final_metrics = evaluate(model, val_loader, criterion, device)
    total_time = time.time() - t_start
    print(f"  R2_avg: {final_metrics['r2_avg']:.4f}")
    print(f"  R2: Lx={final_metrics['r2_lx']:.3f} Ly={final_metrics['r2_ly']:.3f} "
          f"Rx={final_metrics['r2_rx']:.3f} Ry={final_metrics['r2_ry']:.3f}")
    print(f"  Button acc: {final_metrics['btn_acc']:.4f}")
    print(f"  Button F1: {final_metrics['btn_f1']:.3f}")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  Checkpoints: {CONFIG['save_dir']}")


if __name__ == "__main__":
    main()
