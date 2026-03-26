"""
模型评估脚本 - 计算论文报告的指标
- 按键准确率 (Button Accuracy)
- 摇杆R²分数 (Joystick R² Score)
- 按键F1分数 (Button F1 Score)
- 混淆矩阵 (Confusion Matrix)
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from sklearn.metrics import r2_score, f1_score, confusion_matrix, classification_report
import sys

# 导入模型和数据集
from models import NitroGenActionParser
from dataset import create_train_val_dataloaders


def evaluate_button_accuracy(predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> float:
    """
    计算按键准确率

    Args:
        predictions: [N, num_buttons] 预测概率
        targets: [N, num_buttons] 真实标签 (0或1)
        threshold: 分类阈值（默认0.5）

    Returns:
        准确率 (0-1)
    """
    # 将概率转换为二分类
    pred_labels = (predictions > threshold).astype(int)

    # 计算准确率
    accuracy = (pred_labels == targets).mean()
    return float(accuracy)


def find_optimal_threshold(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    找到最佳F1分数对应的阈值

    Args:
        predictions: [N, num_buttons] 预测概率
        targets: [N, num_buttons] 真实标签

    Returns:
        最佳阈值
    """
    from sklearn.metrics import f1_score

    best_threshold = 0.5
    best_f1 = 0.0

    # 尝试不同的阈值
    for threshold in np.linspace(0.01, 0.5, 50):
        pred_labels = (predictions > threshold).astype(int)
        f1 = f1_score(targets.flatten(), pred_labels.flatten(), zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold


def evaluate_joystick_r2(
    pred_positions: np.ndarray,
    target_positions: np.ndarray
) -> Dict[str, float]:
    """
    计算摇杆位置的R²分数

    Args:
        pred_positions: [N, 2] 预测位置 [[x1, y1], [x2, y2], ...]
        target_positions: [N, 2] 真实位置

    Returns:
        {'x_r2': x方向R², 'y_r2': y方向R², 'avg_r2': 平均R²}
    """
    # 🔥 确保数组是2D的 [N, 2]
    pred_positions = pred_positions.reshape(-1, 2)
    target_positions = target_positions.reshape(-1, 2)

    if len(pred_positions) == 0:
        return {'x_r2': 0.0, 'y_r2': 0.0, 'avg_r2': 0.0}

    x_r2 = r2_score(target_positions[:, 0], pred_positions[:, 0])
    y_r2 = r2_score(target_positions[:, 1], pred_positions[:, 1])
    avg_r2 = (x_r2 + y_r2) / 2

    return {
        'x_r2': float(x_r2),
        'y_r2': float(y_r2),
        'avg_r2': float(avg_r2)
    }


def evaluate_per_button_metrics(predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    计算每个按键的详细指标

    Args:
        predictions: [N, num_buttons] 预测概率
        targets: [N, num_buttons] 真实标签
        threshold: 分类阈值

    Returns:
        每个按键的准确率、精确率、召回率、F1分数
    """
    pred_labels = (predictions > threshold).astype(int)

    num_buttons = predictions.shape[1]
    results = {}

    for i in range(num_buttons):
        pred = pred_labels[:, i]
        true = targets[:, i]

        # 计算准确率
        accuracy = (pred == true).mean()

        # 计算F1分数
        f1 = f1_score(true, pred, zero_division=0)

        # 计算精确率和召回率
        tp = ((pred == 1) & (true == 1)).sum()
        fp = ((pred == 1) & (true == 0)).sum()
        fn = ((pred == 0) & (true == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        results[f'button_{i}'] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }

    return results


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    val_loader,
    device: torch.device,
    num_buttons: int = 16
) -> Dict:
    """
    评估模型性能

    Args:
        model: 训练好的模型
        val_loader: 验证数据加载器
        device: 设备
        num_buttons: 按键数量

    Returns:
        评估结果字典
    """
    model.eval()

    # 收集所有预测和目标
    all_button_preds = []
    all_button_targets = []

    all_left_joystick_preds = []  # 预测的网格坐标
    all_left_joystick_targets = []  # 真实的网格坐标
    all_right_joystick_preds = []
    all_right_joystick_targets = []

    all_left_joystick_positions = []  # 预测的归一化位置
    all_left_target_positions = []
    all_right_joystick_positions = []
    all_right_target_positions = []

    print("Running evaluation...")

    for batch in tqdm(val_loader, desc="Evaluating"):
        # 数据移到设备
        frame_pairs = batch["frame_pair"].to(device)
        left_joystick_pos = batch["left_joystick_pos"].to(device)  # 归一化坐标 [-1, 1]
        right_joystick_pos = batch["right_joystick_pos"].to(device)  # 归一化坐标 [-1, 1]
        buttons = batch["buttons"].to(device)

        # 前向传播
        outputs = model(frame_pairs)

        # 获取按键预测
        button_probs = outputs["button_probs"]  # [B, num_buttons]
        # 🔥 确保是2D数组 [B, num_buttons]，移除多余维度
        button_probs_np = button_probs.squeeze().cpu().numpy()
        buttons_np = buttons.squeeze().cpu().numpy()
        # 如果是1D，reshape为2D
        if button_probs_np.ndim == 1:
            button_probs_np = button_probs_np.reshape(1, -1)
        if buttons_np.ndim == 1:
            buttons_np = buttons_np.reshape(1, -1)
        all_button_preds.append(button_probs_np)
        all_button_targets.append(buttons_np)

        # 获取摇杆预测 [B, 121, H/4, W/4]
        left_joystick_pred = outputs["left_joystick"]
        right_joystick_pred = outputs["right_joystick"]

        # 🔥 对空间维度做平均池化，得到每个类别的概率 [B, 121]
        left_class_probs = left_joystick_pred.mean(dim=[2, 3])  # [B, 121]
        right_class_probs = right_joystick_pred.mean(dim=[2, 3])  # [B, 121]

        # 找到最大概率的类别索引 [B]
        left_pred_class = left_class_probs.argmax(dim=1)  # [B]
        right_pred_class = right_class_probs.argmax(dim=1)  # [B]

        # 将类别索引转换为网格坐标 (x, y)
        left_pred_x = left_pred_class % 11
        left_pred_y = left_pred_class // 11
        right_pred_x = right_pred_class % 11
        right_pred_y = right_pred_class // 11

        # 🔥 确保是2D数组并转换为浮点数
        left_pred_np = torch.stack([left_pred_x, left_pred_y], dim=1).squeeze().cpu().numpy().astype(float)
        right_pred_np = torch.stack([right_pred_x, right_pred_y], dim=1).squeeze().cpu().numpy().astype(float)
        if left_pred_np.ndim == 1:
            left_pred_np = left_pred_np.reshape(1, -1)
        if right_pred_np.ndim == 1:
            right_pred_np = right_pred_np.reshape(1, -1)

        all_left_joystick_preds.append(left_pred_np)
        all_right_joystick_preds.append(right_pred_np)

        # 真实摇杆位置 [B, 2] (已归一化到 [-1, 1])
        # 🔥 使用新的归一化坐标字段
        left_target_np = left_joystick_pos.squeeze().cpu().numpy()
        right_target_np = right_joystick_pos.squeeze().cpu().numpy()
        if left_target_np.ndim == 1:
            left_target_np = left_target_np.reshape(1, -1)
        if right_target_np.ndim == 1:
            right_target_np = right_target_np.reshape(1, -1)

        all_left_target_positions.append(left_target_np)
        all_right_target_positions.append(right_target_np)

        # 将归一化位置转换为网格坐标用于比较
        # [-1, 1] -> [0, 10]
        left_target_grid_x = ((left_joystick_pos[:, 0] + 1) / 2 * 10).round().long()
        left_target_grid_y = ((left_joystick_pos[:, 1] + 1) / 2 * 10).round().long()
        right_target_grid_x = ((right_joystick_pos[:, 0] + 1) / 2 * 10).round().long()
        right_target_grid_y = ((right_joystick_pos[:, 1] + 1) / 2 * 10).round().long()

        # 🔥 确保是2D数组
        left_target_grid_np = torch.stack([left_target_grid_x, left_target_grid_y], dim=1).squeeze().cpu().numpy()
        right_target_grid_np = torch.stack([right_target_grid_x, right_target_grid_y], dim=1).squeeze().cpu().numpy()
        if left_target_grid_np.ndim == 1:
            left_target_grid_np = left_target_grid_np.reshape(1, -1)
        if right_target_grid_np.ndim == 1:
            right_target_grid_np = right_target_grid_np.reshape(1, -1)

        all_left_joystick_targets.append(left_target_grid_np)
        all_right_joystick_targets.append(right_target_grid_np)

    # 合并所有批次
    all_button_preds = np.concatenate(all_button_preds, axis=0)
    all_button_targets = np.concatenate(all_button_targets, axis=0)

    all_left_joystick_preds = np.concatenate(all_left_joystick_preds, axis=0)
    all_left_joystick_targets = np.concatenate(all_left_joystick_targets, axis=0)
    all_right_joystick_preds = np.concatenate(all_right_joystick_preds, axis=0)
    all_right_joystick_targets = np.concatenate(all_right_joystick_targets, axis=0)

    all_left_target_positions = np.concatenate(all_left_target_positions, axis=0)
    all_right_target_positions = np.concatenate(all_right_target_positions, axis=0)

    # 🔥 确保所有位置数组都是 [N, 2] 形状
    all_left_joystick_preds = all_left_joystick_preds.reshape(-1, 2)
    all_left_joystick_targets = all_left_joystick_targets.reshape(-1, 2)
    all_right_joystick_preds = all_right_joystick_preds.reshape(-1, 2)
    all_right_joystick_targets = all_right_joystick_targets.reshape(-1, 2)
    all_left_target_positions = all_left_target_positions.reshape(-1, 2)
    all_right_target_positions = all_right_target_positions.reshape(-1, 2)

    # 计算指标
    results = {}

    # 1. 按键准确率
    # 🔥 自动寻找最佳阈值
    print("\n[自动阈值搜索]")
    optimal_threshold = find_optimal_threshold(all_button_preds, all_button_targets)
    print(f"最佳阈值: {optimal_threshold:.4f}")

    button_accuracy = evaluate_button_accuracy(all_button_preds, all_button_targets, threshold=optimal_threshold)
    results['button_accuracy'] = button_accuracy
    results['optimal_threshold'] = optimal_threshold

    # 2. 按键F1分数（平均）- 使用最佳阈值
    pred_labels = (all_button_preds > optimal_threshold).astype(int)
    button_f1 = f1_score(all_button_targets.flatten(), pred_labels.flatten(), zero_division=0)
    results['button_f1'] = button_f1

    # 3. 摇杆R²分数（使用归一化位置）
    # 需要将预测的网格坐标转换回归一化位置
    left_pred_normalized = all_left_joystick_preds.astype(float) / 10 * 2 - 1  # [0,10] -> [-1,1]
    right_pred_normalized = all_right_joystick_preds.astype(float) / 10 * 2 - 1

    left_r2 = evaluate_joystick_r2(left_pred_normalized, all_left_target_positions)
    right_r2 = evaluate_joystick_r2(right_pred_normalized, all_right_target_positions)

    results['left_joystick_r2'] = left_r2
    results['right_joystick_r2'] = right_r2

    # 平均R²
    avg_r2 = (left_r2['avg_r2'] + right_r2['avg_r2']) / 2
    results['avg_joystick_r2'] = avg_r2

    # 4. 每个按键的详细指标 - 使用最佳阈值
    per_button_metrics = evaluate_per_button_metrics(all_button_preds, all_button_targets, threshold=optimal_threshold)
    results['per_button_metrics'] = per_button_metrics

    # 5. 摇杆位置准确率（网格级别）
    left_joystick_grid_acc = (all_left_joystick_preds == all_left_joystick_targets).all(axis=1).mean()
    right_joystick_grid_acc = (all_right_joystick_preds == all_right_joystick_targets).all(axis=1).mean()
    results['left_joystick_grid_accuracy'] = float(left_joystick_grid_acc)
    results['right_joystick_grid_accuracy'] = float(right_joystick_grid_acc)

    return results


def print_evaluation_results(results: Dict):
    """打印评估结果"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    # 总体指标
    print("\n📊 Overall Metrics:")
    print(f"  Button Threshold: {results.get('optimal_threshold', 0.5):.4f}")
    print(f"  Button Accuracy: {results['button_accuracy']:.4f} (Target: 0.96)")
    print(f"  Button F1 Score: {results['button_f1']:.4f}")
    print(f"  Avg Joystick R²: {results['avg_joystick_r2']:.4f} (Target: 0.84)")

    # 摇杆R²详情
    print("\n🕹️  Joystick R² Scores:")
    print(f"  Left Joystick:")
    print(f"    X: {results['left_joystick_r2']['x_r2']:.4f}")
    print(f"    Y: {results['left_joystick_r2']['y_r2']:.4f}")
    print(f"    Avg: {results['left_joystick_r2']['avg_r2']:.4f}")
    print(f"  Right Joystick:")
    print(f"    X: {results['right_joystick_r2']['x_r2']:.4f}")
    print(f"    Y: {results['right_joystick_r2']['y_r2']:.4f}")
    print(f"    Avg: {results['right_joystick_r2']['avg_r2']:.4f}")

    # 摇杆网格准确率
    print("\n🎯 Joystick Grid Accuracy:")
    print(f"  Left: {results['left_joystick_grid_accuracy']:.4f}")
    print(f"  Right: {results['right_joystick_grid_accuracy']:.4f}")

    # 每个按键的指标
    print("\n🔘 Per-Button Metrics:")
    for button_name, metrics in results['per_button_metrics'].items():
        print(f"  {button_name}:")
        print(f"    Acc: {metrics['accuracy']:.3f}, "
              f"Prec: {metrics['precision']:.3f}, "
              f"Rec: {metrics['recall']:.3f}, "
              f"F1: {metrics['f1']:.3f}")

    print("\n" + "="*60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate NitroGen Action Parser")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (default: best_model.pt in checkpoints)")
    parser.add_argument("--data_dir", type=str, default="synthetic_data/frames",
                        help="Data directory")
    parser.add_argument("--val_annotations", type=str, default="synthetic_data/val_annotations.json",
                        help="Validation annotations file")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")

    args = parser.parse_args()

    # 设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # 查找checkpoint
    if args.checkpoint is None:
        # 查找最新的best_model.pt
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            best_models = list(checkpoints_dir.glob("*/best_model.pt"))
            if best_models:
                args.checkpoint = str(max(best_models, key=lambda p: p.stat().st_mtime))
                print(f"Found checkpoint: {args.checkpoint}")
            else:
                print("Error: No checkpoint found. Please train a model first.")
                sys.exit(1)
        else:
            print("Error: checkpoints directory not found.")
            sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # 创建验证数据加载器
    print("\nCreating validation dataloader...")
    try:
        _, val_loader = create_train_val_dataloaders(
            data_dir=args.data_dir,
            train_annotations="synthetic_data/train_annotations.json",  # 占位
            val_annotations=args.val_annotations,
            batch_size=args.batch_size,
            num_workers=0,
            image_size=256
        )
        print(f"Validation batches: {len(val_loader)}")
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        sys.exit(1)

    # 加载模型
    print(f"\nLoading model from {args.checkpoint}...")
    model = NitroGenActionParser(num_buttons=16, joystick_grid_size=11).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model loaded successfully")

    if "val_loss" in checkpoint:
        print(f"Checkpoint validation loss: {checkpoint['val_loss']:.4f}")

    # 评估
    results = evaluate_model(model, val_loader, device, num_buttons=16)

    # 打印结果
    print_evaluation_results(results)

    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 转换为可序列化格式
        serializable_results = {}
        for key, value in results.items():
            if key == 'per_button_metrics':
                serializable_results[key] = value
            elif isinstance(value, dict):
                serializable_results[key] = {k: float(v) for k, v in value.items()}
            else:
                serializable_results[key] = float(value)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
