"""
NitroGen Stage 2 动作解析推理脚本
功能：从视频文件中解析手柄动作（摇杆位置 + 按键状态）

使用方法:
    # 单个视频推理
    python inference.py --video path/to/video.mp4

    # 批量推理
    python inference.py --video_dir path/to/videos/

    # 指定模型checkpoint
    python inference.py --video video.mp4 --checkpoint checkpoints/best_model.pt

    # 指定输出目录
    python inference.py --video video.mp4 --output results/
"""

import os
import sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
import numpy as np
import cv2
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import argparse

from models import NitroGenActionParser
from dataset import VideoFramePairDataset


def load_model(checkpoint_path: str, device: torch.device) -> NitroGenActionParser:
    """
    加载训练好的模型

    Args:
        checkpoint_path: 模型checkpoint路径
        device: 设备

    Returns:
        加载好的模型
    """
    print(f"加载模型: {checkpoint_path}")

    model = NitroGenActionParser(num_buttons=16, joystick_grid_size=11).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if "val_loss" in checkpoint:
        print(f"Checkpoint验证损失: {checkpoint['val_loss']:.4f}")
    if "epoch" in checkpoint:
        print(f"训练轮数: {checkpoint['epoch']}")

    print("模型加载成功")
    return model


def grid_to_normalized_position(grid_x: int, grid_y: int, grid_size: int = 11) -> Tuple[float, float]:
    """
    将网格坐标转换为归一化坐标

    Args:
        grid_x: 网格X坐标 [0, grid_size-1]
        grid_y: 网格Y坐标 [0, grid_size-1]
        grid_size: 网格大小

    Returns:
        (x, y): 归一化坐标 [-1, 1]
    """
    x = grid_x / (grid_size - 1) * 2 - 1
    y = grid_y / (grid_size - 1) * 2 - 1
    return x, y


@torch.no_grad()
def inference_on_video(
    model: NitroGenActionParser,
    video_path: str,
    stage1_bbox: Optional[List[int]] = None,
    device: torch.device = torch.device("cpu"),
    image_size: int = 256,
    temporal_gap: int = 1,
    max_frames: Optional[int] = None
) -> Dict:
    """
    对单个视频进行推理

    Args:
        model: 训练好的模型
        video_path: 视频文件路径
        stage1_bbox: Stage 1检测的bbox [x1, y1, x2, y2]
        device: 设备
        image_size: 输入图像尺寸
        temporal_gap: 帧间间隔
        max_frames: 最大处理帧数

    Returns:
        推理结果字典
    """
    video_name = Path(video_path).stem

    # 创建数据集
    dataset = VideoFramePairDataset(
        video_paths=[video_path],
        stage1_results={video_name: {"bbox": stage1_bbox}} if stage1_bbox else {},
        image_size=image_size,
        temporal_gap=temporal_gap,
        max_frames=max_frames,
        default_bbox=None
    )

    if len(dataset) == 0:
        print(f"警告: 视频 {video_path} 没有帧")
        return {
            "video": video_path,
            "frames": [],
            "error": "No frames found"
        }

    print(f"处理视频: {video_name} ({len(dataset)} 帧对)")

    # 收集结果
    results = {
        "video": video_path,
        "total_frames": len(dataset),
        "frames": []
    }

    # 逐帧推理
    for idx in tqdm(range(len(dataset)), desc=f"推理 {video_name}"):
        sample = dataset[idx]

        # 获取数据
        frame_pair = sample["frame_pair"].unsqueeze(0).to(device)  # [1, 6, H, W]

        # 模型推理
        outputs = model(frame_pair)

        # 解析按键
        button_probs = outputs["button_probs"][0]  # [16]
        button_states = (button_probs > 0.5).cpu().numpy().astype(int).tolist()

        # 解析摇杆（空间池化 + argmax）
        left_joystick_pred = outputs["left_joystick"][0]  # [121, H/4, W/4]
        right_joystick_pred = outputs["right_joystick"][0]

        # 空间平均池化
        left_class_probs = left_joystick_pred.mean(dim=[1, 2])  # [121]
        right_class_probs = right_joystick_pred.mean(dim=[1, 2])

        # Argmax得到类别索引
        left_pred_class = left_class_probs.argmax().item()
        right_pred_class = right_class_probs.argmax().item()

        # 转换为网格坐标
        left_grid_x = left_pred_class % 11
        left_grid_y = left_pred_class // 11
        right_grid_x = right_pred_class % 11
        right_grid_y = right_pred_class // 11

        # 转换为归一化坐标
        left_x, left_y = grid_to_normalized_position(left_grid_x, left_grid_y)
        right_x, right_y = grid_to_normalized_position(right_grid_x, right_grid_y)

        # 记录结果
        frame_idx = sample["frame_idx"]
        if hasattr(frame_idx, 'item'):
            frame_idx = frame_idx.item()
        else:
            frame_idx = int(frame_idx)

        frame_result = {
            "frame_idx": frame_idx,
            "left_joystick": {
                "x": round(float(left_x), 4),
                "y": round(float(left_y), 4),
                "grid_x": int(left_grid_x),
                "grid_y": int(left_grid_y)
            },
            "right_joystick": {
                "x": round(float(right_x), 4),
                "y": round(float(right_y), 4),
                "grid_x": int(right_grid_x),
                "grid_y": int(right_grid_y)
            },
            "buttons": button_states,
            "button_probs": button_probs.cpu().numpy().tolist()
        }

        results["frames"].append(frame_result)

    return results


def save_results(results: Dict, output_path: str):
    """
    保存推理结果到JSON文件

    Args:
        results: 推理结果字典
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"结果已保存: {output_path}")


def batch_inference(
    model: NitroGenActionParser,
    video_dir: str,
    output_dir: str,
    device: torch.device,
    **kwargs
):
    """
    批量推理

    Args:
        model: 训练好的模型
        video_dir: 视频目录
        output_dir: 输出目录
        device: 设备
        **kwargs: 其他推理参数
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 查找所有视频文件
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    video_files = []

    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
        video_files.extend(video_dir.glob(f"*{ext.upper()}"))

    if not video_files:
        print(f"错误: 在 {video_dir} 中没有找到视频文件")
        return

    print(f"找到 {len(video_files)} 个视频文件")

    # 处理每个视频
    for video_path in tqdm(video_files, desc="批量推理"):
        print(f"\n处理: {video_path.name}")

        try:
            results = inference_on_video(
                model=model,
                video_path=str(video_path),
                device=device,
                **kwargs
            )

            # 保存结果
            output_path = output_dir / f"{video_path.stem}_actions.json"
            save_results(results, str(output_path))

        except Exception as e:
            print(f"错误处理 {video_path.name}: {e}")
            continue


def find_best_checkpoint(checkpoints_dir: str = "checkpoints") -> Optional[str]:
    """
    查找最新的best_model.pt

    Args:
        checkpoints_dir: checkpoints目录路径

    Returns:
        checkpoint路径，如果没找到返回None
    """
    checkpoints_dir = Path(checkpoints_dir)

    if not checkpoints_dir.exists():
        return None

    best_models = list(checkpoints_dir.glob("*/best_model.pt"))

    if not best_models:
        return None

    # 返回最新的
    return str(max(best_models, key=lambda p: p.stat().st_mtime))


def main():
    parser = argparse.ArgumentParser(
        description="NitroGen Stage 2 动作解析推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单个视频推理
  python inference.py --video test.mp4

  # 批量推理
  python inference.py --video_dir videos/

  # 指定模型和输出
  python inference.py --video test.mp4 --checkpoint model.pt --output results/
        """
    )

    # 输入选项
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--video", type=str, help="单个视频文件路径")
    input_group.add_argument("--video_dir", type=str, help="视频目录（批量处理）")

    # 模型选项
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="模型checkpoint路径（默认：自动查找最新的best_model.pt）")
    parser.add_argument("--device", type=str, default=None,
                        help="设备（cuda/cpu，默认：自动）")

    # 输出选项
    parser.add_argument("--output", type=str, default="output",
                        help="输出目录（默认：output/）")

    # 推理参数
    parser.add_argument("--image_size", type=int, default=256,
                        help="输入图像尺寸（默认：256）")
    parser.add_argument("--temporal_gap", type=int, default=1,
                        help="帧间间隔（默认：1）")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="最大处理帧数（默认：全部）")
    parser.add_argument("--bbox", type=int, nargs=4, default=None,
                        metavar=('X1', 'Y1', 'X2', 'Y2'),
                        help="手柄区域bbox [x1 y1 x2 y2]（可选）")

    args = parser.parse_args()

    # IDE模式：如果没有提供输入参数，使用默认值
    if not args.video and not args.video_dir:
        if len(sys.argv) <= 1:  # 没有命令行参数（IDE运行）
            print("[IDE模式] 检测到在IDE中运行，使用默认参数")
            # 检测工作目录
            cwd = os.getcwd()
            if "stage2" in cwd:
                # 如果在stage2目录中运行
                args.video_dir = "../../action/data/videos"
            elif "action" in cwd:
                # 如果在action目录中运行
                args.video_dir = "data/videos"
            else:
                # 如果在项目根目录或其他位置运行
                args.video_dir = "action/data/videos"
            print(f"  工作目录: {cwd}")
            print(f"  --video_dir {args.video_dir}")
            print(f"  --output {args.output}")
            print()

    # 验证必须有输入参数
    if not args.video and not args.video_dir:
        parser.error("必须提供 --video 或 --video_dir 参数")

    # 设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*60)
    print("NitroGen Stage 2 动作解析推理")
    print("="*60)
    print(f"设备: {device}")

    # 查找checkpoint
    if args.checkpoint is None:
        args.checkpoint = find_best_checkpoint()

    if args.checkpoint is None or not os.path.exists(args.checkpoint):
        print(f"\n错误: 找不到模型checkpoint")
        print(f"请先训练模型: python train.py")
        print(f"或指定checkpoint: --checkpoint path/to/model.pt")
        return

    # 加载模型
    model = load_model(args.checkpoint, device)

    # 执行推理
    if args.video:
        # 单个视频
        print(f"\n处理视频: {args.video}")
        results = inference_on_video(
            model=model,
            video_path=args.video,
            stage1_bbox=args.bbox,
            device=device,
            image_size=args.image_size,
            temporal_gap=args.temporal_gap,
            max_frames=args.max_frames
        )

        # 保存结果
        output_path = Path(args.output) / f"{Path(args.video).stem}_actions.json"
        save_results(results, str(output_path))

    elif args.video_dir:
        # 批量推理
        print(f"\n批量推理目录: {args.video_dir}")
        batch_inference(
            model=model,
            video_dir=args.video_dir,
            output_dir=args.output,
            device=device,
            image_size=args.image_size,
            temporal_gap=args.temporal_gap,
            max_frames=args.max_frames
        )

    print("\n" + "="*60)
    print("推理完成")
    print("="*60)


if __name__ == "__main__":
    main()
