"""
两帧拼接数据集加载器
按照NitroGen论文实现: 加载连续帧对用于分割模型训练

增强功能:
    - 颜色抖动 (Color Jittering)
    - 随机水平翻转 (Random Horizontal Flip)
    - 随机亮度/对比度调整
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class FramePairDataset(Dataset):
    """
    连续帧对数据集（支持数据增强）

    输入: 两帧拼接的RGB图像
    输出:
        - left_joystick_mask: 11×11网格标注
        - right_joystick_mask: 11×11网格标注
        - button_states: 按键二值状态
    """

    def __init__(
        self,
        data_dir: str,
        annotations_path: str,
        image_size: int = 256,
        joystick_grid_size: int = 11,
        temporal_gap: int = 1,
        enable_augmentation: bool = True,
        augment_prob: float = 0.5
    ):
        """
        Args:
            data_dir: 视频帧目录
            annotations_path: 标注JSON文件路径
            image_size: 输入图像尺寸
            joystick_grid_size: 摇杆网格大小
            temporal_gap: 帧间间隔
            enable_augmentation: 是否启用数据增强
            augment_prob: 数据增强应用概率
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.joystick_grid_size = joystick_grid_size
        self.temporal_gap = temporal_gap
        self.enable_augmentation = enable_augmentation
        self.augment_prob = augment_prob

        # 加载标注
        with open(annotations_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        # 构建样本列表
        self.samples = self._build_samples()

        aug_status = "enabled" if enable_augmentation else "disabled"
        print(f"Loaded {len(self.samples)} frame pairs from {len(self.annotations)} videos")
        print(f"Data augmentation: {aug_status}")

    def _build_samples(self) -> List[Dict]:
        """构建帧对样本列表"""
        samples = []

        for video_id, video_anns in self.annotations.items():
            frames_dir = self.data_dir / video_id
            if not frames_dir.exists():
                continue

            # 获取所有帧
            frame_files = sorted(frames_dir.glob("frame_*.jpg"))
            if len(frame_files) < 2:
                continue

            # 为每一对连续帧创建样本
            for i in range(len(frame_files) - self.temporal_gap):
                frame1_path = frame_files[i]
                frame2_path = frame_files[i + self.temporal_gap]

                # 获取该时间步的标注
                frame_idx = i
                if frame_idx < len(video_anns["frames"]):
                    ann = video_anns["frames"][frame_idx]
                    samples.append({
                        "video_id": video_id,
                        "frame1_path": str(frame1_path),
                        "frame2_path": str(frame2_path),
                        "frame_idx": frame_idx,
                        "annotation": ann
                    })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # 加载两帧
        frame1 = cv2.imread(sample["frame1_path"])
        frame2 = cv2.imread(sample["frame2_path"])

        if frame1 is None or frame2 is None:
            raise ValueError(f"Failed to load frames: {sample}")

        # 调整尺寸
        frame1 = cv2.resize(frame1, (self.image_size, self.image_size))
        frame2 = cv2.resize(frame2, (self.image_size, self.image_size))

        # 🔥 数据增强（仅在训练时应用）
        if self.enable_augmentation and np.random.random() < self.augment_prob:
            frame1, frame2 = self._apply_augmentation(frame1, frame2)

        # 归一化到 [0, 1]
        frame1 = frame1.astype(np.float32) / 255.0
        frame2 = frame2.astype(np.float32) / 255.0

        # 拼接两帧: [H, W, 6]
        frame_pair = np.concatenate([frame1, frame2], axis=2)

        # 转换为 PyTorch 格式: [6, H, W]
        frame_pair = torch.from_numpy(frame_pair).permute(2, 0, 1)

        # 立即释放临时数组
        del frame1, frame2

        # 处理标注
        ann = sample["annotation"]

        # 摇杆位置转换为网格
        left_joystick_grid = self._position_to_grid(
            ann["left_joystick"]["x"],
            ann["left_joystick"]["y"]
        )

        right_joystick_grid = self._position_to_grid(
            ann["right_joystick"]["x"],
            ann["right_joystick"]["y"]
        )

        # 创建摇杆网格分类标签
        mask_size = self.image_size // 32

        # 将11×11网格坐标转换为单一类别索引（0-120）
        left_class_idx = left_joystick_grid[1] * self.joystick_grid_size + left_joystick_grid[0]
        right_class_idx = right_joystick_grid[1] * self.joystick_grid_size + right_joystick_grid[0]

        # 创建标签：所有位置都是同一个类别（摇杆的网格位置）
        left_mask = np.full((mask_size, mask_size), left_class_idx, dtype=np.int64)
        right_mask = np.full((mask_size, mask_size), right_class_idx, dtype=np.int64)

        # 🔥 保留原始归一化坐标用于评估
        left_joystick_normalized = np.array([
            ann["left_joystick"]["x"],
            ann["left_joystick"]["y"]
        ], dtype=np.float32)
        right_joystick_normalized = np.array([
            ann["right_joystick"]["x"],
            ann["right_joystick"]["y"]
        ], dtype=np.float32)

        # 按键状态
        button_states = torch.tensor(ann["buttons"], dtype=torch.float32)

        return {
            "frame_pair": frame_pair,  # [6, H, W]
            "left_joystick": torch.from_numpy(left_mask),  # [H/4, W/4] 分割mask
            "right_joystick": torch.from_numpy(right_mask),  # [H/4, W/4] 分割mask
            "left_joystick_pos": torch.from_numpy(left_joystick_normalized),  # [2] 归一化坐标
            "right_joystick_pos": torch.from_numpy(right_joystick_normalized),  # [2] 归一化坐标
            "buttons": button_states,  # [num_buttons]
            "video_id": sample["video_id"],
            "frame_idx": sample["frame_idx"]
        }

    def _apply_augmentation(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用数据增强到两帧

        注意：对两帧应用相同的变换，保持时序一致性

        Args:
            frame1, frame2: [H, W, 3] BGR格式的帧

        Returns:
            增强后的 frame1, frame2
        """
        # 1. 随机水平翻转（两帧一起翻转）
        if np.random.random() < 0.5:
            frame1 = cv2.flip(frame1, 1)
            frame2 = cv2.flip(frame2, 1)

        # 2. 颜色抖动（两帧应用相同的参数）
        if np.random.random() < 0.5:
            # 亮度调整
            brightness_delta = np.random.uniform(-0.1, 0.1)
            frame1 = np.clip(frame1 + brightness_delta * 255, 0, 255).astype(np.uint8)
            frame2 = np.clip(frame2 + brightness_delta * 255, 0, 255).astype(np.uint8)

        # 3. 对比度调整（两帧应用相同的参数）
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            frame1 = np.clip(frame1 * contrast_factor, 0, 255).astype(np.uint8)
            frame2 = np.clip(frame2 * contrast_factor, 0, 255).astype(np.uint8)

        # 4. 饱和度调整（两帧应用相同的参数）
        if np.random.random() < 0.5:
            # 转换到HSV
            frame1_hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV).astype(np.float32)
            frame2_hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV).astype(np.float32)

            # 调整饱和度
            saturation_factor = np.random.uniform(0.8, 1.2)
            frame1_hsv[:, :, 1] = np.clip(frame1_hsv[:, :, 1] * saturation_factor, 0, 255)
            frame2_hsv[:, :, 1] = np.clip(frame2_hsv[:, :, 1] * saturation_factor, 0, 255)

            # 转回BGR
            frame1 = cv2.cvtColor(frame1_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            frame2 = cv2.cvtColor(frame2_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 5. 轻微高斯噪声（两帧独立）
        if np.random.random() < 0.3:
            noise_sigma = np.random.uniform(0.001, 0.005)
            noise1 = np.random.normal(0, noise_sigma * 255, frame1.shape)
            noise2 = np.random.normal(0, noise_sigma * 255, frame2.shape)
            frame1 = np.clip(frame1 + noise1, 0, 255).astype(np.uint8)
            frame2 = np.clip(frame2 + noise2, 0, 255).astype(np.uint8)

        return frame1, frame2

    def _position_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        将归一化位置 [-1, 1] 转换为网格坐标 [0-10]

        Args:
            x, y: 归一化坐标 [-1, 1]

        Returns:
            (grid_x, grid_y): 网格坐标 [0-10]
        """
        # 将 [-1, 1] 映射到 [0, 10]
        grid_x = int((x + 1) / 2 * (self.joystick_grid_size - 1))
        grid_y = int((y + 1) / 2 * (self.joystick_grid_size - 1))

        # 限制范围
        grid_x = np.clip(grid_x, 0, self.joystick_grid_size - 1)
        grid_y = np.clip(grid_y, 0, self.joystick_grid_size - 1)

        return grid_x, grid_y


class VideoFramePairDataset(Dataset):
    """
    直接从视频加载帧对的轻量级数据集

    用于推理或小规模训练
    """

    def __init__(
        self,
        video_paths: List[str],
        stage1_results: Dict,
        image_size: int = 256,
        temporal_gap: int = 1,
        max_frames: Optional[int] = None,
        default_bbox: Optional[List[int]] = None
    ):
        """
        Args:
            video_paths: 视频文件路径列表
            stage1_results: Stage 1检测的bbox结果
            image_size: 输入图像尺寸
            temporal_gap: 帧间间隔
            max_frames: 最大帧数 (None表示全部)
            default_bbox: 当stage1_results中找不到对应视频时使用的默认bbox
        """
        self.video_paths = video_paths
        self.stage1_results = stage1_results
        self.image_size = image_size
        self.temporal_gap = temporal_gap
        self.max_frames = max_frames
        self.default_bbox = default_bbox

        # 预加载所有帧
        self.frames_cache = self._load_frames()

    def _load_frames(self) -> List[Dict]:
        """加载视频帧"""
        all_frames = []

        for video_idx, video_path in enumerate(self.video_paths):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open {video_path}")
                continue

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

                if self.max_frames and len(frames) >= self.max_frames:
                    break

            cap.release()

            # 跳过没有帧的视频
            if not frames:
                print(f"Warning: No frames found in {video_path}")
                continue

            # 获取bbox
            video_prefix = Path(video_path).stem
            if video_prefix in self.stage1_results:
                bbox = self.stage1_results[video_prefix]["bbox"]
            elif self.default_bbox is not None:
                bbox = self.default_bbox
            else:
                # 使用整个帧（从第一帧获取尺寸）
                bbox = [0, 0, frames[0].shape[1], frames[0].shape[0]]

            all_frames.append({
                "video_idx": video_idx,
                "video_path": video_path,
                "frames": frames,
                "bbox": bbox
            })

        return all_frames

    def __len__(self) -> int:
        total = 0
        for video_data in self.frames_cache:
            num_frames = len(video_data["frames"])
            if num_frames >= self.temporal_gap:
                total += num_frames - self.temporal_gap
        return total

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 找到对应的视频和帧索引
        cumulative = 0
        video_data = None
        frame_idx = None

        for vd in self.frames_cache:
            num_pairs = len(vd["frames"]) - self.temporal_gap
            if cumulative + num_pairs > idx:
                video_data = vd
                frame_idx = idx - cumulative
                break
            cumulative += num_pairs

        if video_data is None:
            raise IndexError("Index out of range")

        frames = video_data["frames"]
        bbox = video_data["bbox"]

        # 获取两帧
        frame1 = frames[frame_idx]
        frame2 = frames[frame_idx + self.temporal_gap]

        # 裁剪bbox
        x1, y1, x2, y2 = bbox
        frame1 = frame1[y1:y2, x1:x2]
        frame2 = frame2[y1:y2, x1:x2]

        # 调整尺寸
        frame1 = cv2.resize(frame1, (self.image_size, self.image_size))
        frame2 = cv2.resize(frame2, (self.image_size, self.image_size))

        # 归一化
        frame1 = frame1.astype(np.float32) / 255.0
        frame2 = frame2.astype(np.float32) / 255.0

        # 拼接
        frame_pair = np.concatenate([frame1, frame2], axis=2)
        frame_pair = torch.from_numpy(frame_pair).permute(2, 0, 1)

        # 🔥 立即释放临时数组
        del frame1, frame2
        import gc
        gc.collect()

        return {
            "frame_pair": frame_pair,
            "video_idx": video_data["video_idx"],
            "frame_idx": frame_idx,
            "video_path": video_data["video_path"]
        }


def create_train_val_dataloaders(
    data_dir: str,
    train_annotations: str,
    val_annotations: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 256,
    enable_augmentation: bool = True,
    augment_prob: float = 0.5
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器

    Args:
        data_dir: 数据目录
        train_annotations: 训练标注文件
        val_annotations: 验证标注文件
        batch_size: 批次大小
        num_workers: 数据加载线程数
        image_size: 图像尺寸
        enable_augmentation: 是否对训练集启用数据增强
        augment_prob: 数据增强应用概率

    Returns:
        (train_loader, val_loader)
    """
    # 训练集：启用数据增强
    train_dataset = FramePairDataset(
        data_dir=data_dir,
        annotations_path=train_annotations,
        image_size=image_size,
        enable_augmentation=enable_augmentation,
        augment_prob=augment_prob
    )

    # 验证集：不启用数据增强
    val_dataset = FramePairDataset(
        data_dir=data_dir,
        annotations_path=val_annotations,
        image_size=image_size,
        enable_augmentation=False,  # 验证集不使用增强
        augment_prob=0.0
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # 禁用pin_memory减少内存压力
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False  # 禁用pin_memory减少内存压力
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # 测试数据集
    print("Testing dataset...")

    import os

    # 检查是否存在synthetic_data
    synthetic_data_dir = "synthetic_data"
    train_ann_path = os.path.join(synthetic_data_dir, "train_annotations.json")

    if os.path.exists(train_ann_path):
        print(f"Using synthetic data from {synthetic_data_dir}")

        dataset = FramePairDataset(
            data_dir=os.path.join(synthetic_data_dir, "frames"),
            annotations_path=train_ann_path,
            image_size=256,
            enable_augmentation=True
        )
        print(f"Loaded {len(dataset)} frame pairs")
        print(f"Data augmentation: {dataset.enable_augmentation}")

        # 测试加载一个样本
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample structure:")
            for key, value in sample.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  {key}: {type(value).__name__}")
    else:
        print(f"Error: {train_ann_path} not found")
        print("Please run: python generate_synthetic_training_data.py --num_samples 100")
