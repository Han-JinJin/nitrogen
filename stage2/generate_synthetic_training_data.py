"""
合成训练数据生成器
按照NitroGen论文: 使用Open Joystick Display/Input Overlay生成8M标注帧

功能:
    1. 加载手柄模板
    2. 随机生成按键状态和摇杆位置
    3. 渲染手柄图像
    4. 保存标注数据
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import random
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# 按键名称定义
BUTTON_NAMES = [
    "dpad_up", "dpad_down", "dpad_left", "dpad_right",
    "face_cross", "face_circle", "face_square", "face_triangle",
    "shoulder_l1", "shoulder_r1", "shoulder_l2", "shoulder_r2",
    "stick_l3", "stick_r3",
    "options", "share"
]


@dataclass
class ControllerState:
    """手柄状态"""
    # 摇杆位置 [-1, 1]
    left_joystick_x: float
    left_joystick_y: float
    right_joystick_x: float
    right_joystick_y: float

    # 按键状态 [16]
    buttons: List[bool]


class SyntheticDataGenerator:
    """
    合成训练数据生成器
    """

    def __init__(
        self,
        output_dir: str,
        templates_dir: str = "templates",
        image_size: int = 256,
        num_variants: int = 5
    ):
        """
        Args:
            output_dir: 输出目录
            templates_dir: 手柄模板目录
            image_size: 图像尺寸
            num_variants: 每个模板的变体数量 (不同透明度、尺寸等)
        """
        self.output_dir = Path(output_dir)
        self.templates_dir = Path(templates_dir)
        self.image_size = image_size
        self.num_variants = num_variants

        # 创建输出目录
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        # 加载手柄模板
        self.templates = self._load_templates()

        print(f"Loaded {len(self.templates)} controller templates")

    def _load_templates(self) -> List[Dict]:
        """加载手柄模板"""
        templates = []

        # 如果有预渲染的模板图片，加载它们
        if self.templates_dir.exists():
            for template_file in self.templates_dir.glob("*.png"):
                templates.append({
                    "name": template_file.stem,
                    "image": cv2.imread(str(template_file), cv2.IMREAD_UNCHANGED),
                    "type": "pre_rendered"
                })

        # 如果没有模板，创建程序化模板
        if not templates:
            templates = self._create_programmatic_templates()

        return templates

    def _create_programmatic_templates(self) -> List[Dict]:
        """创建程序化手柄模板"""
        templates = []

        # PS4 DualShock 风格
        ps4_base = self._create_ps4_template()
        templates.append({
            "name": "ps4_standard",
            "image": ps4_base,
            "type": "ps4",
            "layout": self._get_ps4_layout()
        })

        # Xbox 风格
        xbox_base = self._create_xbox_template()
        templates.append({
            "name": "xbox_standard",
            "image": xbox_base,
            "type": "xbox",
            "layout": self._get_xbox_layout()
        })

        return templates

    def _create_ps4_template(self) -> np.ndarray:
        """创建PS4风格手柄模板"""
        size = self.image_size
        img = np.ones((size, size, 4), dtype=np.uint8) * 255

        # 创建透明背景
        img[:, :, 3] = 0  # Alpha通道

        # 手柄主体 (半透明黑色)
        overlay = img.copy()
        cv2.rectangle(overlay, (20, 40), (size-20, size-30), (30, 30, 30, 200), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

        # 绘制按键位置
        # 左摇杆
        cv2.circle(img, (70, 160), 25, (50, 50, 50, 220), -1)
        cv2.circle(img, (70, 160), 15, (100, 100, 100, 255), 2)

        # 右摇杆
        cv2.circle(img, (170, 180), 25, (50, 50, 50, 220), -1)
        cv2.circle(img, (170, 180), 15, (100, 100, 100, 255), 2)

        # 方向键
        dpad_center = (70, 90)
        cv2.rectangle(img, (55, 75), (85, 105), (40, 40, 40, 200), -1)

        # 面键 (三角形、圆形、X、方形)
        face_buttons = [(160, 70), (180, 90), (140, 90), (160, 110)]
        labels = ["△", "○", "×", "□"]
        for (x, y), label in zip(face_buttons, labels):
            cv2.circle(img, (x, y), 12, (40, 40, 40, 200), -1)
            # 添加文字需要PIL
            # cv2.putText(img, label, (x-5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

        # 肩键
        cv2.rectangle(img, (40, 20), (80, 35), (40, 40, 40, 180), -1)  # L1
        cv2.rectangle(img, (size-80, 20), (size-40, 35), (40, 40, 40, 180), -1)  # R1

        # 功能键
        cv2.rectangle(img, (110, 60), (130, 70), (40, 40, 40, 180), -1)  # Share
        cv2.rectangle(img, (size-130, 60), (size-110, 70), (40, 40, 40, 180), -1)  # Options

        return img

    def _create_xbox_template(self) -> np.ndarray:
        """创建Xbox风格手柄模板"""
        size = self.image_size
        img = np.ones((size, size, 4), dtype=np.uint8) * 255
        img[:, :, 3] = 0

        # 手柄主体
        overlay = img.copy()
        cv2.rectangle(overlay, (20, 40), (size-20, size-30), (20, 20, 20, 200), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

        # 左摇杆
        cv2.circle(img, (70, 160), 25, (50, 50, 50, 220), -1)
        cv2.circle(img, (70, 160), 15, (100, 100, 100, 255), 2)

        # 右摇杆
        cv2.circle(img, (170, 180), 25, (50, 50, 50, 220), -1)
        cv2.circle(img, (170, 180), 15, (100, 100, 100, 255), 2)

        # 方向键
        cv2.rectangle(img, (55, 75), (85, 105), (30, 30, 30, 200), -1)

        # 面键 (A, B, X, Y)
        face_buttons = [(160, 110), (180, 90), (140, 90), (160, 70)]
        colors = [(0, 150, 0), (200, 0, 0), (0, 100, 255), (255, 255, 0)]
        for (x, y), color in zip(face_buttons, colors):
            cv2.circle(img, (x, y), 12, (40, 40, 40, 200), -1)

        # 肩键
        cv2.rectangle(img, (40, 20), (80, 35), (30, 30, 30, 180), -1)  # LB
        cv2.rectangle(img, (size-80, 20), (size-40, 35), (30, 30, 30, 180), -1)  # RB

        return img

    def _get_ps4_layout(self) -> Dict:
        """获取PS4按键布局"""
        size = self.image_size  # 使用实例变量
        return {
            "left_joystick_center": (70, 160),
            "right_joystick_center": (170, 180),
            "dpad_center": (70, 90),
            "face_buttons": {
                "triangle": (160, 70),
                "circle": (180, 90),
                "cross": (160, 110),
                "square": (140, 90)
            },
            "shoulders": {
                "L1": (40, 20, 80, 35),
                "R1": (size-80, 20, size-40, 35)
            }
        }

    def _get_xbox_layout(self) -> Dict:
        """获取Xbox按键布局"""
        size = self.image_size  # 使用实例变量
        return {
            "left_joystick_center": (70, 160),
            "right_joystick_center": (170, 180),
            "dpad_center": (70, 90),
            "face_buttons": {
                "Y": (160, 70),
                "B": (180, 90),
                "A": (160, 110),
                "X": (140, 90)
            },
            "shoulders": {
                "L1": (40, 20, 80, 35),  # 统一使用L1/R1键名
                "R1": (size-80, 20, size-40, 35)
            }
        }

    def generate_random_state(self) -> ControllerState:
        """生成随机手柄状态"""
        # 随机摇杆位置 (偏向中心分布)
        def random_joystick_pos():
            # 50%概率在中心附近
            if random.random() < 0.5:
                return random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3)
            else:
                return random.uniform(-1, 1), random.uniform(-1, 1)

        lx, ly = random_joystick_pos()
        rx, ry = random_joystick_pos()

        # 随机按键状态 (稀疏激活)
        buttons = [random.random() < 0.15 for _ in range(16)]

        return ControllerState(lx, ly, rx, ry, buttons)

    def render_state(
        self,
        template: Dict,
        state: ControllerState,
        opacity: float = 0.8,
        scale: float = 1.0,
        add_compression_artifacts: bool = False
    ) -> np.ndarray:
        """
        渲染手柄状态

        Args:
            template: 手柄模板
            state: 手柄状态
            opacity: 透明度
            scale: 缩放
            add_compression_artifacts: 是否添加压缩伪影

        Returns:
            渲染后的图像 [H, W, 3]
        """
        # 复制模板
        img = template["image"].copy()

        # 调整透明度
        img[:, :, 3] = (img[:, :, 3] * opacity).astype(np.uint8)

        # 缩放
        if scale != 1.0:
            new_size = (int(self.image_size * scale), int(self.image_size * scale))
            img = cv2.resize(img, new_size)

            # 居中粘贴
            canvas = np.ones((self.image_size, self.image_size, 4), dtype=np.uint8) * 255
            canvas[:, :, 3] = 0
            offset_x = (self.image_size - new_size[0]) // 2
            offset_y = (self.image_size - new_size[1]) // 2
            canvas[offset_y:offset_y+new_size[1], offset_x:offset_x+new_size[0]] = img
            img = canvas

        # 渲染摇杆位置
        layout = template.get("layout", self._get_ps4_layout())
        self._render_joystick(img, layout["left_joystick_center"], state.left_joystick_x, state.left_joystick_y)
        self._render_joystick(img, layout["right_joystick_center"], state.right_joystick_x, state.right_joystick_y)

        # 渲染按键状态
        self._render_buttons(img, layout, state.buttons)

        # 转换为RGB
        if img.shape[2] == 4:
            # 使用alpha混合
            background = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 240
            alpha = img[:, :, 3:4].astype(np.float32) / 255.0
            img_rgb = (img[:, :, :3].astype(np.float32) * alpha +
                      background.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        else:
            img_rgb = img[:, :, :3]

        # 添加压缩伪影 (模拟YouTube视频)
        if add_compression_artifacts:
            # JPEG压缩
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(60, 85)]
            _, encoded = cv2.imencode('.jpg', img_rgb, encode_param)
            img_rgb = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        return img_rgb

    def _render_joystick(self, img: np.ndarray, center: Tuple[int, int], x: float, y: float):
        """渲染摇杆位置"""
        cx, cy = center
        radius = 20

        # 计算摇杆位置 (基于网格偏移)
        offset_x = int(x * radius * 0.8)
        offset_y = int(y * radius * 0.8)

        # 绘制外圈
        cv2.circle(img, (cx, cy), radius, (40, 40, 40, 200), 2)

        # 绘制摇杆帽 (偏移)
        stick_x = cx + offset_x
        stick_y = cy + offset_y
        cv2.circle(img, (stick_x, stick_y), 12, (180, 180, 180, 255), -1)
        cv2.circle(img, (stick_x, stick_y), 12, (100, 100, 100, 255), 1)

    def _render_buttons(self, img: np.ndarray, layout: Dict, buttons: List[bool]):
        """渲染按键状态"""
        # 按下的按键高亮显示
        pressed_color = (0, 255, 0, 200)

        # 方向键
        if buttons[0]:  # dpad_up
            cv2.circle(img, (layout["dpad_center"][0], layout["dpad_center"][1] - 10), 5, pressed_color, -1)
        if buttons[1]:  # dpad_down
            cv2.circle(img, (layout["dpad_center"][0], layout["dpad_center"][1] + 10), 5, pressed_color, -1)
        if buttons[2]:  # dpad_left
            cv2.circle(img, (layout["dpad_center"][0] - 10, layout["dpad_center"][1]), 5, pressed_color, -1)
        if buttons[3]:  # dpad_right
            cv2.circle(img, (layout["dpad_center"][0] + 10, layout["dpad_center"][1]), 5, pressed_color, -1)

        # 面键
        for i, (name, pos) in enumerate(layout["face_buttons"].items()):
            if buttons[4 + i]:
                cv2.circle(img, pos, 8, pressed_color, -1)

        # 肩键
        if buttons[8]:  # L1
            x1, y1, x2, y2 = layout["shoulders"]["L1"]
            cv2.rectangle(img, (x1, y1), (x2, y2), pressed_color, -1)
        if buttons[9]:  # R1
            x1, y1, x2, y2 = layout["shoulders"]["R1"]
            cv2.rectangle(img, (x1, y1), (x2, y2), pressed_color, -1)

    def generate_dataset(
        self,
        num_samples: int = 100000,
        samples_per_video: int = 100
    ):
        """
        生成训练数据集

        Args:
            num_samples: 总样本数
            samples_per_video: 每个视频的样本数
        """
        print(f"Generating {num_samples} synthetic training samples...")

        # 创建视频目录
        num_videos = num_samples // samples_per_video + 1
        annotations = {}

        for video_idx in tqdm(range(num_videos), desc="Generating videos"):
            video_id = f"synthetic_{video_idx:06d}"
            video_dir = self.frames_dir / video_id
            video_dir.mkdir(exist_ok=True)

            # 随机选择模板
            template = random.choice(self.templates)

            # 生成该视频的样本
            video_annotations = {
                "template": template["name"],
                "frames": []
            }

            for frame_idx in range(samples_per_video):
                # 生成随机状态
                state = self.generate_random_state()

                # 渲染
                opacity = random.uniform(0.6, 1.0)
                scale = random.uniform(0.8, 1.0)
                add_artifacts = random.random() < 0.3

                img = self.render_state(
                    template,
                    state,
                    opacity=opacity,
                    scale=scale,
                    add_compression_artifacts=add_artifacts
                )

                # 保存图像
                frame_path = video_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), img)

                # 保存标注
                frame_ann = {
                    "left_joystick": {"x": state.left_joystick_x, "y": state.left_joystick_y},
                    "right_joystick": {"x": state.right_joystick_x, "y": state.right_joystick_y},
                    "buttons": [int(b) for b in state.buttons]
                }
                video_annotations["frames"].append(frame_ann)

            annotations[video_id] = video_annotations

            # 定期保存标注
            if (video_idx + 1) % 100 == 0:
                ann_path = self.output_dir / "annotations_temp.json"
                with open(ann_path, 'w', encoding='utf-8') as f:
                    json.dump(annotations, f, indent=2)

        # 保存最终标注
        ann_path = self.output_dir / "annotations.json"
        with open(ann_path, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2)

        print(f"\nDataset generated!")
        print(f"  Total samples: {num_samples}")
        print(f"  Total videos: {num_videos}")
        print(f"  Frames directory: {self.frames_dir}")
        print(f"  Annotations: {ann_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic training data for NitroGen action parser")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to generate (default: 10000,论文: 8000000)"
    )
    parser.add_argument(
        "--samples_per_video",
        type=int,
        default=100,
        help="Number of samples per video (default: 100)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="synthetic_data",
        help="Output directory (default: synthetic_data)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image size (default: 256)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Synthetic Training Data Generator")
    print("  NitroGen Paper Implementation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Number of samples: {args.num_samples:,}")
    print(f"  Samples per video: {args.samples_per_video}")
    print(f"  Image size: {args.image_size}x{args.image_size}")

    if args.num_samples < 1000:
        print(f"\n[WARNING] Generating very small dataset ({args.num_samples} samples)")
        print("  This is only for testing. For real training, use at least 10,000 samples")
    elif args.num_samples < 100000:
        print(f"\n[INFO] Generating small dataset ({args.num_samples:,} samples)")
        print("  For production performance, consider generating 8M samples as in the paper")

    # 创建生成器
    generator = SyntheticDataGenerator(
        output_dir=args.output_dir,
        image_size=args.image_size
    )

    # 生成数据集
    generator.generate_dataset(
        num_samples=args.num_samples,
        samples_per_video=args.samples_per_video
    )

    print(f"\n[SUCCESS] Dataset generated successfully!")
    print(f"   Location: {args.output_dir}/")
    print(f"   Total samples: {args.num_samples:,}")
    print(f"\nNext steps:")
    print(f"   1. Train model: python train.py")
    print(f"   2. Monitor training: tensorboard --logdir=logs")


if __name__ == "__main__":
    main()
