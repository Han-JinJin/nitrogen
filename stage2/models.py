"""
SegFormer-based Action Parsing Model (Fixed Version)
按照NitroGen论文实现: Stage 2动作解析的分割模型

核心特性:
    1. 输入两帧拼接图像 (2*3=6通道)
    2. 输出摇杆11×11网格位置 + 按键二分类
    3. 使用SegFormer encoder + decoder
"""

import os
# 强制离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 获取当前文件所在目录，构建本地模型绝对路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOCAL_MODEL = os.path.join(CURRENT_DIR, "segformer-b5-local")

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

try:
    from transformers import SegformerForSemanticSegmentation
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class NitroGenActionParser(nn.Module):
    """
    NitroGen动作解析模型

    输入: 两帧拼接的RGB图像 [B, 6, H, W]
    输出:
        - left_joystick_mask: [B, 121, H/4, W/4] 11×11=121类
        - right_joystick_mask: [B, 121, H/4, W/4]
        - button_logits: [B, num_buttons, H/4, W/4]
    """

    def __init__(
        self,
        num_buttons: int = 16,
        joystick_grid_size: int = 11,
        hidden_size: int = 768,
        backbone: str = None,  # 默认使用本地模型
        use_pretrained: bool = True
    ):
        # 如果没有指定 backbone，使用本地模型路径
        if backbone is None:
            backbone = DEFAULT_LOCAL_MODEL
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required. Install with: pip install transformers")

        self.num_buttons = num_buttons
        self.joystick_grid_size = joystick_grid_size
        self.num_joystick_classes = joystick_grid_size * joystick_grid_size

        # 加载预训练SegFormer
        if use_pretrained:
            try:
                print(f"Loading pretrained SegFormer from {backbone}...")
                self.backbone = SegformerForSemanticSegmentation.from_pretrained(
                    backbone,
                    num_labels=self.num_joystick_classes + num_buttons,
                    local_files_only=True,
                    ignore_mismatched_sizes=True
                )

                # 保存原始卷积层权重
                original_conv = self.backbone.segformer.encoder.patch_embeddings[0].proj
                original_weight = original_conv.weight.clone()
                original_bias = original_conv.bias.clone() if original_conv.bias is not None else None

                # 创建新的6通道卷积层
                new_conv = nn.Conv2d(
                    6,  # 2帧 × 3通道
                    original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=original_bias is not None  # 保持原有bias配置
                )

                # 初始化权重: 复制预训练权重到前后3个通道
                with torch.no_grad():
                    # 复制到前3个通道 [out_channels, 3, H, W]
                    new_conv.weight[:, :3] = original_weight
                    # 复制到后3个通道 [out_channels, 3, H, W]
                    new_conv.weight[:, 3:] = original_weight
                    # 复制偏置
                    if original_bias is not None and new_conv.bias is not None:
                        new_conv.bias.copy_(original_bias)

                # 替换原卷积层
                self.backbone.segformer.encoder.patch_embeddings[0].proj = new_conv
                print("Successfully adapted pretrained weights for 6-channel input")

            except Exception as e:
                print(f"Warning: Failed to load pretrained backbone: {e}")
                print("Using random initialization instead")
                use_pretrained = False

        # 如果预训练加载失败或不需要预训练，创建随机初始化的模型
        if not use_pretrained:
            print(f"Creating randomly initialized SegFormer model...")
            # 直接从本地 JSON 文件加载配置
            import json
            config_path = os.path.join(backbone, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"Config file not found: {config_path}\n"
                    f"Please ensure {backbone}/config.json exists"
                )

            with open(config_path, 'r') as f:
                config_dict = json.load(f)

            # 更新类别数
            config_dict['num_labels'] = self.num_joystick_classes + num_buttons

            # 创建配置对象
            config = SegformerForSemanticSegmentation.config_class(**config_dict)
            self.backbone = SegformerForSemanticSegmentation(config)

            # 修改第一层卷积为6通道
            original_conv = self.backbone.segformer.encoder.patch_embeddings[0].proj
            original_weight = original_conv.weight.clone()
            original_bias = original_conv.bias.clone() if original_conv.bias is not None else None

            new_conv = nn.Conv2d(
                6,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_bias is not None
            )

            with torch.no_grad():
                new_conv.weight[:, :3] = original_weight
                new_conv.weight[:, 3:] = original_weight
                if original_bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(original_bias)

            self.backbone.segformer.encoder.patch_embeddings[0].proj = new_conv
            print("Created 6-channel input model with random initialization")

        # 获取实际的hidden_size（Segformer使用hidden_sizes数组，取最后一个）
        hidden_size = self.backbone.segformer.config.hidden_sizes[-1]
        print(f"Model hidden_size: {hidden_size}")

        # 分离的解码头（使用实际的hidden_size）
        self.joystick_decoder = nn.Sequential(
            nn.Conv2d(hidden_size, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 2 * self.num_joystick_classes, 1)  # 左右摇杆
        )

        self.button_decoder = nn.Sequential(
            nn.Conv2d(hidden_size, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_buttons, 1)
        )

    def forward(
        self,
        frame_pairs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            frame_pairs: [B, 6, H, W] 两帧拼接

        Returns:
            Dict包含:
                - left_joystick: [B, 121, H/4, W/4]
                - right_joystick: [B, 121, H/4, W/4]
                - buttons: [B, num_buttons, H/4, W/4]
                - button_probs: [B, num_buttons]
        """
        # 获取encoder特征
        outputs = self.backbone.segformer(
            frame_pairs,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )

        # 获取最后一个encoder层的特征
        encoder_features = outputs.last_hidden_state  # [B, hidden_size, H/4, W/4]

        # 解码
        joystick_logits = self.joystick_decoder(encoder_features)  # [B, 242, H/4, W/4]
        button_logits = self.button_decoder(encoder_features)  # [B, num_buttons, H/4, W/4]

        # 分离左右摇杆
        left_joystick = joystick_logits[:, :self.num_joystick_classes]  # [B, 121, H/4, W/4]
        right_joystick = joystick_logits[:, self.num_joystick_classes:]  # [B, 121, H/4, W/4]

        # 按键全局平均池化得到二分类
        button_probs = torch.sigmoid(button_logits.mean(dim=[2, 3]))  # [B, num_buttons]

        return {
            "left_joystick": left_joystick,
            "right_joystick": right_joystick,
            "buttons": button_logits,
            "button_probs": button_probs
        }


class ActionParsingLoss(nn.Module):
    """
    动作解析损失函数

    组合:
        1. 摇杆分割损失 (交叉熵)
        2. 按键分类损失 (二元交叉熵)
    """

    def __init__(
        self,
        joystick_weight: float = 1.0,
        button_weight: float = 2.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.joystick_weight = joystick_weight
        self.button_weight = button_weight
        self.class_weights = class_weights

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算损失

        Args:
            predictions: 模型输出
            targets: Ground truth
                - left_joystick: [B, H, W] 类别索引 0-120
                - right_joystick: [B, H, W]
                - buttons: [B, num_buttons] 二值
        """
        # 确保targets的形状与predictions匹配
        # predictions["left_joystick"]: [B, 121, H, W]
        # targets["left_joystick"]: [B, H, W]

        # 摇杆分割损失
        left_joystick_loss = F.cross_entropy(
            predictions["left_joystick"],
            targets["left_joystick"].long(),
            weight=self.class_weights
        )

        right_joystick_loss = F.cross_entropy(
            predictions["right_joystick"],
            targets["right_joystick"].long(),
            weight=self.class_weights
        )

        joystick_loss = (left_joystick_loss + right_joystick_loss) / 2

        # 按键分类损失
        button_loss = F.binary_cross_entropy_with_logits(
            predictions["button_probs"],
            targets["buttons"].float()
        )

        # 总损失
        total_loss = (
            self.joystick_weight * joystick_loss +
            self.button_weight * button_loss
        )

        losses = {
            "total": total_loss.item(),
            "joystick": joystick_loss.item(),
            "button": button_loss.item()
        }

        return total_loss, losses


def extract_joystick_position_from_mask(
    mask: torch.Tensor,
    grid_size: int = 11,
    threshold: float = 0.5
) -> Tuple[int, int]:
    """
    从分割mask中提取摇杆位置

    Args:
        mask: [121, H, W] 摇杆分割logits
        grid_size: 网格大小 (11)
        threshold: 置信度阈值

    Returns:
        (grid_x, grid_y): 网格坐标 [0-10, 0-10]
    """
    # 转换为概率
    probs = F.softmax(mask, dim=0)  # [121, H, W]

    # 找到最大概率的类别
    max_class = torch.argmax(probs.mean(dim=[1, 2]))  # 标量

    # 转换为网格坐标
    grid_y = max_class // grid_size  # 行
    grid_x = max_class % grid_size   # 列

    return grid_x.item(), grid_y.item()


def normalize_joystick_position(
    grid_x: int,
    grid_y: int,
    grid_size: int = 11
) -> Tuple[float, float]:
    """
    将网格坐标归一化到 [-1, 1]

    Args:
        grid_x, grid_y: 网格坐标 [0-10]

    Returns:
        (norm_x, norm_y): 归一化坐标 [-1, 1]
    """
    # 将 [0, 10] 映射到 [-1, 1]
    norm_x = 2.0 * grid_x / (grid_size - 1) - 1.0
    norm_y = 2.0 * grid_y / (grid_size - 1) - 1.0

    return norm_x, norm_y


def contour_based_position_refinement(
    mask: np.ndarray,
    center: Tuple[float, float],
    reference_size: Tuple[float, float]
) -> Tuple[float, float]:
    """
    基于轮廓检测的精化位置提取 (论文推理阶段使用)

    Args:
        mask: [H, W] 二值mask
        center: 初始中心位置
        reference_size: 参考尺寸 (用于归一化)

    Returns:
        (x, y): 精化后的位置
    """
    import cv2

    # 寻找轮廓
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return center

    # 找到最大轮廓
    largest_contour = max(contours, key=cv2.contourArea)

    # 计算质心
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return center

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # 归一化到参考尺寸
    ref_w, ref_h = reference_size
    norm_x = (cx - ref_w / 2) / (ref_w / 2)
    norm_y = (cy - ref_h / 2) / (ref_h / 2)

    return np.clip(norm_x, -1.0, 1.0), np.clip(norm_y, -1.0, 1.0)


if __name__ == "__main__":
    # 测试模型
    print("Testing NitroGen Action Parser...")

    try:
        model = NitroGenActionParser(
            num_buttons=16,
            use_pretrained=True  # 使用默认的本地模型路径
        )
        frame_pairs = torch.randn(2, 6, 256, 256)

        print("\nRunning forward pass...")
        outputs = model(frame_pairs)
        print("Model output shapes:")
        for k, v in outputs.items():
            print(f"  {k}: {v.shape}")

        # 测试损失
        print("\nTesting loss function...")
        criterion = ActionParsingLoss()

        targets = {
            "left_joystick": torch.randint(0, 121, (2, 8, 8)),  # 匹配输出尺寸 [2, 121, 8, 8]
            "right_joystick": torch.randint(0, 121, (2, 8, 8)),
            "buttons": torch.randint(0, 2, (2, 16))
        }

        loss, losses = criterion(outputs, targets)
        print(f"Loss: {losses}")

        print("\n[SUCCESS] All tests passed!")

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure transformers is installed: pip install transformers")
        print("  2. Try: export HF_HUB_OFFLINE=1  (Linux/Mac)")
        print("  3. Or use local model only")
        raise
