"""
按键布局定义
为每个手柄类型定义按键的相对位置和区域
使用相对坐标 (0-1) 以适应不同尺寸
"""

import cv2
import numpy as np

# PS4 DualShock 4 按键布局
PS4_LAYOUT = {
    "face_buttons": {
        # (x_center, y_center, radius) 相对坐标
        "cross": (0.70, 0.55, 0.08),
        "circle": (0.82, 0.45, 0.08),
        "square": (0.58, 0.45, 0.08),
        "triangle": (0.70, 0.35, 0.08),
    },
    "shoulder_buttons": {
        # (x_center, y_center, width, height)
        "L1": (0.20, 0.15, 0.12, 0.08),
        "R1": (0.80, 0.15, 0.12, 0.08),
        "L2": (0.15, 0.08, 0.12, 0.08),
        "R2": (0.85, 0.08, 0.12, 0.08),
    },
    "dpad": {
        "dpad_up": (0.45, 0.40, 0.06, 0.06),
        "dpad_down": (0.45, 0.60, 0.06, 0.06),
        "dpad_left": (0.35, 0.50, 0.06, 0.06),
        "dpad_right": (0.55, 0.50, 0.06, 0.06),
    },
    "action_buttons": {
        "options": (0.75, 0.30, 0.08, 0.06),
        "share": (0.25, 0.30, 0.08, 0.06),
        "ps_button": (0.50, 0.50, 0.06),  # 圆形
    },
    "sticks": {
        "L3": (0.30, 0.65, 0.10),  # 左摇杆按下
        "R3": (0.70, 0.70, 0.10),  # 右摇杆按下
    }
}

# Xbox One 按键布局
XBOX_LAYOUT = {
    "face_buttons": {
        "A": (0.70, 0.55, 0.08),
        "B": (0.82, 0.45, 0.08),
        "X": (0.58, 0.45, 0.08),
        "Y": (0.70, 0.35, 0.08),
    },
    "shoulder_buttons": {
        "LB": (0.20, 0.15, 0.12, 0.08),
        "RB": (0.80, 0.15, 0.12, 0.08),
        "LT": (0.15, 0.08, 0.12, 0.08),
        "RT": (0.85, 0.08, 0.12, 0.08),
    },
    "dpad": {
        "dpad_up": (0.45, 0.40, 0.06, 0.06),
        "dpad_down": (0.45, 0.60, 0.06, 0.06),
        "dpad_left": (0.35, 0.50, 0.06, 0.06),
        "dpad_right": (0.55, 0.50, 0.06, 0.06),
    },
    "action_buttons": {
        "menu": (0.75, 0.30, 0.08, 0.06),
        "view": (0.25, 0.30, 0.08, 0.06),
        "xbox_button": (0.50, 0.50, 0.06),
    },
    "sticks": {
        "L3": (0.30, 0.65, 0.10),
        "R3": (0.70, 0.70, 0.10),
    }
}

# 模板名称到布局的映射
TEMPLATE_TO_LAYOUT = {
    # PS4 模板
    "ps4_red_v1": PS4_LAYOUT,
    "ps4_red_v2": PS4_LAYOUT,
    "ps4_red_v3": PS4_LAYOUT,
    "DS4_V1_base": PS4_LAYOUT,
    "DS4_V2_base": PS4_LAYOUT,
    "DualSense_base": PS4_LAYOUT,
    # Xbox 模板
    "xbox_green_v4": XBOX_LAYOUT,
    "xbox_black_v5": XBOX_LAYOUT,
    "XB1_S_base": XBOX_LAYOUT,
    "XBSeries_base": XBOX_LAYOUT,
}


def get_layout(template_name):
    """根据模板名称获取按键布局"""
    return TEMPLATE_TO_LAYOUT.get(template_name, PS4_LAYOUT)  # 默认使用PS4布局


def get_all_buttons(layout):
    """获取布局中所有按键的名称和区域定义"""
    all_buttons = {}
    for category, buttons in layout.items():
        all_buttons.update(buttons)
    return all_buttons


def extract_button_region(roi, button_def, roi_shape):
    """
    从手柄ROI中提取按键区域

    Args:
        roi: 手柄区域图像
        button_def: 按键定义 (x, y, ...) 或 (x, y, w, h)
        roi_shape: ROI形状 (height, width)

    Returns:
        按键区域图像
    """
    h, w = roi_shape[:2]

    if len(button_def) == 3:
        # 圆形区域: (x_center, y_center, radius)
        cx, cy, radius_rel = button_def
        cx_abs = int(cx * w)
        cy_abs = int(cy * h)
        radius = int(radius_rel * min(w, h))

        # 创建圆形mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx_abs, cy_abs), radius, 255, -1)

        # 应用mask
        if len(roi.shape) == 3:
            masked = cv2.bitwise_and(roi, roi, mask=mask)
            # 获取边界框以裁剪
            x1 = max(0, cx_abs - radius)
            y1 = max(0, cy_abs - radius)
            x2 = min(w, cx_abs + radius)
            y2 = min(h, cy_abs + radius)
            return masked[y1:y2, x1:x2], mask[y1:y2, x1:x2]
        else:
            masked = cv2.bitwise_and(roi, roi, mask=mask)
            x1 = max(0, cx_abs - radius)
            y1 = max(0, cy_abs - radius)
            x2 = min(w, cx_abs + radius)
            y2 = min(h, cy_abs + radius)
            return masked[y1:y2, x1:x2], mask[y1:y2, x1:x2]

    elif len(button_def) == 4:
        # 矩形区域: (x_center, y_center, width, height)
        cx, cy, w_rel, h_rel = button_def
        w_abs = int(w_rel * w)
        h_abs = int(h_rel * h)

        x1 = max(0, int(cx * w - w_abs // 2))
        y1 = max(0, int(cy * h - h_abs // 2))
        x2 = min(w, int(cx * w + w_abs // 2))
        y2 = min(h, int(cy * h + h_abs // 2))

        return roi[y1:y2, x1:x2], None

    return None, None
