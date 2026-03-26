# Stage 2 执行顺序指南

## 📋 核心代码文件（7个）

| 文件 | 大小 | 作用 | 被依赖者 | 是否必需 |
|------|------|------|---------|---------|
| **models.py** | 14KB | SegFormer模型架构 | train.py, inference.py, evaluate.py, check_deps.py | ⭐ 必需 |
| **dataset.py** | 12KB | 数据集类(训练/推理) | train.py, inference.py, generate_synthetic_training_data.py | ⭐ 必需 |
| **button_layouts.py** | 5KB | PS4/Xbox按键布局 | inference.py | ⭐ 必需 |
| **train.py** | 19KB | 训练脚本 | - | ⭐ 必需 |
| **evaluate.py** | 14KB | 模型评估脚本 | - | ⭐ 必需 |
| **inference.py** | 14KB | 推理脚本 | - | ⭐ 必需 |
| **generate_synthetic_training_data.py** | 18KB | 合成数据生成 | - | ⭐ 必需 |
| **check_deps.py** | 4KB | 依赖检查工具 | - | 工具 |

**总计**: 8个核心Python文件

> **论文复现状态**: ~85% | 详细分析见 [PAPER_REPRODUCTION_ANALYSIS.md](PAPER_REPRODUCTION_ANALYSIS.md)

---

## 🔄 完整执行流程

### 方案A: 从零训练模型

```
步骤1: 检查依赖
  └─→ python check_deps.py
       └─→ 验证 models.py 可导入

步骤2: 安装依赖（如需要）
  └─→ pip install torch torchvision transformers opencv-python numpy tqdm tensorboard
       (详见 README.md 依赖列表)

步骤3: 生成训练数据
  └─→ python generate_synthetic_training_data.py --num_samples 10000
       ├─→ 读取: 无
       ├─→ 使用: dataset.py (FramePairDataset)
       └─→ 输出: synthetic_data/

步骤4: 训练模型
  └─→ python train.py --batch_size 4 --num_epochs 10
       ├─→ 导入: models.py, dataset.py
       ├─→ 读取: synthetic_data/
       ├─→ 读取: segformer-b5-local/ (预训练模型)
       └─→ 输出: checkpoints/, logs/

步骤5: 监控训练（可选，另开终端）
  └─→ tensorboard --logdir=logs
```

### 方案B: 仅推理（使用已有模型）

```
推理视频动作
  └─→ python inference.py --video_dir ../data/videos
       ├─→ 导入: models.py, dataset.py, button_layouts.py
       ├─→ 读取: checkpoints/best_model.pt (自动查找最新)
       ├─→ 读取: 视频文件
       └─→ 输出: output/*.json

评估模型性能
  └─→ python evaluate.py --checkpoint checkpoints/YYYYMMDD_HHMMSS/best_model.pt
       ├─→ 导入: models.py, dataset.py
       ├─→ 读取: synthetic_data/ (验证集)
       └─→ 输出: 按键准确率、摇杆R²、F1分数
```

---

## ⚙️ 环境设置

### 离线模式（自动）
所有Python脚本已内置离线模式，**无需额外配置**：

```python
# 代码内部已设置
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
```

### GPU/CPU模式
自动检测，可通过命令行参数指定：

```bash
# 自动检测（默认）
python train.py

# 强制使用CPU
python train.py --device cpu

# 强制使用GPU
python train.py --device cuda
```

### 关于bat脚本
已删除 `set_env.bat` 和 `run_with_env.bat`，因为：
- ✅ 离线模式已内置到Python代码中
- ✅ GPU/CPU设置通过命令行参数控制
- ✅ 跨平台兼容性更好（Windows/Linux/Mac）

---

## 📊 文件依赖关系

### 训练相关

```
train.py
├── models.py (NitroGenActionParser, ActionParsingLoss)
│   └── transformers, torch
└── dataset.py (FramePairDataset, create_train_val_dataloaders)
    └── cv2, torch

generate_synthetic_training_data.py
└── dataset.py (FramePairDataset)
    └── cv2, json
```

### 推理相关

```
inference.py
├── models.py
│   ├── NitroGenActionParser
│   ├── extract_joystick_position_from_mask
│   ├── normalize_joystick_position
│   └── contour_based_position_refinement
├── dataset.py (VideoFramePairDataset)
│   └── cv2, torch
└── button_layouts.py
    ├── get_layout
    └── get_all_buttons

evaluate.py
├── models.py
│   ├── NitroGenActionParser
│   └── ActionParsingLoss
└── dataset.py (FramePairDataset)
    └── synthetic_data/
```

### 工具相关

```
check_deps.py
└── models.py (测试导入)
```

---

## 🎯 按使用场景的执行顺序

### 场景1: 首次使用/测试环境

```bash
# 1. 检查环境（验证 models.py 可导入）
python check_deps.py

# 2. 小规模测试
python generate_synthetic_training_data.py --num_samples 100
python train.py --num_epochs 1 --batch_size 2

# 3. 如果测试通过，进行正式训练
python generate_synthetic_training_data.py --num_samples 10000
python train.py --batch_size 4 --num_epochs 10
```

### 场景2: 生产环境训练

```bash
# 1. 生成大规模数据（时间较长）
python generate_synthetic_training_data.py --num_samples 100000

# 2. 完整训练
python train.py --batch_size 8 --num_epochs 50 --mixed_precision

# 3. 监控训练（另开终端）
tensorboard --logdir=logs
```

### 场景3: 仅推理/使用已有模型

```bash
# 1. 推理视频动作（自动使用最新模型）
python inference.py --video_dir ../data/videos

# 2. 评估模型性能
python evaluate.py

# 3. 指定模型推理
python inference.py --video video.mp4 --checkpoint checkpoints/best_model.pt
```

---

## 📁 各文件输入输出详解

### models.py
- **类型**: 核心模块（被导入）
- **输入**: 不直接执行
- **输出**: 提供 `NitroGenActionParser` 类和相关函数
- **依赖**: transformers, torch

### dataset.py
- **类型**: 核心模块（被导入）
- **输入**: 视频帧目录 + 标注JSON
- **输出**: PyTorch DataLoader
- **提供的类**:
  - `FramePairDataset` - 训练用
  - `VideoFramePairDataset` - 推理用
  - `create_train_val_dataloaders()` - 创建加载器
- **依赖**: cv2, torch, json

### button_layouts.py
- **类型**: 辅助模块（被导入）
- **输入**: 模板名称字符串
- **输出**: 按键位置字典
- **提供的函数**:
  - `get_layout(template_name)` - 获取布局
  - `get_all_buttons()` - 获取所有按钮
- **依赖**: 无

### generate_synthetic_training_data.py
- **类型**: 独立脚本
- **输入**: 命令行参数 `--num_samples`
- **输出**: `synthetic_data/frames/` 和 `synthetic_data/annotations.json`
- **依赖**: dataset.py
- **用法**: `python generate_synthetic_training_data.py --num_samples 10000`

### train.py
- **类型**: 独立脚本
- **输入**: `synthetic_data/` 目录
- **输出**: `checkpoints/` 和 `logs/`
- **依赖**: models.py, dataset.py
- **用法**: `python train.py --batch_size 4 --num_epochs 10`

### inference.py
- **类型**: 独立脚本
- **输入**: 视频文件/目录 + 训练好的模型
- **输出**: `output/*.json` 动作文件
- **依赖**: models.py, dataset.py, button_layouts.py
- **用法**: `python inference.py --video video.mp4` 或 `--video_dir videos/`
- **功能**: 批量推理、自动查找checkpoint、GPU/CPU自适应

### evaluate.py
- **类型**: 独立脚本
- **输入**: 训练好的模型
- **输出**: 控制台（性能指标）
- **依赖**: models.py, dataset.py
- **用法**: `python evaluate.py`
- **功能**: 计算按键准确率、摇杆R²、F1分数

---

## ⚡ 快速命令清单

### 环境相关
```bash
python check_deps.py              # 检查依赖（验证models.py可导入）
pip install torch torchvision transformers opencv-python numpy tqdm tensorboard
```

### 数据生成
```bash
python generate_synthetic_training_data.py                    # 默认1000样本
python generate_synthetic_training_data.py --num_samples 100  # 小规模
python generate_synthetic_training_data.py --num_samples 100000 # 大规模
```

### 训练
```bash
python train.py                                                   # 默认配置
python train.py --batch_size 2 --num_epochs 5                    # 快速测试
python train.py --batch_size 8 --mixed_precision                 # 推荐
python train.py --batch_size 16 --num_epochs 50                  # 完整训练
```

### 监控
```bash
tensorboard --logdir=logs          # 启动TensorBoard
```

### 推理
```bash
# 单个视频
python inference.py --video video.mp4

# 批量推理
python inference.py --video_dir ../data/videos --output output/

# 评估性能
python evaluate.py
```

---

## 🔍 验证执行是否成功

### 1. 依赖检查成功
```
✓ All dependencies are installed!
✓ models.py can be imported successfully
```

### 2. 数据生成成功
```
synthetic_data/
├── frames/
│   ├── synthetic_000001/
│   └── ...
└── annotations.json
```

### 3. 训练成功
```
checkpoints/YYYYMMDD_HHMMSS/
├── best_model.pt              ← 最佳模型
└── checkpoint_epoch_*.pt
```

训练日志显示：
```
Epoch 1/10 - Train: 2.3456 (js=1.2345, btn=1.1111) | Val: 2.1234 | Best: 2.1234
Training complete!
Best validation loss: 2.1234
```

### 4. 推理成功
```
output/
├── video1_actions.json
├── video2_actions.json
└── ...
```

### 5. 评估成功
```
Button Metrics:
  Overall Accuracy: 0.840
  Overall F1: 0.750

Joystick R²:
  Left Joystick: 0.623
  Right Joystick: 0.589
  Average: 0.606
```

---

## 📝 注意事项

1. **核心文件不可删除**
   - ❌ 不要删除: models.py, dataset.py, button_layouts.py
   - 这些是训练和推理必需的
   - 总共只有7个Python文件，每个都有用途

2. **环境设置已内置**
   - ✅ 离线模式已内置（HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE）
   - ✅ GPU/CPU自动检测，可通过参数指定
   - ❌ 不需要bat脚本或额外环境配置

3. **首次运行**
   - 必须先生成训练数据才能训练
   - 使用 `check_deps.py` 验证环境

3. **训练时间**
   - 100样本 ~ 2分钟
   - 10000样本 ~ 30分钟
   - 论文规模(8M) ~ 数天

4. **中断恢复**
   - 当前版本不支持断点续训

5. **模型选择**
   - 推理会自动使用 `checkpoints/` 中最新的 `best_model.pt`

---

## 🛠️ 故障排查

### 问题1: "No module named 'models'"
```
解决: 确保在 stage2/ 目录下运行脚本
```

### 问题2: "CUDA out of memory"
```
解决: python train.py --batch_size 2 --mixed_precision
```

### 问题3: "No frames found in video"
```
解决: 检查视频文件路径和格式
```

### 问题4: "checkpoint not found"
```
解决: 先运行 train.py 训练模型，或下载预训练模型
```
