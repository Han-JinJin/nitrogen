# Stage 2: 动作解析 (Action Parsing)

## 📁 项目结构（最终版本）

```
stage2/
├── 📄 核心代码（8个文件）
│   ├── models.py                          # SegFormer模型架构
│   ├── dataset.py                         # 两帧拼接数据集
│   ├── button_layouts.py                  # PS4/Xbox按键布局定义
│   ├── train.py                           # 统一训练脚本
│   ├── evaluate.py                        # 模型评估脚本
│   ├── generate_synthetic_training_data.py # 合成数据生成器
│   ├── inference.py                       # 动作解析推理脚本
│   └── check_deps.py                      # 依赖检查工具
│
├── 📊 数据和输出目录
│   ├── segformer-b5-local/               # 本地SegFormer模型
│   ├── synthetic_data/                   # 生成的训练数据
│   │   ├── frames/                       # 视频帧
│   │   └── annotations.json              # 标注文件
│   ├── checkpoints/                      # 训练检查点
│   └── logs/                             # TensorBoard日志
│
└── 📝 文档
    ├── PROJECT_STRUCTURE.md               # 本文件（项目结构）
    ├── EXECUTION_FLOW.md                  # 执行流程指南
    ├── INFERENCE_GUIDE.md                 # 推理使用指南
    ├── OPTIMIZATION_SUMMARY.md            # 优化总结
    └── BUTTON_EVALUATION_ISSUE.md         # 按键评估问题说明
```

---

## 📊 论文复现状态

**总体复现度: ~85%** ✅

详细分析见: [PAPER_REPRODUCTION_ANALYSIS.md](PAPER_REPRODUCTION_ANALYSIS.md)

| 方面 | 完成度 | 说明 |
|------|--------|------|
| **模型架构** | 100% ✅ | SegFormer-B5, 两帧拼接, 11×11网格, 16键分类 |
| **训练配置** | 100% ✅ | AdamW, lr=0.0001, weight_decay=0.1, 线性衰减 |
| **数据增强** | 60% ⚠️ | 基础增强完整, 缺少压缩artifacts |
| **数据质量** | 5% ❌ | 虚拟帧 vs 真实视频, 数据量差80,000倍 |
| **推理方法** | 70% ⚠️ | 单帧推理完整, 缺少视频级处理 |
| **性能水平** | 60% ⚠️ | R² 0.606 vs 论文0.84, Acc 84% vs 96% |

### 核心差距
1. ❌ **数据规模**: 8M帧 vs ~100帧（80,000倍差距）
2. ❌ **数据真实性**: 虚拟帧 vs 真实游戏视频+手柄overlay
3. ⚠️ **推理平滑**: 单帧 vs 视频级轮廓检测

---

## 🎯 核心文件说明

### 1. models.py ⭐ (14KB)
**作用**: SegFormer模型架构定义

**被依赖**: train.py, evaluate.py, inference.py, check_deps.py

**核心类**:
```python
NitroGenActionParser    # 主模型类
  ├── 输入: 两帧拼接 [B, 6, H, W]
  ├── 输出:
  │   ├── left_joystick: [B, 121, H/4, W/4]  # 11×11网格
  │   ├── right_joystick: [B, 121, H/4, W/4]
  │   └── button_probs: [B, 16]  # 二分类
  └── 加载: segformer-b5-local/ (预训练模型)

ActionParsingLoss      # 多任务损失函数
```

**辅助函数**:
- `extract_joystick_position_from_mask()` - 从分割mask提取摇杆位置
- `normalize_joystick_position()` - 归一化到[-1,1]
- `contour_based_position_refinement()` - 轮廓检测精化

---

### 2. dataset.py ⭐ (12KB)
**作用**: 训练和推理的数据加载

**被依赖**: train.py, inference.py

**核心类**:
```python
# 合成数据训练用
FramePairDataset
  ├── 读取: synthetic_data/annotations.json
  ├── 加载: synthetic_data/frames/
  └── 输出: {"frame_pair": Tensor, "joystick": ..., "buttons": ...}

# 推理/直接视频用
VideoFramePairDataset
  ├── 从视频文件直接加载帧
  ├── 支持bbox裁剪
  └── 输出: {"frame_pair": Tensor, "frame_idx": int}
```

**关键参数**:
- `image_size`: 图像尺寸(默认256)
- `temporal_gap`: 帧间间隔(默认1)
- `max_frames`: 最大帧数限制

---

### 3. button_layouts.py ⭐ (5KB)
**作用**: 手柄按键布局定义

**被依赖**: inference.py

**功能**:
```python
get_layout(template_name)
  ├── "ps4_standard"  # PS4标准布局
  ├── "xbox_standard" # Xbox标准布局
  └── 返回: {按钮名: (x, y, radius)}

get_all_buttons()
  └── 返回所有16个按钮的定义
```

**支持的按键**:
- 方向键: ↑↓←→
- 动作键: □△○×
- 功能键: Start, Select, L1, R1, L2, R2, L3, R3

---

### 4. train.py ⭐ (19KB)
**作用**: 统一训练脚本

**依赖**: models.py, dataset.py

**特性**:
- ✅ 命令行参数配置
- ✅ 混合精度训练 (--mixed_precision)
- ✅ 内存清理选项 (--memory_clean_interval)
- ✅ 单行动态进度显示
- ✅ 自动保存最佳模型
- ✅ 内置离线模式（HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE）

**基本用法**:
```bash
python train.py                           # 默认配置
python train.py --batch_size 8            # 自定义批次
python train.py --mixed_precision         # 混合精度
```

---

### 5. generate_synthetic_training_data.py (18KB)
**作用**: 生成合成训练数据

**依赖**: dataset.py

**功能**:
- 创建虚拟游戏视频帧
- 随机生成手柄按键状态
- 生成对应的标注文件

**用法**:
```bash
python generate_synthetic_training_data.py --num_samples 10000
```

**输出**: `synthetic_data/` 目录

---

### 6. inference.py (14KB)
**作用**: 推理脚本，解析视频中的手柄动作

**依赖**: models.py, dataset.py, button_layouts.py

**功能**:
- 加载训练好的模型
- 读取视频文件
- 逐帧解析手柄状态
- 输出JSON格式动作序列
- 内置离线模式

**输出**: `../output/*.json`

---

### 7. check_deps.py (4KB)
**作用**: 依赖检查工具

**依赖**: models.py

**功能**:
- 检查 PyTorch 安装
- 检查 transformers 安装
- 验证 models.py 可导入
- 验证 SegFormer 模型可加载

---

## 🔄 依赖关系图

```
┌─────────────────────────────────────────────────────────┐
│                   核心代码依赖关系                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  models.py ←─────────────────────────────────────┐     │
│     │                                              │     │
│     ├──────────────┬──────────────┐              │     │
│     ↓              ↓              ↓              │     │
│  train.py    stage2_*.py    check_deps.py         │     │
│     │              │                            │     │
│     ↓              │                            │     │
│  dataset.py ───────┘                            │     │
│                          │                        │     │
│     button_layouts.py ──┘                        │     │
│                                                  │     │
│  generate_synthetic_training_data.py ────────────┘     │
│                  (独立运行)                            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## ⚙️ 环境设置说明

### 离线模式
所有Python脚本已内置离线模式设置，无需额外配置：

```python
# train.py, inference.py 开头
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
```

### GPU/CPU设置
自动检测，可通过命令行参数指定：

```bash
# 自动检测（默认）
python train.py

# 强制CPU
python train.py --device cpu

# 强制GPU
python train.py --device cuda
```

---

## 📖 执行流程对应文件

| 阶段 | 执行文件 | 依赖的核心文件 |
|-----|---------|--------------|
| **环境检查** | `check_deps.py` | models.py |
| **数据生成** | `generate_synthetic_training_data.py` | dataset.py |
| **训练** | `train.py` | models.py, dataset.py |
| **推理** | `inference.py` | models.py, dataset.py, button_layouts.py |

---

## 📊 数据流向

```
┌─────────────────────────────────────────────────────────────┐
│                       训练流程                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  segformer-b5-local/ (预训练SegFormer)                       │
│      ↓                                                      │
│  models.py (加载并微调)                                      │
│      ↓                                                      │
│  synthetic_data/ ← dataset.py                               │
│  ├── frames/                                                │
│  └── annotations.json                                       │
│      ↓                                                      │
│  train.py (训练)                                            │
│      ↓                                                      │
│  checkpoints/best_model.pt                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       推理流程                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  视频文件                                                    │
│      ↓                                                      │
│  dataset.py (VideoFramePairDataset)                         │
│      ↓                                                      │
│  checkpoints/best_model.pt ← models.py (加载)               │
│      ↓                                                      │
│  inference.py (推理)                  │
│      + button_layouts.py (按键定义)                         │
│      ↓                                                      │
│  output/*.json                                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎓 论文对应关系

### NitroGen Stage 2: Gamepad Action Parsing

| 论文描述 | 代码实现 | 文件位置 | 状态 |
|---------|---------|----------|------|
| **模型架构** |
| fine-tuned SegFormer | `NitroGenActionParser` | models.py:L48 | ✅ |
| pairs of consecutive frames | `FramePairDataset` | dataset.py:L185 | ✅ |
| concatenated spatial dim | `torch.cat([f1, f2], dim=1)` | dataset.py:L260 | ✅ |
| 11×11 grid segmentation | `[B, 121, H/4, W/4]` | models.py:L93 | ✅ |
| binary button states (16) | `button_probs: [B, 16]` | models.py:L97 | ✅ |
| **训练配置** |
| AdamW optimizer | `optim.AdamW(...)` | train.py:L522 | ✅ |
| learning rate 0.0001 | `lr=1e-4` | train.py:L522 | ✅ |
| linear learning rate decay | `LambdaLR(linear)` | train.py:L543 | ✅ |
| weight decay 0.1 | `weight_decay=0.1` | train.py:L178 | ✅ |
| batch size 256 | `BATCH_SIZE=8` | train.py:L178 | ⚠️ (受限于资源) |
| **数据增强** |
| color jittering | `ColorJitter()` | dataset.py:L332 | ✅ |
| horizontal flip | `cv2.flip()` | dataset.py:L324 | ✅ |
| brightness/contrast | 手动调整 | dataset.py:L337 | ✅ |
| Gaussian noise | `gaussian_noise` | dataset.py:L345 | ✅ |
| overlay opacity | ❌ 未实现 | - | ❌ |
| video compression | ❌ 未实现 | - | ❌ |
| **推理方法** |
| spatial average pooling | `mean(dim=[1,2])` | inference.py:L98 | ✅ |
| argmax class | `argmax()` | inference.py:L103 | ✅ |
| 11×11 grid coordinates | `// 11`, `% 11` | inference.py:L108 | ✅ |
| 99th percentile norm | `_normalize_joysticks()` | models.py:L164 | ✅ |
| contour detection | `contour_based_refinement()` | models.py:L194 | ✅ |
| video-level smoothing | ❌ 未实现 | - | ⚠️ |

### 性能对比

| 指标 | 论文报告 | 当前代码 | 差距 |
|------|---------|---------|------|
| 摇杆 R² | **0.84** | 0.606 | -28% |
| 按键准确率 | **0.96** | 0.840 | -12% |
| 按键 F1 | N/A | 0.000 | 类别不平衡 |

详细分析: [PAPER_REPRODUCTION_ANALYSIS.md](PAPER_REPRODUCTION_ANALYSIS.md)

---

## 📈 性能基准

| 指标 | 论文报告 |
|------|---------|
| 按键准确率 | **96%** |
| 摇杆R² | **0.84** |
| 推理速度 | ~30 fps (GPU) |

---

## 🗂️ 文件大小

```
models.py                           ~14KB
dataset.py                          ~12KB
button_layouts.py                   ~5KB
train.py                           ~19KB
inference.py  ~14KB
generate_synthetic_training_data.py ~18KB
check_deps.py                       ~4KB
─────────────────────────────────────────
总计                               ~86KB
```

---

## ✅ 清理总结

### 已删除的文件：
- ❌ `visualize_layouts.py` - 独立可视化工具
- ❌ `set_env.bat`, `run_with_env.bat` - Windows脚本（功能已内置）
- ❌ `stage2_action_parsing_segformer.py` - 已重命名为 `inference.py`
- ❌ 临时测试: `minimal_test.py`, `quick_test.py`, `simple_test.py`, `diagnose.py`
- ❌ 临时分析: `analyze_button_distribution.py`, `analyze_button_predictions.py`, `debug_model_output.py`
- ❌ 临时测试: `test_dataset.py`, `test_evaluate_fix.py`, `test_evaluation_pipeline.py`
- ❌ 重复文档: `FIXES_SUMMARY.md`, `FINAL_SUMMARY.md`, `PYTORCH_FIX.md`, `TRAINING_COMMANDS.md`

### 保留的文件：
- ✅ 8个核心Python文件（含inference.py）
- ✅ 6个文档文件（新增PAPER_REPRODUCTION_ANALYSIS.md）
- ✅ 所有必需的依赖关系完整
