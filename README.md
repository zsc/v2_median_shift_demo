# v2 Median/Mean Shift 图像滤波演示

本项目实现了类 Mathematica/Wolfram 风格的 `MeanShiftFilter` 均值漂移滤波，以及中值/几何中值变体算法，并提供一个批量处理 CLI 工具，可生成 `index.html` 可视化页面来对比每一轮迭代的效果。

## 算法简介

### Mean Shift（均值漂移）
均值漂移是一种迭代式的图像平滑算法，通过将每个像素值向其邻域内相似像素的平均值移动来实现滤波。它能在平滑图像的同时保留边缘信息。

### Median Shift（中值漂移）
中值漂移是均值漂移的变体，使用几何中值或分量中值代替均值。对椒盐噪声等异常值有更好的鲁棒性。

- **geometric（几何中值）**：计算多维空间中的几何中值，较慢但更准确
- **component（分量中值）**：分别对每个颜色通道取中值，速度快，适合 GPU 加速

## 环境配置

### 创建虚拟环境
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt
```

### 依赖要求
- Python >= 3.9
- PyTorch >= 2.0（用于 GPU 加速）
- NumPy >= 1.20, < 2
- Pillow >= 9.0
- tqdm >= 4.0

## 使用方法

### 批量处理 + 生成对比页面

```bash
.venv/bin/python run_batch.py \
  --inputs <目录或文件路径...> \
  --outdir out \
  --r 4 \
  --d 0.12 \
  --max_iter 3 \
  --median_mode component \
  --resize_max 800
```

然后用浏览器打开 `out/index.html` 查看对比结果。

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--inputs` | 输入图片路径，支持目录、glob 模式或多个文件 | 必填 |
| `--outdir` | 输出目录 | 必填 |
| `-r` | 空间邻域半径（像素） | 4 |
| `-d` | 颜色距离阈值 | 0.12 |
| `--max_iter` | 最大迭代次数 | 3 |
| `--median_mode` | 中值模式：`geometric` 或 `component` | component |
| `--distance` | 距离度量：euclidean/sqeuclidean/manhattan/cosine | sqeuclidean |
| `--resize_max` | 最长边缩放尺寸（像素），0 表示不缩放 | 0 |
| `--backend` | 计算后端：auto/numpy/torch | auto |
| `--device` | 设备：auto/cpu/mps | auto |

### 示例命令

处理单张图片：
```bash
.venv/bin/python run_batch.py --inputs test.png --outdir out --r 4 --d 0.12
```

处理多张图片：
```bash
.venv/bin/python run_batch.py --inputs test.png celeba.png --outdir out --r 4 --d 0.12 --max_iter 3
```

处理整个目录：
```bash
.venv/bin/python run_batch.py --inputs ./images/ --outdir out --r 4 --d 0.12 --median_mode component
```

## Apple MPS（GPU 加速）

- 默认 `--backend auto --device auto` 会自动检测并使用 `torch` + `mps`（Apple Silicon）
- 强制使用 CPU：`--device cpu`
- 强制使用 MPS：`--device mps`

## 注意事项

- `--median_mode geometric` 是纯 CPU/Numpy 实现，计算较慢，适合小图或作为参考
- `--median_mode component` 支持 GPU 加速，推荐用于批量处理大图
- 建议先用 `--max_iter 1` 测试效果，再逐步增加迭代次数

## 项目结构

```
.
├── filters.py      # 核心滤波算法实现
├── run_batch.py    # 批量处理 CLI 工具
├── test_filters.py # 单元测试
├── requirements.txt
└── README.md
```

## 许可证

MIT License
