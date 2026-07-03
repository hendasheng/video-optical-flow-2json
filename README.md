# Video Optical Flow to JSON

视频光流离线检测，支持稠密网格和稀疏追踪两种模式，导出 JSON 供前端做特效。

## 安装

### Windows

```powershell
# 安装 Miniconda（如已安装跳过）
# 下载 https://docs.conda.io/en/latest/miniconda.html 并安装

conda create -n optical-flow python=3.11 -y
conda activate optical-flow
pip install opencv-python numpy
```

### macOS

```bash
# 安装 Homebrew（如已安装跳过）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装 Miniconda
brew install --cask miniconda
conda init "$(basename $SHELL)"

conda create -n optical-flow python=3.11 -y
conda activate optical-flow
pip install opencv-python numpy
```

## 检测光流

```bash
python detector.py <video> [选项]
```

### 通用选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--scale` | 1.0 | 缩放因子，4K 视频建议 `0.5` |
| `--start` | 0 | 起始帧 |
| `--end` | - | 结束帧 |
| `-o, --output` | `output/<视频名>_flow.json` | 输出路径 |
| `--viz` | - | 输出可视化帧 |
| `--viz-step` | 10 | 可视化帧间隔 |

### 稠密模式（默认）

Farneback 算法，输出固定网格运动向量。适合做流体扰动、全场变形等效果。

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `-s, --step` | 16 | 网格步长（越小越密，数据越大） |

```bash
# 4K 视频缩到 1080p，步长 32
python detector.py video.mp4 -s 32 --scale 0.5

# 输出箭头预览验证
python detector.py video.mp4 -s 32 --scale 0.5 --viz --viz-step 5
```

输出：`output/<视频名>_flow.json`

```json
{
  "mode": "dense",
  "width": 1920, "height": 1080,
  "grid_step": 32, "grid_cols": 60, "grid_rows": 34,
  "frames": [
    { "frame": 1, "flow": [[[dx,dy], ...], ...] }
  ]
}
```

### 稀疏模式

Lucas-Kanade 追踪 Shi-Tomasi 角点，输出特征点轨迹。数据量极小，适合粒子跟随、点云效果。

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--sparse` | - | 启用稀疏模式 |
| `-n, --max-points` | 500 | 最大追踪点数 |
| `-q, --quality` | 0.01 | 角点质量阈值（越小点越多） |

```bash
# 追踪 800 个特征点
python detector.py video.mp4 --sparse -n 800 --scale 0.5

# 降低质量阈值获得更多点
python detector.py video.mp4 --sparse -n 500 -q 0.005 --viz
```

输出：`output/<视频名>_sparse.json`

```json
{
  "mode": "sparse",
  "width": 1920, "height": 1080,
  "max_points": 500,
  "frames": [
    { "frame": 1, "points": [[id, x, y, dx, dy], ...] }
  ]
}
```

点数低于 30% 时会自动补充检测新角点。

## 预览光流

用浏览器打开 `web/index.html`，拖入视频和对应的光流 JSON 文件即可播放。

- **空格**：播放 / 暂停
- **← →**：逐帧进退
- **进度条**：拖拽跳转
- **箭头缩放滑块**：调整箭头长度

视频分辨率与 JSON 中存储的分辨率不一致时会自动对齐。
