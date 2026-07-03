# Optical Flow Detection

视频稠密光流离线检测工具，导出 JSON 网格数据供前端做流体扰动等特效。

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

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `-s, --step` | 16 | 网格采样步长（越小越密，数据越大） |
| `--scale` | 1.0 | 缩放因子，4K 视频建议 `0.5` |
| `--start` | 0 | 起始帧 |
| `--end` | - | 结束帧 |
| `-o, --output` | `output/<视频名>_flow.json` | 输出路径 |
| `--viz` | - | 同时输出箭头可视化帧 |
| `--viz-step` | 10 | 可视化帧间隔 |

### 示例

```bash
# 4K 视频缩到 1080p 检测，步长 32
python detector.py video.mp4 -s 32 --scale 0.5

# 同时输出箭头预览验证光流质量
python detector.py video.mp4 -s 32 --scale 0.5 --viz --viz-step 5
```

输出 JSON 会存到 `output/` 目录，可视化帧存到 `output/<视频名>_viz/`。

## 预览光流

用浏览器打开 `web/index.html`，拖入视频和对应的光流 JSON 文件即可播放。

- **空格**：播放 / 暂停
- **← →**：逐帧进退
- **进度条**：拖拽跳转
- **箭头缩放滑块**：调整箭头长度

视频分辨率与 JSON 中存储的分辨率不一致时会自动对齐。
