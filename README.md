# 手机、人脸与拍摄者视频识别

此仓库包含手机、人脸与人体姿态检测代码，以及两份匹配的 YOLO 权重。在实时网页中，Qwen 进一步核对人物是否正在拍照／录像，并维护每位拍摄者最多 5 张人脸截图。网页支持本机摄像头、RTSP/RTMP 视频源、拖拽上传视频，以及批量生成打框 MP4。仓库不包含 Qwen 权重、Docker 镜像、真实测试视频、FTP 凭据或截图数据。

## 版本与目录

- **本仓库的源码**：推理与网页代码位于 `src/phone_face`；网页启动时无需预置 `video_test/test.mp4`。YOLO 权重的 SHA-256 见 [WEIGHTS.md](WEIGHTS.md)。
- **`phone-face-qwen35:ftp-export-v4`**：独立 Docker 镜像，内置 Qwen3.5-4B、YOLO、网页和 FTP 截图导出功能。该镜像文件**不在本仓库中**，本仓库源码也不是该镜像的逐字节快照。已有此镜像时可按下文运行。
- **本仓库的 Qwen 配置**：默认模型为 **Qwen3.5-4B**。本仓库源码不含 Docker 版的 FTP 导出功能。

目录：`src/phone_face/` 为推理和网页代码，`weights/` 为 YOLO 权重，`scripts/` 为命令行脚本和 Docker 版 FTP 上传器。`models/`、`video_test/`、`outputs/` 均不提交真实数据。

## 方案一：以 Qwen3.5-4B 运行本仓库源码

在带 NVIDIA GPU 的 Linux 主机上安装 `ffmpeg`、Miniconda 和可用的 NVIDIA 驱动。以下命令在仓库根目录执行，示例按 Python 3.10 和 CUDA 12.4 的 PyTorch 包编写；CUDA 包版本应与目标机器的驱动能力匹配。

```bash
conda create -n phone-face-vision python=3.10 -y
conda activate phone-face-vision
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-vision.txt
python -c 'import torch, cv2, ultralytics; print(torch.cuda.is_available(), cv2.__version__, ultralytics.__version__)'
```

Qwen 使用独立环境和 8001 端口。下载 [官方 Qwen3.5-4B 模型](https://huggingface.co/Qwen/Qwen3.5-4B)；该模型的官方页面提供了 vLLM 启动方式。网络受限时可先在联网机器下载，再完整复制 `models/Qwen3.5-4B/` 到服务器。下列原生配置为部署示例，尚未在新机器上实测；已有 v4 镜像可直接参考方案二。

```bash
conda create -n phone-face-qwen python=3.10 -y
conda activate phone-face-qwen
python -m pip install 'huggingface_hub[cli]' uv
uv pip install vllm --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly
hf download Qwen/Qwen3.5-4B \
  --local-dir models/Qwen3.5-4B

# 示例用 GPU 1 加载 Qwen，GPU 0 留给 YOLO；按实际显卡配置调整。
CUDA_VISIBLE_DEVICES=1 vllm serve "$PWD/models/Qwen3.5-4B" \
  --served-model-name Qwen3.5-4B \
  --tensor-parallel-size 1 --max-model-len 32768 \
  --port 8001 --gpu-memory-utilization 0.80
```

保持 Qwen 服务运行，在另一终端启动网页。`QWEN_VL_URL` 是 OpenAI 兼容接口地址；`DEVICE=0` 是 YOLO 使用的 GPU。启动成功后访问 `http://服务器IP:7860/dashboard`。右侧截图需要 Qwen 服务可用；Qwen 不可用时左侧 YOLO 打框仍能工作。

```bash
conda activate phone-face-vision
DEVICE=0 \
QWEN_VL_MODEL=Qwen3.5-4B \
QWEN_VL_URL=http://127.0.0.1:8001/v1/chat/completions \
python -m src.phone_face.web
```

检查服务：

```bash
curl http://127.0.0.1:8001/v1/models
curl http://127.0.0.1:7860/api/health
```

仓库不发布真实人脸演示视频。网页可直接上传自己的视频；如需 `/demo` 默认演示，可自行放入 `video_test/test.mp4`。批量打框也可运行：

```bash
DEVICE=0 ./scripts/run_phone_face_video.sh /path/to/input.mp4 result_name
```

## 实时 RTMP/RTSP 推流

相机或手机推送 RTMP 到 MediaMTX，例如 `rtmp://服务器IP:1935/live/action6`。在同一台服务器读取对应的 RTSP 地址 `rtsp://127.0.0.1:8554/live/action6`。启动识别网页时可预填视频源：

```bash
DEVICE=0 NETWORK_SOURCE=rtsp://127.0.0.1:8554/live/action6 \
QWEN_VL_MODEL=Qwen3.5-4B \
python -m src.phone_face.web
```

网页中的“RTMP / RTSP 网络推流”区域点击启动后才会读取视频源。RTMP 路径由推流端设置，不在识别代码中写死。网页预览可压缩，检测和人脸裁剪使用解码后的原始帧；YOLO 网络内部仍按 `LIVE_IMG_SIZE` 缩放推理输入。浏览器摄像头以无损 PNG 上传画面，服务器返回的带框预览为 JPEG。

## 方案二：运行已有的 `ftp-export-v4` Docker 镜像

此方案要求机器上**已经有** `phone-face-qwen35:ftp-export-v4` 镜像。它内置 Qwen3.5-4B，因此运行镜像时无需另行下载 Qwen。若手头只有镜像归档，先运行 `docker load -i /path/to/image.tar`，再用 `docker image inspect phone-face-qwen35:ftp-export-v4` 确认标签。此镜像是历史版本，不自动包含本仓库较新的画质修改。

上传本地视频、使用容器内网页：

```bash
mkdir -p outputs
docker run -d --name phone-face-web --restart unless-stopped \
  --gpus device=0 --shm-size=8g -p 7860:7860 \
  -e DEVICE=0 -e PHONE_FACE_EXPORT_DIR=/app/outputs/ftp_spool \
  -v "$PWD/outputs:/app/outputs" \
  phone-face-qwen35:ftp-export-v4 web
```

若从同一 Linux 主机读取 MediaMTX 推流，把上面的 `-p 7860:7860` 换成 `--network host`，并加上 `-e NETWORK_SOURCE=rtsp://127.0.0.1:8554/live/action6`。容器需要能访问主机的 MediaMTX 端口。GPU 号按实际机器修改。`docker logs -f phone-face-web` 查看启动日志；`docker stop phone-face-web` 停止。

Docker 版确认拍摄者后，将 JPEG 截图写入 `outputs/ftp_spool`。**写入目录不等于已上传 FTP**：主机还需运行上传器，并提供可用的 FTP 连接及现有 `ftp.py` 客户端示例文件；凭据只保留在主机，不应提交到 GitHub。

```bash
python3 scripts/ftp_upload_faces.py \
  --spool "$PWD/outputs/ftp_spool" \
  --ftp-example /path/to/private/ftp.py \
  --remote-root /your/remote/root \
  --state-dir "$PWD/outputs/ftp_upload_state" \
  --interval 1
```

如果需要**从源码构建新的**一体化镜像，而不是运行已有的 v4 镜像，需要准备 `models/Qwen3.5-4B` 中的模型权重：

```bash
python -m pip install 'huggingface_hub[cli]'
hf download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B
```

这条下载命令只准备 Qwen3.5-4B 权重，**不会生成或替换**历史的 `ftp-export-v4` 镜像。Qwen 模型权重不上传到本 GitHub 仓库。

## 画质与数据边界

输入视频可能已由相机有损编码。离线视频和网络直播在检测前不会被转成 JPEG；保存的打框 MP4 使用 H.264 有损编码。实时网页的 JPEG 仅用于显示。人脸截图按原始帧裁剪，再以 JPEG 保存；FTP 上传器直接传输截图字节，不二次压缩。截图是否清晰仍受拍摄距离、对焦、运动和光照限制。

公开仓库不包含拍摄对象的视频、人脸截图、FTP 地址／密码或 Qwen 大模型权重。请只处理和发布自己有权使用的影像。
