# 手机、人脸与拍摄者识别

本仓库包含 YOLO 手机／人脸检测、人体姿态关联、Qwen3.5-4B 拍摄动作核对、人物重入关联、右侧拍摄者人脸 Top-5 画廊，以及 PNG 截图的 FTP 待传目录与上传器。网页支持视频上传、RTSP/RTMP／摄像头直播和打框视频导出；命令行也能处理视频。公开仓库只包含代码与小模型权重，**不包含 Qwen 权重、真实视频、人脸截图或 FTP 凭据**。

`weights/phone_face/best.pt` 检测 `0=phone` 和 `1=face`；`weights/yolov8n-pose.pt` 用于手机与持有者的姿态关联；`weights/face_reid/` 下的 YuNet/SFace 用于同一网页会话中人物离开后重入的关联。权重校验值见 [WEIGHTS.md](WEIGHTS.md)。所有默认权重路径都相对于仓库根目录。

## 环境与 Qwen 模型

以下是带 NVIDIA GPU 的 Linux 原生部署方式。先安装 `ffmpeg`、适合显卡驱动的 PyTorch，然后在两个独立 Python 环境中安装视觉依赖和 vLLM。PyTorch/CUDA 包应按目标机器选择；下方 CUDA 12.4 仅为安装示例。

```bash
conda create -n phone-face-vision python=3.10 -y
conda activate phone-face-vision
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-vision.txt

conda create -n phone-face-qwen python=3.10 -y
conda activate phone-face-qwen
python -m pip install 'huggingface_hub[cli]' uv
uv pip install vllm --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly
hf download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B
```

模型来源：[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)。`models/` 已被 `.gitignore` 排除；若已在别处下载，设置 `QWEN_MODEL_DIR` 指向含 `config.json` 的模型目录即可。

## 四卡启动

四卡分配为：GPU 0、1、2 各运行一个 Qwen3.5-4B 实例，第四张空闲卡运行 YOLO 与姿态估计。脚本默认使用 GPU 3；如果该卡已被占用，可通过 `DEVICE` 和 `LIVE_POSE_DEVICE` 指定另一张卡。三个 Qwen 实例分别监听本机的 8101–8103 端口，网页监听 7860。脚本从仓库根目录定位文件，不写死用户名或机器路径。

```bash
# 终端 1：激活安装了 vLLM 的环境
conda activate phone-face-qwen
./scripts/start_four_card_qwen.sh

# 终端 2：激活视觉环境
conda activate phone-face-vision
./scripts/start_four_card_web.sh
# 例如 GPU 3 正忙，视觉处理改用 GPU 5：
# DEVICE=5 LIVE_POSE_DEVICE=5 ./scripts/start_four_card_web.sh
```

打开 `http://服务器IP:7860/dashboard` 上传视频或连接摄像头；把自己的 `video_test/test.mp4` 放好后可打开 `/demo?video=test`，低光样本如命名为 `video_test/20261003_C8920.MP4` 可打开 `/demo?video=lowlight`。**启动网页不要求预置测试视频**，缺少测试视频只影响相应的演示页。设置 `NETWORK_SOURCE=rtsp://127.0.0.1:8554/live/action6` 可为网页预填 MediaMTX 流地址，随后在网页点击启动。直播采用“处理最新帧”策略，避免推流帧率高于推理速度时延迟无限累积；极短动作可能因跳帧漏检。文件视频仍逐帧处理。

只有一张 GPU 时，也可自行启动一个 Qwen 服务并设置 `QWEN_VL_URL`，然后用 `DEVICE=0 python -m src.phone_face.web` 启动网页；Qwen 与 YOLO 共卡可能更慢。环境变量 `LIVE_IMG_SIZE` 控制 YOLO 内部推理尺寸，默认 1280；检测接收视频解码后的完整帧。人物上下文和手机图送 Qwen 前不缩小，人脸画廊和 FTP 导出均为原帧裁剪的无损 PNG，网页左侧预览则是 JPEG。

拍摄动作由连续帧的手机与手腕关联触发，再交给 Qwen 核对；确认后从动作前后约 `CAPTURE_FACE_WINDOW=1.5` 秒的候选画面挑选人脸，侧脸也可收录，不再为了等待正脸而长期跟踪。连续同一人物会复用最近的 SFace 特征，默认每 `FACE_REID_REFRESH_INTERVAL=0.5` 秒刷新，新人物、重入或相邻多人情况立即重算；设为 `0` 可关闭复用。人脸框短边默认至少 `CAPTURE_MIN_FACE_SIDE=24` 像素，可在启动网页前设置该环境变量调整；这是截图候选门槛，不能让 YOLO 检出原本漏掉的人脸。低光人脸的初筛最低亮度为 `CAPTURE_MIN_FACE_BRIGHTNESS=18`，后续仍检查对比度、清晰度及 Qwen 判定。每人画廊最多保留 5 张图。

同一台四卡 RTX 3090 服务器上的 `test.mp4`（720×1280、30 fps、724 帧、视频时长 24.13 秒）实测：视觉卡为 GPU 5，Qwen 使用 GPU 0–2。默认 24 像素门槛下完整离线处理耗时约 84.1 秒，生成 7 组画廊人物、28 张截图；把门槛设为 96 像素时耗时约 39.5 秒，生成 2 组、10 张。这里的“组”是算法建立的人物轨迹，并非人工核实的不同人数。默认 24 像素可纳入远处小脸，但会显著增加 Qwen 核对量，不能把离线耗时直接当作固定直播延迟。直播采用最新帧读取，处理慢于输入帧率时会跳过过期帧，可能漏掉极短动作；候选 Qwen 核对队列满时还可能阻塞处理，右侧截图会晚于左侧画面出现。测试视频不随仓库发布。

## 命令行与 FTP

命令行生成打框 MP4：

```bash
DEVICE=3 ./scripts/run_phone_face_video.sh /path/to/input.mp4 result_name
```

网页将已进入右侧画廊的 Top-5 原图写入 `outputs/ftp_spool`：同一会话、同一人物各有一个目录，文件为 `01.png` 至 `05.png` 及 `manifest.json`。画廊排名改变时对应 PNG 会更新。FTP 上传器默认每秒检查一次，每轮上传一张新增或更新的截图；写入待传目录本身不代表 FTP 已上传。上传器可读环境变量 `FTP_HOST`、`FTP_USER`、`FTP_PASSWORD`、`FTP_REMOTE_ROOT`，或仓库根目录下私有的 `ftp.py`。后者可从 [ftp.example.py](ftp.example.py) 复制并填写，已被 `.gitignore` 排除。

```bash
conda activate phone-face-vision
FTP_HOST=ftp.example.com FTP_USER=your_user FTP_PASSWORD=your_password \
  python scripts/ftp_upload_faces.py --spool outputs/ftp_spool --interval 1
```

FTP 上传失败时待传 PNG 会留在本地，网络恢复后继续。`outputs/` 不提交。不要把真实 FTP 地址、密码或拍摄对象影像推到公开仓库。

## Docker 历史镜像

`phone-face-qwen35:ftp-export-v4` 是独立的旧镜像，**不会自动包含本仓库现在的低光召回、PNG 画廊与提速修改**。若要运行这版代码，使用上述原生环境，或据此重新构建镜像。仓库没有发布包含 Qwen 权重的一体化 Docker 镜像。
