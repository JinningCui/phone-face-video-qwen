# YOLO 权重

| 文件 | 用途 | SHA-256 |
| --- | --- | --- |
| `weights/phone_face/best.pt` | 项目训练的手机／人脸检测模型，类别 `0=phone, 1=face` | `bfb467b6b8059d5bb9bc405289853310224dcb863384872d39788f84016b6d70` |
| `weights/yolov8n-pose.pt` | 姿态估计模型 | `c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212` |
| `weights/face_reid/face_detection_yunet_2023mar.onnx` | 返回画面人物的人脸对齐 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `weights/face_reid/face_recognition_sface_2021dec.onnx` | 同一会话内的人脸特征提取 | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |

`best.pt` 内保存的训练数据和输出目录已改为相对路径，模型张量保持不变。`yolov8n-pose.pt` 源自 [Ultralytics YOLOv8 Pose](https://docs.ultralytics.com/tasks/pose/)。使用者应遵守上游模型和依赖的许可证。手机／人脸模型的训练数据及评测集没有随仓库发布；不要把此权重当作人脸身份识别模型。

YuNet 和 SFace 来自 [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models)，各自的许可证保存在 `weights/face_reid/`。特征仅在网页进程内存中用于本次会话关联，不作为实名身份判断。
