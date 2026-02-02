## 项目简介

本仓库是一个基于 Python、OpenCV 和 MediaPipe 的计算机视觉实验项目，包含 **人脸检测 / 识别** 与 **手势（手指数量）识别** 等多个小实验，用于学习和演示深度学习与传统视觉算法在实际场景中的应用。

项目中提供了完整的示例代码、级联分类器模型（Haar/LBP）、训练好的模型文件以及测试图片/视频素材，适合作为入门与教学示例使用。

---

## 功能特性

- **人脸检测与识别（RK3568 模块）**
  - 使用 OpenCV 级联分类器进行人脸检测
  - 提供训练好的 `trainer.yml` 模型和 `names_mapping.json` 映射示例
  - 面向嵌入式平台（如 RK3568）的人脸识别实验脚本

- **手势 / 手指数量识别（hands 模块）**
  - 基于 MediaPipe Hands 检测 21 个关键点
  - 自动区分左右手，统计 **0–5 根手指** 的张开数量
  - 根据张开的手指数量，在画面左上角显示对应的示意图片（`hands/fingers/0.png` ~ `5.png`）

- **OpenCV 基础实验（OpenCV 模块）**
  - 人脸检测、眼睛检测、人体检测、车牌检测等基础示例
  - 提供多种测试图片与视频（`body.jpg`, `face.jpg`, `traffic.flv`, `video1.mp4`, `vtest.avi` 等）
  - 多个独立小脚本演示不同的经典视觉任务

---

## 环境依赖

建议使用 **Python 3.8+**，并安装以下主要依赖库：

pip install opencv-python mediapipe numpy其他依赖（如果有）可根据运行时报错信息，通过 `pip install ...` 安装。

---


###项目目录结构如下（省略部分文件）：

Face-Recognition/
├── OpenCV/
│   ├── data/                     # 各类 Haar 级联模型类 Haar 级联模型睛、人体、   ├── body.jpg / face.jpg ...   #   #像
│   ├── traffic.flv / video1.mp4  # 测试视频
│  
│   ├1_*.py ~ t t1_*.py ~ t   # 多个 OpenCV 基础实验脚本
│   ...
├── RK3568*/                      # RK3568 平台相关的人脸识别实验
│的人脸识别实验
│*.py                    #                    #cascade_frontalface_default.xml
│   ├──
│   ├──ade_animeface.xml
│   ├── names_mapping.json
│.json
│└rainer/
│       └── trainer.yml           # 训练 trainer.yml           # 训练ands/
│il handutil         # 基于 MediaPipe               # 基于 MediaPipe                 # 手指数量识别主程序
│   └── fingers/0.png   └── fingers/0.pngpng       # 对应 .idea/                        # 开                        # 开
