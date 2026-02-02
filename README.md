## Face-Recognition（隐私安全版）

本仓库只包含**人脸检测/人脸识别**相关代码，并默认避免把任何可能涉及隐私的数据提交到 GitHub：

- **不会提交**：人脸图片数据集、训练出的模型文件、姓名映射文件
- **会提交**：运行代码、依赖、以及用于人脸检测的公开级联模型文件（OpenCV Haar/LBP）

### 重要提醒（隐私与合规）

- **采集/存储/识别人脸属于生物识别数据处理**。请确保你在当地法律法规允许的范围内使用，并且获得被采集者的明确授权。
- 程序在执行“采集”前会要求你输入 `YES` 进行确认。

### 环境（Python 3.8）

你提到的解释器路径：

- `C:\Users\Lenovo\AppData\Local\Programs\Python\Python38\python.exe`

安装依赖（PowerShell）：

```powershell
& "C:\Users\Lenovo\AppData\Local\Programs\Python\Python38\python.exe" -m pip install -r requirements.txt
```

### 1) RK3568 / PC：LBPH 人脸识别（采集→训练→识别）

脚本：`rk3568_face_recognition/app.py`

- **采集（会写入本地数据目录，且默认被 .gitignore 忽略）**

```powershell
& "C:\Users\Lenovo\AppData\Local\Programs\Python\Python38\python.exe" rk3568_face_recognition\app.py collect --name "user1" --camera-index 0
```

- **训练（输出模型到本地 data/trainer/，默认忽略）**

```powershell
& "C:\Users\Lenovo\AppData\Local\Programs\Python\Python38\python.exe" rk3568_face_recognition\app.py train
```

- **实时识别（摄像头）**

```powershell
& "C:\Users\Lenovo\AppData\Local\Programs\Python\Python38\python.exe" rk3568_face_recognition\app.py recognize --camera-index 0
```

### 2) OpenCV：通用人脸检测（图片/视频/摄像头）

脚本：`opencv_face_detection/detect_faces.py`

- 图片：

```powershell
& "C:\Users\Lenovo\AppData\Local\Programs\Python\Python38\python.exe" opencv_face_detection\detect_faces.py --image "path\\to\\image.jpg"
```

- 视频：

```powershell
& "C:\Users\Lenovo\AppData\Local\Programs\Python\Python38\python.exe" opencv_face_detection\detect_faces.py --video "path\\to\\video.mp4"
```

- 摄像头：

```powershell
& "C:\Users\Lenovo\AppData\Local\Programs\Python\Python38\python.exe" opencv_face_detection\detect_faces.py --camera 0
```

### 目录结构

- `models/`: 公开的人脸检测模型（xml）
- `rk3568_face_recognition/`: LBPH 人脸识别（采集/训练/识别）
- `opencv_face_detection/`: 通用人脸检测示例
- `local_only/`: 原始材料（包含数据集/媒体/无关功能），默认不提交


