import cv2 as cv
import numpy as np
from PIL import Image
import os
import json

# 人脸图像数据库、训练模型和名字映射文件路径
dataset_path = 'dataset'  # 人脸图像数据存储路径
trainer_path = 'trainer'  # 训练好的模型存储路径
names_mapping_path = 'names_mapping.json'  # 名字到 ID 的映射文件路径

def main():
    # 如果数据集目录不存在，则创建
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    if not os.path.exists(trainer_path):
        os.makedirs(trainer_path)

    while True:
        print("请选择一个操作：")
        print("1. 数据采集")
        print("2. 模型训练")
        print("3. 实时识别")
        print("4. 退出")
        choice = input("输入你的选择：")
        if choice == '1':
            face_id = input_names()  # 输入名字并获取 ID
            capture_faces(face_id)  # 捕获人脸图像
        elif choice == '2':
            train_model()  # 训练模型
        elif choice == '3':
            recognize_faces()  # 调用实时识别函数
        elif choice == '4':
            print("退出程序")
            break
        else:
            print("无效的选择，请重试。")

def input_names():
    """
    输入名字并自动分配 ID，保存到名字映射文件中
    """
    # 读取已存在的名字到 ID 映射
    if os.path.exists(names_mapping_path):
        with open(names_mapping_path, 'r') as f:
            names_mapping = json.load(f)
    else:
        names_mapping = {}

    # 自动分配 ID
    new_id = max(names_mapping.values(), default=0) + 1
    name = input(f"请输入名字（ID 将自动分配为 {new_id}）：")
    names_mapping[name] = new_id

    # 保存名字到 ID 映射到文件
    with open(names_mapping_path, 'w') as f:
        json.dump(names_mapping, f)

    return new_id

def capture_faces(face_id):
    """
    捕获人脸图像并保存到指定路径
    :param face_id: 人脸 ID
    """
    # 初始化并启动实时视频捕获
    # 如果在RK3568开发板中运行，需要改为：cam = cv.VideoCapture(9)
    cam = cv.VideoCapture(0)
    #cam = cv.VideoCapture(9)
    if not cam.isOpened():
        print("错误：无法打开摄像头。")
        return
    cam.set(3, 640)  # 设置视频宽度
    cam.set(4, 480)  # 设置视频高度
    # 加载人脸检测模型
    face_detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    print("\n[信息] 正在初始化人脸捕捉。看着摄像头并等待...")
    # 初始化样本计数
    count = 0
    detect_interval = 5  # 每隔多少帧进行一次检测
    frame_count = 0  # 帧计数器

    while True:
        ret, img = cam.read()
        if not ret:
            print("抓取帧失败")
            break
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
        # 每隔一定的帧数进行一次人脸检测
        if frame_count % detect_interval == 0:
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 画出检测到的人脸矩形框
                count += 1
                # 保存捕获的图像到数据集文件夹中
                cv.imwrite(os.path.join(dataset_path, f"User.{face_id}.{count}.jpg"), gray[y:y + h, x:x + w])

                if count >= 10:  # 捕获到 10 张图像后停止
                    break
        cv.imshow('image', img)  # 显示当前帧图像
        frame_count += 1  # 帧计数器加 1
        k = cv.waitKey(1) & 0xff  # 按 'ESC' 键退出
        if k == 27 or count >= 10:
            break
    cam.release()
    cv.destroyAllWindows()

def train_model():
    """
    训练人脸识别模型并保存
    """
    # 创建 LBPH 面部识别器
    recognizer = cv.face.LBPHFaceRecognizer_create()
    detector = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

    def getImagesAndLabels(path):
        """
        从指定路径获取图像和标签
        :param path: 图像路径
        :return: 图像样本和对应的 ID 列表
        """
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')  # 转换为灰度图像
            img_numpy = np.array(PIL_img, dtype='uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)
        return faceSamples, ids

    print("\n[信息] 正在训练人脸识别模型。请稍候...")
    faces, ids = getImagesAndLabels(dataset_path)
    recognizer.train(faces, np.array(ids))  # 训练模型
    # 保存模型
    recognizer.write('trainer/trainer.yml')
    print(f"\n[信息] {len(np.unique(ids))} 张人脸已训练。程序结束")


def recognize_faces():
    """
    实时识别摄像头中的人脸
    """
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')  # 读取训练好的模型
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv.CascadeClassifier(cascadePath)
    font = cv.FONT_HERSHEY_SIMPLEX
    # 读取名字到 ID 的映射
    if os.path.exists(names_mapping_path):
        with open(names_mapping_path, 'r') as f:
            names_mapping = json.load(f)
        names = {v: k for k, v in names_mapping.items()}  # 反反映射: ID -> 名字
    else:
        print("错误: 未找到名字映射文件。")
        return

    # 初始化并启动实时视频捕获
    cam = cv.VideoCapture(0)  # 普通设备
    # cam = cv.VideoCapture(9)  # RK3568开发板
    if not cam.isOpened():
        print("错误: 无法打开摄像头。")
        return
    cam.set(3, 640)  # 宽度
    cam.set(4, 480)  # 高度
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()
        if not ret:
            print("抓取帧失败")
            break

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence < 100:
                name = names.get(id, "未知")
            else:
                name = "未知"
            cv.putText(img, name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)

        # 显示画面（移到人脸循环外，确保始终显示）
        cv.imshow('camera', img)

        # 检测ESC键（移到此处，无论是否有人脸都能响应）
        k = cv.waitKey(10) & 0xff
        if k == 27:  # 按ESC键退出
            break

    print("[信息] 退出实时识别。")
    cam.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()