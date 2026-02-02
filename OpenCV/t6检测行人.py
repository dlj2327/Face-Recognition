import cv2
import numpy as np

# 初始化视频捕捉对象，用于读取视频文件
camera = cv2.VideoCapture('vtest.avi')
# 创建一个椭圆结构的元素，用于形态学操作
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
# 创建一个5x5的矩阵，用于图像处理中的膨胀或腐蚀操作
kernel = np.ones((5, 5), np.uint8)
# 初始化背景变量，开始时为None
background = None

while True:
    # 读取视频的下一帧
    ret, frame = camera.read()
    # 如果正确读取帧，ret为True，否则为False，表示视频结束或读取出错
    if not ret:
        break  # 如果没有帧可以读取，退出循环
    # 如果背景还未初始化，则使用第一帧作为背景
    if background is None:
        # 将当前帧转换为灰度图像
        background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 对背景进行高斯模糊处理，以减少图像噪声
        background = cv2.GaussianBlur(background, (21, 21), 0)
        continue  # 继续下一次循环，直到读取到第一帧

    # 将当前帧转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 对当前帧进行高斯模糊处理，以减少图像噪声
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    # 计算当前帧与背景的差分图，以识别移动物体
    diff = cv2.absdiff(background, gray_frame)
    # 应用阈值化，将差分图转换为二值图像，便于轮廓检测
    diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    # 对二值图像进行膨胀操作，填补物体内部的空洞
    diff = cv2.dilate(diff, es, 2)
    # 在膨胀后的图像中查找轮廓
    cnts, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历所有检测到的轮廓
    for c in cnts:
        # 如果轮廓面积小于1500像素，则忽略它，可能是噪声
        if cv2.contourArea(c) < 1500:
            continue
        # 计算轮廓的边界框
        (x, y, w, h) = cv2.boundingRect(c)
        # 在原始帧上绘制矩形框，标记出移动物体的位置
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    # 显示结果帧
    cv2.imshow("contours", frame)
    # 按'q'键退出
    if cv2.waitKey(30) & 0xff == ord("q"):
        break

# 释放窗口，关闭程序
cv2.destroyAllWindows()
# 释放视频捕捉对象
camera.release()