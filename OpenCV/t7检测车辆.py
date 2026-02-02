import cv2

# 使用KNN算法创建背景减除对象，detectShadows=True表示检测阴影
bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)
# 初始化视频捕捉对象，用于读取视频文件
camera = cv2.VideoCapture('traffic.flv')

# 循环直到视频结束或用户退出
while True:
    # 读取视频的下一帧
    ret, frame = camera.read()
    # 如果ret为False，表示视频结束或读取出错，退出循环
    if not ret:
        break
    # 使用背景减除算法处理当前帧，得到前景掩模
    fgmask = bs.apply(frame)
    # 对前景掩模应用阈值化，得到二值图像，244是阈值
    th = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)[1]
    # 创建一个椭圆形态的结构元素
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3))
    # 对二值图像进行膨胀操作，增强前景物体的轮廓
    dilated = cv2.dilate(th, element, 2)
    # 在膨胀后的图像中查找所有轮廓
    contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历所有检测到的轮廓
    for c in contours:
        # 如果轮廓的面积大于1000像素，认为是有效的物体
        if cv2.contourArea(c) > 1000:
            # 计算轮廓的边界框
            (x, y, w, h) = cv2.boundingRect(c)
            # 在原始帧上绘制矩形框，标记出前景物体的位置
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    # 显示处理后的视频帧
    cv2.imshow('video', frame)
    # 按'q'键退出循环
    if cv2.waitKey(30) & 0xff == ord('q'):
        break

# 释放视频捕捉对象
camera.release()
# 销毁所有OpenCV窗口
cv2.destroyAllWindows()