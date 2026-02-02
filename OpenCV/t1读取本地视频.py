import cv2

# 打开视频文件 "video1.mp4"
video = cv2.VideoCapture("video1.mp4")
# 进入一个无限循环，用于逐帧读取和显示视频
while True:
    # 从视频捕获对象中读取一帧
    retval, image = video.read()
    # 检查是否成功读取了一帧
    if retval == True:
        # 如果成功读取，则在窗口中显示这一帧
        cv2.imshow("video1", image)
    else:
        # 如果未能读取（通常意味着视频结束），则退出循环
        break
    # 等待100毫秒并检查是否按下了键
    key = cv2.waitKey(100)
    # 如果按下的键是 ESC 键（ASCII码为27），则退出循环
    if key == 27:
        break
# 释放视频捕获对象，释放资源
video.release()
# 关闭所有OpenCV窗口
cv2.destroyAllWindows()