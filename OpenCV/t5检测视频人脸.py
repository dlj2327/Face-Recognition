import cv2

def face_demo(img):
    if img is None:  # 检查图像是否为空
        return
    # 将读取的图像转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 加载分类器文件
    faceCascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
    # 识别出图像的所有人脸
    faces = faceCascade.detectMultiScale(gray, 1.15)
    # 遍历所有人脸的区域
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow("faces", img)

video = cv2.VideoCapture("video1.mp4")
while True:  # 视频文件被打开
    retval, image = video.read()  # 读取视频文件
    if not retval:  # 如果没有更多的帧
        break  # 退出循环
    face_demo(image)
    key = cv2.waitKey(1)
    if key == 27:  # 27表示esc
        break
video.release()  # 关闭文件
cv2.destroyAllWindows()