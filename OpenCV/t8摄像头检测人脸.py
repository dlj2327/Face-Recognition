import cv2

def face_demo(img):
    # 使用cv2.flip 翻转图像 1表示沿X轴翻转（水平）
    frame = cv2.flip(img, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
    faces = faceCascade.detectMultiScale(gray, 1.15)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imshow('face', img)

# 初始化摄像头对象 电脑一般为0
video = cv2.VideoCapture(0)
while True:
    retval, img = video.read()
    # 显示原始视频帧
    #cv2.imshow('Video', img)
    # 调用face_demo函数
    face_demo(img)
    key = cv2.waitKey(1)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()