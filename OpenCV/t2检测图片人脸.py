import cv2

# 多张人脸的图片加载
img = cv2.imread('face.jpg')
# 转为灰度图的目的减少计算量
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 加载识别人脸的级联分类器
faceCascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
# 识别出图像的所有人脸，以一定的缩放比例显示
faces = faceCascade.detectMultiScale(gray, 1.15)
# 遍历所有人脸的区域
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
# 展示图片
cv2.imshow('img1', img)
# 等待用户按键，保持窗口打开
cv2.waitKey()
# 关闭窗口
cv2.destroyAllWindows()