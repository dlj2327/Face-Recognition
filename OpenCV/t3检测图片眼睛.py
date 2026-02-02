import cv2
# 多张人脸的图片加载
img = cv2.imread('eye.jpg')

# 加载识别眼睛的级联分类器
faceCascade = cv2.CascadeClassifier("data/haarcascade_eye.xml")
# 识别出图像的所有眼睛，以一定的缩放比例显示
eyes = faceCascade.detectMultiScale(img, 1.15, 50)
# 遍历所有人脸的区域
for (x,y,w,h) in eyes:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
# 展示图片
cv2.imshow('img1',img)
# 等待用户按键，保持窗口打开
cv2.waitKey()
# 关闭窗口
cv2.destroyAllWindows()