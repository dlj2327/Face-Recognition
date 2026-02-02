import cv2
img = cv2.imread("body.jpg")
# 加载识别人脸的级联分类器
faceCascade = cv2.CascadeClassifier("data/haarcascade_fullbody.xml")
body = faceCascade.detectMultiScale(img, 1.15)
for x, y, w, h in body:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()