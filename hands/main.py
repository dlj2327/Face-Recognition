import cv2 as cv
import numpy as np
from hands.handutil import HandDetector  


def main():
    # 指尖索引列表（大拇指到小指的指尖关节ID）
    tip_ids = [4, 8, 12, 16, 20]

    # 加载代表0~5根手指的图片
    finger_img_list = [
        'fingers/0.png',
        'fingers/1.png',
        'fingers/2.png',
        'fingers/3.png',
        'fingers/4.png',
        'fingers/5.png',
    ]
    finger_list = []
    for img_path in finger_img_list:
        # 读取图片并存储到列表
        img = cv.imread(img_path)
        if img is not None:
            finger_list.append(img)
        else:
            print(f"警告：未找到图片 {img_path}")

    # 打开摄像头
    cap = cv.VideoCapture(0)
    # 创建手势识别对象
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = cv.flip(img, flipCode=1)  # 水平翻转图像（镜像效果）

        if success:
            # 检测并绘制手部关键点
            img = detector.find_hands(img)
            # 获取手部关节坐标列表
            lmslist = detector.find_positions(img)

            if len(lmslist) > 0:
                fingers = []
                for tid in tip_ids:
                    # 绘制指尖标记（绿色实心圆）
                    x, y = lmslist[tid][1], lmslist[tid][2]
                    cv.circle(img, center=(x, y), radius=10, color=(0, 255, 0), thickness=cv.FILLED)

                    # 判断手指开合状态（区分大拇指和其他手指）
                    if tid == 4:  # 大拇指逻辑（区分左右手）
                        # 通过食指和中指位置判断左右手
                        if lmslist[8][1] < lmslist[12][1]:
                            # 右手：大拇指指尖x < 指节x → 打开
                            if lmslist[tid][1] < lmslist[tid - 1][1]:
                                fingers.append(1)
                            else:
                                fingers.append(0)
                        else:
                            # 左手：大拇指指尖x > 指节x → 打开
                            if lmslist[tid][1] > lmslist[tid - 1][1]:
                                fingers.append(1)
                            else:
                                fingers.append(0)
                    else:  # 其他手指（食指到小指）
                        # 指尖y < 第二关节y → 打开（y坐标从上到下递增）
                        if lmslist[tid][2] < lmslist[tid - 2][2]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                # 统计打开的手指数量
                cnt = fingers.count(1)
                # 显示对应数量的手指图片（若图片存在）
                # if 0 <= cnt < len(finger_list) and finger_list[cnt] is not None:
                #     finger_img = finger_list[cnt]
                #     # 获取图片尺寸并贴到摄像头画面左上角
                #     w, h, c = finger_img.shape
                #     # 确保图片尺寸不超过摄像头画面
                #     if w <= img.shape[1] and h <= img.shape[0]:
                #         img[0:h, 0:w] = finger_img  # 注意坐标顺序：[y范围, x范围]
                # 找到对应的手势图片并显示
                finger_img = finger_list[cnt]
                w, h, c = finger_img.shape
                img[0:w, 0:h] = finger_img

            # 显示最终画面
            cv.imshow("Image", img)

        # 按 'q' 键退出
        k = cv.waitKey(1)
        if k == ord('q'):
            break

    # 释放资源
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()