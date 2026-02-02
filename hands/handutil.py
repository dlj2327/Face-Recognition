import cv2 as cv
import mediapipe as mp


class HandDetector:
    """手势识别类，用于检测手部关键点并返回坐标"""

    def __init__(self, mode=False, max_hands=2, complexity=1, detection_con=0.5, track_con=0.5):
        """
        初始化手势识别参数
        :param mode: 是否为静态图片（默认False：实时视频）
        :param max_hands: 最大检测手数（默认2）
        :param complexity: 模型复杂度（默认1）
        :param detection_con: 检测置信度阈值（默认0.5）
        :param track_con: 追踪置信度阈值（默认0.5）
        """
        self.mode = mode
        self.max_hands = max_hands
        self.complexity = complexity
        self.detection_con = detection_con
        self.track_con = track_con

        # 初始化MediaPipe手势模型
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=mode,
            max_num_hands=max_hands,
            model_complexity=complexity,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con
        )
        self.results = None  # 用于存储检测结果

    def find_hands(self, img):
        """
        检测手部并绘制关键点
        :param img: 输入图像（BGR格式）
        :return: 绘制关键点后的图像
        """
        # 转换BGR到RGB（MediaPipe需要RGB格式）
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # 检测手部
        self.results = self.hands.process(img_rgb)

        # 若检测到手，绘制关键点及连接
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    img_rgb,  # 绘制的图像
                    hand_lms,  # 手部关键点
                    mp.solutions.hands.HAND_CONNECTIONS  # 关键点连接关系
                )

        # 转换回BGR格式（OpenCV默认BGR）
        img = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)
        return img



    def find_positions(self, img, hand_no=0):
        """
        获取手部关键点的坐标
        :param img: 输入图像（用于获取尺寸）
        :param hand_no: 手的索引（默认第1只手）
        :return: 关键点列表（id, x, y）
        """
        lms_list = []
        if self.results.multi_hand_landmarks:
            # 获取指定手的关键点
            hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(hand.landmark):
                # 转换比例坐标为像素坐标
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lms_list.append([id, cx, cy])
        return lms_list