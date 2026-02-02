import argparse
from pathlib import Path

import cv2 as cv


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_cascade() -> Path:
    return _repo_root() / "models" / "haarcascade_frontalface_default.xml"


def _load_cascade(cascade_path: Path) -> cv.CascadeClassifier:
    cascade = cv.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"无法加载级联分类器：{cascade_path}")
    return cascade


def _draw_faces(frame, faces):
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def detect_in_image(image_path: Path, cascade_path: Path) -> None:
    img = cv.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"无法读取图片：{image_path}")

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cascade = _load_cascade(cascade_path)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5)
    _draw_faces(img, faces)

    cv.imshow("faces(image)", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def detect_in_video(capture: cv.VideoCapture, cascade_path: Path, window_name: str) -> None:
    cascade = _load_cascade(cascade_path)
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5)
        _draw_faces(frame, faces)
        cv.imshow(window_name, frame)
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break

    capture.release()
    cv.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="OpenCV face detection (image / video / webcam).")
    parser.add_argument("--cascade", type=str, default=str(_default_cascade()))
    parser.add_argument("--image", type=str, help="Image path")
    parser.add_argument("--video", type=str, help="Video path")
    parser.add_argument("--camera", type=int, help="Camera index, e.g. 0")
    args = parser.parse_args()

    cascade_path = Path(args.cascade)

    chosen = [x is not None for x in (args.image, args.video, args.camera)]
    if sum(chosen) != 1:
        raise SystemExit("请只选择一种输入：--image 或 --video 或 --camera")

    if args.image:
        detect_in_image(Path(args.image), cascade_path)
        return

    if args.video:
        cap = cv.VideoCapture(str(args.video))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频：{args.video}")
        detect_in_video(cap, cascade_path, "faces(video)")
        return

    cap = cv.VideoCapture(int(args.camera))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头：{args.camera}")
    detect_in_video(cap, cascade_path, "faces(webcam)")


if __name__ == "__main__":
    main()


