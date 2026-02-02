import argparse
import json
import re
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image


RE_USER_FILE = re.compile(r"^User\.(?P<id>\d+)\.(?P<idx>\d+)\.jpg$", re.IGNORECASE)


def _require_cv_face():
    if not hasattr(cv, "face"):
        raise RuntimeError(
            "当前 OpenCV 不包含 cv2.face（需要 opencv-contrib-python）。\n"
            "请安装/替换为：pip install opencv-contrib-python"
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _data_dir() -> Path:
    # All privacy-sensitive outputs live under this folder and are gitignored by default.
    return Path(__file__).resolve().parent / "data"


def _paths():
    data_dir = _data_dir()
    dataset_dir = data_dir / "dataset"
    trainer_dir = data_dir / "trainer"
    model_file = trainer_dir / "trainer.yml"
    names_file = data_dir / "names_mapping.json"
    cascade_file = _repo_root() / "models" / "haarcascade_frontalface_default.xml"
    return data_dir, dataset_dir, trainer_dir, model_file, names_file, cascade_file


def _load_names_mapping(names_file: Path) -> dict:
    if not names_file.exists():
        return {}
    try:
        with names_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        # name -> id(int)
        cleaned = {}
        for k, v in data.items():
            if isinstance(k, str) and isinstance(v, int):
                cleaned[k] = v
        return cleaned
    except Exception:
        return {}


def _save_names_mapping(names_file: Path, mapping: dict) -> None:
    names_file.parent.mkdir(parents=True, exist_ok=True)
    with names_file.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def _assign_id(mapping: dict, name: str) -> int:
    if name in mapping and isinstance(mapping[name], int):
        return mapping[name]
    new_id = (max(mapping.values()) if mapping else 0) + 1
    mapping[name] = int(new_id)
    return int(new_id)


def _consent_or_exit():
    print("隐私/合规提示：采集与存储人脸数据属于生物识别数据处理。")
    print("请确认你已获得被采集者明确授权，并遵守当地法律法规。")
    confirm = input("如果你确认拥有合法授权并同意继续，请输入 YES：").strip()
    if confirm != "YES":
        raise SystemExit("已取消。")


def collect(name: str, camera_index: int, max_samples: int, detect_interval: int) -> None:
    _consent_or_exit()

    data_dir, dataset_dir, trainer_dir, model_file, names_file, cascade_file = _paths()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    trainer_dir.mkdir(parents=True, exist_ok=True)

    mapping = _load_names_mapping(names_file)
    face_id = _assign_id(mapping, name)
    _save_names_mapping(names_file, mapping)

    face_detector = cv.CascadeClassifier(str(cascade_file))
    if face_detector.empty():
        raise RuntimeError(f"无法加载人脸检测模型：{cascade_file}")

    cam = cv.VideoCapture(int(camera_index))
    if not cam.isOpened():
        raise RuntimeError("无法打开摄像头。")

    cam.set(3, 640)
    cam.set(4, 480)

    print(f"[信息] 开始采集：name={name!r}, id={face_id}，目标样本数={max_samples}")
    print(f"[信息] 数据将写入：{dataset_dir}（默认不会提交到 GitHub）")

    count = 0
    frame_count = 0

    try:
        while True:
            ret, img = cam.read()
            if not ret:
                print("[警告] 抓取帧失败，退出。")
                break

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            if frame_count % max(1, int(detect_interval)) == 0:
                faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                for (x, y, w, h) in faces:
                    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    count += 1
                    out_file = dataset_dir / f"User.{face_id}.{count}.jpg"
                    cv.imwrite(str(out_file), gray[y : y + h, x : x + w])
                    if count >= max_samples:
                        break

            cv.imshow("collect", img)
            frame_count += 1

            k = cv.waitKey(1) & 0xFF
            if k == 27 or count >= max_samples:
                break
    finally:
        cam.release()
        cv.destroyAllWindows()

    print(f"[信息] 采集完成：已保存 {count} 张人脸样本。")


def train() -> None:
    _require_cv_face()

    data_dir, dataset_dir, trainer_dir, model_file, names_file, cascade_file = _paths()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    trainer_dir.mkdir(parents=True, exist_ok=True)

    detector = cv.CascadeClassifier(str(cascade_file))
    if detector.empty():
        raise RuntimeError(f"无法加载人脸检测模型：{cascade_file}")

    image_paths = [p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"]
    if not image_paths:
        raise RuntimeError(f"数据集为空：{dataset_dir}\n请先运行 collect 采集样本。")

    face_samples = []
    ids = []

    for image_path in image_paths:
        m = RE_USER_FILE.match(image_path.name)
        if not m:
            continue
        face_id = int(m.group("id"))

        pil_img = Image.open(image_path).convert("L")
        img_numpy = np.array(pil_img, dtype="uint8")
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y : y + h, x : x + w])
            ids.append(face_id)

    if not ids:
        raise RuntimeError("未能从数据集中提取到可用人脸（请确保图片中有人脸且清晰）。")

    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.train(face_samples, np.array(ids))
    recognizer.write(str(model_file))

    print(f"[信息] 训练完成：人数={len(set(ids))}，样本数={len(ids)}")
    print(f"[信息] 模型已保存：{model_file}（默认不会提交到 GitHub）")


def recognize(camera_index: int, threshold: float) -> None:
    _require_cv_face()

    data_dir, dataset_dir, trainer_dir, model_file, names_file, cascade_file = _paths()
    if not model_file.exists():
        raise RuntimeError(f"未找到模型：{model_file}\n请先运行 train。")

    face_cascade = cv.CascadeClassifier(str(cascade_file))
    if face_cascade.empty():
        raise RuntimeError(f"无法加载人脸检测模型：{cascade_file}")

    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.read(str(model_file))

    mapping = _load_names_mapping(names_file)
    id_to_name = {v: k for k, v in mapping.items()}

    cam = cv.VideoCapture(int(camera_index))
    if not cam.isOpened():
        raise RuntimeError("无法打开摄像头。")

    cam.set(3, 640)
    cam.set(4, 480)
    min_w = 0.1 * cam.get(3)
    min_h = 0.1 * cam.get(4)

    font = cv.FONT_HERSHEY_SIMPLEX

    print("[信息] 开始实时识别，按 ESC 退出。")
    try:
        while True:
            ret, img = cam.read()
            if not ret:
                print("[警告] 抓取帧失败，退出。")
                break

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(min_w), int(min_h)),
            )

            for (x, y, w, h) in faces:
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_id, confidence = recognizer.predict(gray[y : y + h, x : x + w])

                # LBPH: lower confidence is better. Use a configurable threshold.
                if confidence <= float(threshold):
                    name = id_to_name.get(face_id, f"ID:{face_id}")
                else:
                    name = "Unknown"

                cv.putText(img, f"{name} ({confidence:.1f})", (x + 5, y - 5), font, 0.8, (255, 255, 255), 2)

            cv.imshow("recognize", img)
            k = cv.waitKey(10) & 0xFF
            if k == 27:
                break
    finally:
        cam.release()
        cv.destroyAllWindows()

    print("[信息] 已退出实时识别。")


def main():
    parser = argparse.ArgumentParser(description="LBPH face recognition (privacy-safe defaults).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_collect = sub.add_parser("collect", help="Collect face samples from camera into local dataset folder (gitignored).")
    p_collect.add_argument("--name", required=True, help="Person name/label (stored locally; gitignored).")
    p_collect.add_argument("--camera-index", type=int, default=0)
    p_collect.add_argument("--max-samples", type=int, default=10)
    p_collect.add_argument("--detect-interval", type=int, default=5, help="Detect every N frames to reduce CPU usage.")

    p_train = sub.add_parser("train", help="Train LBPH model from local dataset (gitignored).")

    p_rec = sub.add_parser("recognize", help="Recognize faces from camera using trained model (gitignored).")
    p_rec.add_argument("--camera-index", type=int, default=0)
    p_rec.add_argument("--threshold", type=float, default=100.0, help="LBPH confidence threshold (lower is stricter).")

    args = parser.parse_args()

    if args.cmd == "collect":
        collect(args.name, args.camera_index, args.max_samples, args.detect_interval)
    elif args.cmd == "train":
        train()
    elif args.cmd == "recognize":
        recognize(args.camera_index, args.threshold)
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()


