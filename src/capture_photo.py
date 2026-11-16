"""用电脑摄像头拍照：窗口内按 s 保存到 data/photo.jpg，按 q 退出。"""

import argparse
from pathlib import Path

import cv2
import numpy as np

PHOTO = Path(__file__).resolve().parents[1] / "data" / "photo.jpg"


def imwrite(path: Path, img: np.ndarray) -> None:
    """cv2.imwrite 遇到含中文的路径会静默失败，改用 imencode 写字节。"""
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise OSError(f"编码失败：{path}")
    path.write_bytes(buf.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description="摄像头拍照：s 保存，q 退出")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号，默认 0")
    parser.add_argument("--auto", action="store_true", help="无人值守：自动保存一张后退出")
    args = parser.parse_args()

    PHOTO.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"打开摄像头 {args.camera} 失败：确认设备存在且未被其他程序占用。")
        return 1

    for _ in range(30):  # 丢弃前几帧，等自动曝光稳定
        cap.read()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("读取画面失败，退出。")
                return 1
            cv2.imshow("camera | s: save  q: quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("已退出，未保存。")
                return 0
            if key == ord("s") or args.auto:
                imwrite(PHOTO, frame)
                print(f"已保存 {PHOTO}（{frame.shape[1]}x{frame.shape[0]}）")
                return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
