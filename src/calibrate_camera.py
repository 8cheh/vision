"""第五节：用 left*.jpg 棋盘图检测角点、标定相机并去畸变。"""

from pathlib import Path

import cv2
import numpy as np

CALIB = Path(__file__).resolve().parents[1] / "data" / "calibration"
RESULTS = CALIB.parents[1] / "results"
PATTERN = (9, 6)                                                      # 内角点列数、行数
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def imread(path: Path) -> np.ndarray:
    """cv2.imread 遇到含中文的路径会静默失败，改用 imdecode。"""
    img = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"解码失败：{path}")
    return img


def imwrite(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise OSError(f"编码失败：{path}")
    path.write_bytes(buf.tobytes())


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    images = sorted(CALIB.glob("left*.jpg"))
    if not images:
        raise SystemExit(f"{CALIB} 中没有 left*.jpg")

    objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2)   # 棋盘平面，Z=0，间距 1

    obj_points, img_points = [], []
    size = (0, 0)
    for path in images:
        gray = cv2.cvtColor(imread(path), cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])                             # (宽, 高)
        ok, corners = cv2.findChessboardCorners(
            gray, PATTERN, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            print(f"{path.name}: 未检测到内角点")
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
        obj_points.append(objp)
        img_points.append(corners)
        marked = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(marked, PATTERN, corners, ok)
        imwrite(RESULTS / f"corners_{path.stem}.jpg", marked)
    print(f"成功检测 {len(img_points)}/{len(images)} 张，内角点 {PATTERN[0]}x{PATTERN[1]}，尺寸 {size}")
    if len(img_points) < 3:
        raise SystemExit("成功检测的图片少于 3 张，无法可靠标定")

    rms, k, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    print(f"内参矩阵 K =\n{k}\n畸变系数 = {dist.ravel()}\nRMS 重投影误差 = {rms:.4f} 像素")
    np.savez(str(RESULTS / "camera_params.npz"), camera_matrix=k, dist_coeffs=dist, image_size=np.array(size))

    src = imread(CALIB / "left13.jpg")
    h, w = src.shape[:2]
    new_k = cv2.getOptimalNewCameraMatrix(k, dist, (w, h), 1, (w, h))[0]
    pair = np.hstack([src, cv2.undistort(src, k, dist, newCameraMatrix=new_k)])
    pair = cv2.resize(pair, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    for text, x in (("before", 10), ("undistorted", pair.shape[1] // 2 + 10)):
        cv2.putText(pair, text, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    imwrite(RESULTS / "undistort_compare.jpg", pair)
    print("角点图与去畸变对比 -> results/")


if __name__ == "__main__":
    main()
