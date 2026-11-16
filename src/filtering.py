"""第四节：给灰度图加椒盐噪声，比较 3x3 与 9x9 中值滤波。"""

from pathlib import Path

import cv2
import numpy as np

PHOTO = Path(__file__).resolve().parents[1] / "data" / "photo.jpg"
RESULTS = PHOTO.parents[1] / "results"

RATIO, SEED = 0.05, 42                        # 噪声比例、随机种子
ZOOM, SCALE = (0.42, 0.20, 0.16, 0.16), 2     # 放大区域相对 x, y, w, h 与倍数


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


def tile(img: np.ndarray, text: str, width: int = 400) -> np.ndarray:
    """缩放到统一宽度，并在上方加一条黑底白字标题栏。"""
    img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    bar = np.full((26, width, 3), 0, np.uint8)
    cv2.putText(bar, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def salt_pepper(gray: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    """随机挑 ratio 的像素，一半置 0（椒）、一半置 255（盐）。"""
    rng = np.random.default_rng(seed)
    noisy, idx = gray.copy(), rng.choice(gray.size, int(gray.size * ratio), replace=False)
    noisy.flat[idx[: idx.size // 2]] = 0
    noisy.flat[idx[idx.size // 2:]] = 255
    return noisy


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    gray = cv2.cvtColor(imread(PHOTO), cv2.COLOR_BGR2GRAY)

    noisy = salt_pepper(gray, RATIO, SEED)
    panels = [("Gray", gray), (f"Salt & pepper {RATIO:.0%}", noisy),
              ("Median 3x3", cv2.medianBlur(noisy, 3)), ("Median 9x9", cv2.medianBlur(noisy, 9))]
    print(f"噪声比例 {RATIO:.0%}，种子 {SEED}，核 3x3 / 9x9")

    tiles = [tile(cv2.cvtColor(p, cv2.COLOR_GRAY2BGR), t) for t, p in panels]
    imwrite(RESULTS / "noise_compare.jpg", np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))

    h, w = gray.shape
    x, y, zw, zh = int(ZOOM[0] * w), int(ZOOM[1] * h), int(ZOOM[2] * w), int(ZOOM[3] * h)
    zoomed = [tile(cv2.resize(cv2.cvtColor(p[y:y + zh, x:x + zw], cv2.COLOR_GRAY2BGR), None,
                              fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST), t) for t, p in panels]
    imwrite(RESULTS / "noise_zoom_compare.jpg", np.vstack([np.hstack(zoomed[:2]), np.hstack(zoomed[2:])]))
    print(f"放大区域 x={x}, y={y}, {zw}x{zh}（{SCALE} 倍）-> results/noise_compare.jpg, noise_zoom_compare.jpg")


if __name__ == "__main__":
    main()
