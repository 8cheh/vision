"""第三节：用 NumPy 读写像素、切片、改色，并按 BGR 通道拆分图像。"""

from pathlib import Path

import cv2
import numpy as np

PHOTO = Path(__file__).resolve().parents[1] / "data" / "photo.jpg"
RESULTS = PHOTO.parents[1] / "results"

PIXEL = (0.50, 0.50)                 # 单像素的相对坐标 u, v
CROP = (0.40, 0.03, 0.32, 0.39)      # 裁剪区域相对 x, y, w, h
BLOCK = (0.03, 0.03, 0.12, 0.10)     # 改红区域相对 x, y, w, h


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
    """缩放到统一宽度，并在上方加一条黑底白字标题栏（putText 不支持中文）。"""
    img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    bar = np.full((26, width, 3), 0, np.uint8)
    cv2.putText(bar, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def box(w: int, h: int, f: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    """相对坐标 -> 像素坐标 (x, y, w, h)。"""
    return int(f[0] * w), int(f[1] * h), int(f[2] * w), int(f[3] * h)


def channel(img: np.ndarray, i: int) -> np.ndarray:
    """只保留第 i 个通道，其余置零，形状不变。"""
    out = np.zeros_like(img)
    out[:, :, i] = img[:, :, i]
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    img = imread(PHOTO)
    h, w, c = img.shape
    print(f"shape = {img.shape}  dtype = {img.dtype} -> 高 {h}, 宽 {w}, 通道 {c}")

    u, v = int(PIXEL[0] * w), int(PIXEL[1] * h)
    print(f"像素 (u={u}, v={v})：img[{v}, {u}] = {img[v, u]}")

    x, y, cw, ch = box(w, h, CROP)
    imwrite(RESULTS / "crop.jpg", img[y:y + ch, x:x + cw])
    bx, by, bw, bh = box(w, h, BLOCK)
    recolored = img.copy()
    recolored[by:by + bh, bx:bx + bw] = (0, 0, 255)      # BGR 红色
    imwrite(RESULTS / "recolored.jpg", recolored)
    print(f"裁剪 img[{y}:{y + ch}, {x}:{x + cw}]；改色 img[{by}:{by + bh}, {bx}:{bx + bw}] -> results/")

    b, g, r = (channel(img, i) for i in range(3))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for name, im in (("b", b), ("g", g), ("r", r)):
        imwrite(RESULTS / f"channel_{name}.jpg", im)
    imwrite(RESULTS / "gray.jpg", gray)
    print(f"通道图 shape = {b.shape}；灰度图 shape = {gray.shape}，少一个通道维")

    panels = [("Original", img), ("Red block (copy)", recolored), ("B only", b),
              ("G only", g), ("R only", r), ("Gray", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))]
    tiles = [tile(p, t) for t, p in panels]
    grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    imwrite(RESULTS / "channel_compare.jpg", grid)
    print("2x3 对比图 -> results/channel_compare.jpg")


if __name__ == "__main__":
    main()
