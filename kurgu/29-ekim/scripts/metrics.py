"""Per-clip metrics sampled at 10 fps: camera shake (global translation), sharpness, motion energy."""
import sys, os, json
import numpy as np, cv2

path, out = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
step = 1 / 10
t_next, prev, rows = 0.0, None, []
win = None
while True:
    ok = cap.grab()
    if not ok:
        break
    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    if t + 1e-6 < t_next:
        continue
    t_next += step
    _, f = cap.retrieve()
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    small = cv2.resize(g, (192, int(192 * h / w))).astype(np.float32)
    sharp = float(cv2.Laplacian(cv2.resize(g, (480, int(480 * h / w))), cv2.CV_64F).var())
    if win is None:
        win = cv2.createHanningWindow(small.shape[::-1], cv2.CV_32F)
    if prev is not None:
        (dx, dy), resp = cv2.phaseCorrelate(prev, small, win)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        comp = cv2.warpAffine(prev, M, small.shape[::-1])
        energy = float(np.abs(small - comp)[8:-8, 8:-8].mean())
        rows.append([round(t, 2), round(float(np.hypot(dx, dy)) / 192 * 100, 3), round(sharp, 1), round(energy, 3)])
    prev = small
json.dump(dict(fps=fps, rows=rows), open(out, "w"))
print(os.path.basename(path), len(rows))
