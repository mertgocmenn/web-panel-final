import sys, glob, os, json
import numpy as np, librosa
from scipy.signal import correlate

W = sys.argv[1]
SR, HOP = 22050, 256
ref_path = os.path.join(W, "WhatsApp Audio 2026-10-26 at 22.35.04.wav")

def feats(p):
    y, _ = librosa.load(p, sr=SR)
    ch = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
    ch = (ch - ch.mean(1, keepdims=True)) / (ch.std(1, keepdims=True) + 1e-6)
    on = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    on = (on - on.mean()) / (on.std() + 1e-6)
    return y, ch, on

ry, rch, ron = feats(ref_path)
out = {}
for p in sorted(glob.glob(os.path.join(W, "WhatsApp Video*.wav"))):
    y, ch, on = feats(p)
    n = ch.shape[1]
    # lag L: clip frame 0 aligns with ref frame L (L may be negative)
    c = sum(correlate(rch[k], ch[k], mode="full") for k in range(12)) / 12
    c += correlate(ron, on, mode="full")
    lags = np.arange(-n + 1, rch.shape[1])
    # normalise by overlap
    ov = np.minimum(lags + n, rch.shape[1]) - np.maximum(lags, 0)
    ov = np.clip(ov, 1, None)
    score = c / ov
    score[ov < min(n, rch.shape[1]) * 0.5] = -9
    i = int(np.argmax(score))
    s = np.sort(score)[::-1]
    # second peak away from first
    far = score.copy(); far[max(0, i - 40):i + 40] = -9
    off = lags[i] * HOP / SR
    out[os.path.basename(p)[:-4]] = dict(offset=round(off, 3), score=round(float(score[i]), 3),
                                        second=round(float(far.max()), 3), dur=round(len(y) / SR, 2))
    print(f"{os.path.basename(p)[:-4]:45s} off={off:8.3f}s score={score[i]:.3f} 2nd={far.max():.3f} dur={len(y)/SR:.1f}", flush=True)
json.dump(out, open(os.path.join(W, "..", "sync.json"), "w"), indent=1)
