import sys, os, json
import numpy as np, librosa
from scipy.signal import correlate

W = sys.argv[1]
SR = 22050
sync = json.load(open(os.path.join(W, "..", "sync.json")))
ref, _ = librosa.load(os.path.join(W, "WhatsApp Audio 2026-10-26 at 22.35.04.wav"), sr=SR)

def env(y):
    # onset envelope at 1 ms resolution for fine alignment
    o = librosa.onset.onset_strength(y=y, sr=SR, hop_length=64)
    return (o - o.mean()) / (o.std() + 1e-6)

res = {}
for name in sys.argv[2:]:
    y, _ = librosa.load(os.path.join(W, name + ".wav"), sr=SR)
    off = sync[name]["offset"]
    rows = []
    for t in np.arange(2, len(y) / SR - 12, 15):
        a = y[int(t * SR):int((t + 10) * SR)]
        r0 = t + off - 1.0
        if r0 < 0 or r0 + 12 > len(ref) / SR:
            continue
        b = ref[int(r0 * SR):int((r0 + 12) * SR)]
        ea, eb = env(a), env(b)
        c = correlate(eb, ea, mode="valid")
        k = int(np.argmax(c))
        fine = r0 + k * 64 / SR - t  # ref_time - clip_time
        rows.append((round(float(t), 1), round(fine, 3), round(float(c[k] / len(ea)), 2)))
    res[name] = rows
    print(name, rows, flush=True)
json.dump(res, open(os.path.join(W, "..", "drift.json"), "w"))
