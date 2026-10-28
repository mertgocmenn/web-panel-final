"""29 Ekim: beat-locked edit list from synced sources + FCPXML timelines for Resolve."""
import json, os, sys
import numpy as np
import librosa
from fractions import Fraction

S = sys.argv[1]
OUT = sys.argv[2]
FPS = 30
SONG = "WhatsApp Audio 2026-10-26 at 22.35.04.mp3"
P = "WhatsApp Video 2026-10-22 at "

# song time of source t=0 (audio xcorr, refined; drones via choreography motion match)
OFFSET = {
    P + "14.06.22": 0.725, P + "22.47.08": 8.763, P + "12.38.08": 7.583, P + "12.40.57": 3.288,
    P + "12.58.15 (1)": 17.685, P + "16.20.55": 21.858, P + "17.36.26": 3.928,
    P + "12.58.09": 147.130, P + "12.58.15": 30.651, P + "12.58.51": 124.755,
    P + "13.01.01 (1)": 8.860, P + "13.32.04": 10.330, P + "13.32.05": 72.578,
    P + "13.32.05 (1)": 103.020, P + "14.07.17": 126.403, P + "14.07.53": 165.045,
    P + "14.07.54": 5.572,
    "DJI_20261022115523_0002_D": 13.95, "DJI_20261022115924_0003_D": 7.16,
}
UNSYNCED = [P + "22.38.01 (1)", P + "13.43.23", P + "13.01.01", P + "22.38.01", P + "22.38.01 (2)",
            P + "14.07.54 (1)", "DJI_20261022115239_0001_D"]
MAIN = [P + "14.06.22", P + "22.47.08", P + "12.38.08", P + "12.40.57"]
WIDE = {P + "14.06.22", P + "22.47.08", P + "12.40.57", P + "16.20.55"}
DRONES = ["DJI_20261022115523_0002_D", "DJI_20261022115924_0003_D"]
OTHERS = [k for k in OFFSET if k not in MAIN and k not in DRONES]

def fname(k):
    return ("dron/" + k + ".MP4") if k.startswith("DJI") else (k + ".mp4")

meta, met = {}, {}
import subprocess
for k in list(OFFSET) + UNSYNCED:
    f = os.path.join(S, "dl/cem", fname(k))
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height:format=duration", "-of", "json", f],
                       capture_output=True, text=True).stdout
    j = json.loads(o)
    meta[k] = dict(w=j["streams"][0]["width"], h=j["streams"][0]["height"],
                   dur=float(j["format"]["duration"]), audio=not k.startswith("DJI"))
    met[k] = np.array(json.load(open(os.path.join(S, "met", k + ".json")))["rows"])

# source ranges with framing problems (phone / hand in frame), seconds in source time
BLOCK = {P + "12.58.15 (1)": [(105.0, 150.0)]}

def window_quality(k, a, b):
    """Quality of song window [a,b] in source k; None if not covered."""
    off = OFFSET[k]
    s0, s1 = a - off, b - off
    if s0 < 0.15 or s1 > meta[k]["dur"] - 0.3:
        return None
    for b0, b1 in BLOCK.get(k, []):
        if s0 < b1 and s1 > b0:
            return None
    r = met[k]
    m = (r[:, 0] >= s0) & (r[:, 0] <= s1)
    if m.sum() < 3:
        return None
    sh = r[m, 1]
    jit = JIT[k][m]
    sharp = r[m, 2] / np.median(r[:, 2])
    # smooth camera travel is fine; penalise jitter (shake) and whip pans
    q = (-4.0 * np.percentile(jit, 90) - 0.4 * np.percentile(sh, 90) - 4.0 * (sh > 2.0).mean()
         + 0.5 * min(sharp.mean(), 1.3))
    if np.percentile(sh, 95) > 3.0 or np.percentile(jit, 90) > 0.6:
        return None  # whip / rotation / heavy shake
    return q

JIT = {}
for _k, _r in met.items():
    _w = np.ones(7) / 7
    JIT[_k] = np.hypot(_r[:, 4] - np.convolve(_r[:, 4], _w, "same"), _r[:, 5] - np.convolve(_r[:, 5], _w, "same"))
    JIT[_k][:3] = JIT[_k][-3:] = 0.2

# --- beat grid
y, sr = librosa.load(os.path.join(S, "wav", SONG[:-4] + ".wav"), sr=22050)
_, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
on = librosa.onset.onset_strength(y=y, sr=sr)
ot = librosa.times_like(on, sr=sr)
bstr = np.interp(beats, ot, on)
phase = int(np.argmax([bstr[p::4].mean() for p in range(4)]))
bars = beats[phase::4]
song_len = len(y) / sr
music_in = float(bars[bars > 2.0][0])

FAV = [P + "12.38.08", P + "22.47.08"]
TARGET = {P + "12.38.08": 0.24, P + "22.47.08": 0.22, P + "12.40.57": 0.08, P + "14.06.22": 0.07,
          DRONES[0]: 0.065, DRONES[1]: 0.065}
OTHER_T = 0.21

def wq(k, a, b):
    q = window_quality(k, a, b)
    if q is None:
        return None
    r = met[k]; off = OFFSET[k]
    m = (r[:, 0] >= a - off) & (r[:, 0] <= b - off)
    rel = r[m, 2] / np.median(r[:, 2])
    if rel.min() < 0.45:
        q -= 1.5  # occlusion / blur somewhere in the window
    return q

# per-source baseline: pick each source's best moments, shares decide how much of each
QMED = {}
for _k in OFFSET:
    _q = [wq(_k, bars[_i], bars[_i + 2]) for _i in range(len(bars) - 2)]
    _q = [x for x in _q if x is not None]
    QMED[_k] = float(np.median(_q)) if _q else 0.0

shots = [dict(src="DJI_20261022115239_0001_D", song_in=0.0, song_out=music_in, src_in=0.5,
              zoom=(1.0, 1.12), note="acilis drone (muzik oncesi)")]
use = {k: 0.0 for k in OFFSET}
hist = ["DJI_20261022115239_0001_D"]
i = int(np.searchsorted(bars, music_in))
final_from = song_len - 11.0
drone_n = 0
while True:
    a = float(bars[i])
    if a >= final_from - 1.0 or i >= len(bars) - 2:
        break
    total = sum(use.values()) + 1e-6
    since_drone = next((n for n, h in enumerate(reversed(hist)) if h in DRONES), 99)
    best = None
    for k in OFFSET:
        if k == hist[-1] or (k in DRONES and hist[-1] in DRONES):
            continue
        for n in (1, 2, 3):
            if i + n >= len(bars):
                continue
            b = float(bars[i + n])
            if b > final_from + 0.5:
                continue
            q = wq(k, a, b)
            if q is None:
                continue
            score = 1.5 * (q - QMED[k]) + 0.3 * q + {1: 0.0, 2: 0.45, 3: 0.1}[n]
            if k in DRONES:
                score += 0.4 if n >= 2 else -2
                score += 3.0 if since_drone >= 7 else (-3.0 if since_drone < 4 else 0)
            if k in TARGET:
                score += 9.0 * (TARGET[k] - use[k] / total)
            else:
                oth = sum(use[x] for x in OTHERS if x not in TARGET) / total
                score += 9.0 * (OTHER_T - oth) - 4.0 * use[k] / total
            if k in FAV:
                score += 0.8
            if (k in WIDE) == (hist[-1] in WIDE) and not k.startswith("DJI") and not hist[-1].startswith("DJI"):
                score -= 0.8
            if best is None or score > best[0]:
                best = (score, k, n, b)
    _, k, n, b = best
    zoom = (1.0, 1.0)
    if k in DRONES:
        zoom = [(1.3, 1.5), (1.5, 1.75), (1.2, 1.45), (1.45, 1.7)][drone_n % 4]
        drone_n += 1
    shots.append(dict(src=k, song_in=a, song_out=b, src_in=a - OFFSET[k], zoom=zoom, q=round(best[0], 2)))
    use[k] += b - a
    hist.append(k)
    i += n
a = shots[-1]["song_out"]
k = DRONES[1]
shots.append(dict(src=k, song_in=a, song_out=song_len, src_in=a - OFFSET[k], zoom=(1.0, 1.1), note="final drone"))
use[k] += song_len - a
cuts = [s["song_in"] for s in shots] + [song_len]
os.makedirs(OUT, exist_ok=True)
json.dump(dict(cuts=cuts, shots=shots, offsets=OFFSET, bars=[float(x) for x in bars]),
          open(os.path.join(OUT, "kurgu_v2.json"), "w"), indent=1)

share = {}
for s in shots:
    share[s["src"]] = share.get(s["src"], 0) + s["song_out"] - s["song_in"]
for k, v in sorted(share.items(), key=lambda x: -x[1]):
    print(f"{k:45s} {v:6.1f}s {100 * v / song_len:5.1f}%")
print("shots", len(shots), "music_in", round(music_in, 2))

# ---------------- FCPXML
def fr(t):
    """seconds -> frame-quantised rational string"""
    n = int(round(t * FPS))
    return f"{n}/{FPS}s" if n else "0s"

def esc(s):
    return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")

MEDIA_ROOT = "file:///29_EKIM_MEDYA/"
def url(k):
    from urllib.parse import quote
    return MEDIA_ROOT + quote(fname(k))

def resources(keys):
    out = ['<format id="r0" name="FFVideoFormat1080p30" frameDuration="1/30s" width="1920" height="1080"/>']
    ids = {}
    for n, k in enumerate(keys, 1):
        rid = f"r{n}"
        ids[k] = rid
        m = meta[k] if k in meta else None
        if k == SONG:
            out.append(f'<asset id="{rid}" name="{esc(k)}" start="0s" duration="{fr(song_len)}" hasVideo="0" hasAudio="1" audioSources="1" audioChannels="2" audioRate="44100">'
                       f'<media-rep kind="original-media" src="{MEDIA_ROOT}mp3/{esc(SONG).replace(" ", "%20")}"/></asset>')
        else:
            out.append(f'<asset id="{rid}" name="{esc(k)}" start="0s" duration="{fr(m["dur"])}" hasVideo="1" format="r0" hasAudio="{1 if m["audio"] else 0}"'
                       f'{" audioSources=\"1\" audioChannels=\"2\" audioRate=\"48000\"" if m["audio"] else ""}>'
                       f'<media-rep kind="original-media" src="{url(k)}"/></asset>')
    return "\n".join(out), ids

def fcpxml(name, body, keys, total):
    res, ids = resources(keys)
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.9">
<resources>
{res}
</resources>
<library>
<event name="29 Ekim">
<project name="{name}">
<sequence format="r0" duration="{fr(total)}" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
<spine>
{body(ids)}
</spine>
</sequence>
</project>
</event>
</library>
</fcpxml>
'''

def transform(z, st=0, du=0):
    z0, z1 = z
    if z0 == 1.0 and z1 == 1.0:
        return ""
    return (f'<adjust-transform scale="{z0:.2f} {z0:.2f}"><param name="scale"><keyframeAnimation>'
            f'<keyframe time="{st}/{FPS}s" value="{z0:.2f} {z0:.2f}"/>'
            f'<keyframe time="{st + du}/{FPS}s" value="{z1:.2f} {z1:.2f}"/>'
            f'</keyframeAnimation></param></adjust-transform>')

# KURGU v1: gap on spine holding the song; video shots connected on lane 1
def kurgu_body(ids):
    inner = [f'<asset-clip ref="{ids[SONG]}" lane="-1" offset="0s" name="{esc(SONG)}" start="0s" duration="{fr(song_len)}"/>']
    for n, s in enumerate(shots, 1):
        a, b = round(s["song_in"] * FPS), round(s["song_out"] * FPS)
        src_in = max(0, round(s["src_in"] * FPS))
        inner.append(f'<asset-clip ref="{ids[s["src"]]}" lane="1" offset="{a}/{FPS}s" name="{n:02d} {esc(s["src"])}" '
                     f'start="{src_in}/{FPS}s" duration="{b - a}/{FPS}s" srcEnable="video">{transform(s["zoom"], src_in, b - a)}</asset-clip>')
    return f'<gap name="KURGU" offset="0s" start="0s" duration="{fr(song_len)}">\n' + "\n".join(inner) + "\n</gap>"

keys = [SONG] + list(OFFSET) + UNSYNCED
open(os.path.join(OUT, "29Ekim_KURGU_v2.fcpxml"), "w").write(fcpxml("29 Ekim KURGU v2", kurgu_body, keys, song_len))

# SENKRON: every synced source stacked at its song position (clip audio kept, muted), unsynced after the song
def senkron_body(ids):
    inner = [f'<asset-clip ref="{ids[SONG]}" lane="-1" offset="0s" name="{esc(SONG)}" start="0s" duration="{fr(song_len)}"/>']
    order = MAIN + DRONES + OTHERS
    for lane, k in enumerate(order, 1):
        off = OFFSET[k]
        start = max(0.0, -off)
        a = round(max(0.0, off) * FPS)
        dur = round((meta[k]["dur"] - start) * FPS) - 1
        vol = '<adjust-volume amount="-96dB"/>' if meta[k]["audio"] else ""
        inner.append(f'<asset-clip ref="{ids[k]}" lane="{lane}" offset="{a}/{FPS}s" name="{esc(k)}" '
                     f'start="{round(start * FPS)}/{FPS}s" duration="{dur}/{FPS}s">{vol}</asset-clip>')
    t = song_len + 5
    for k in UNSYNCED:
        d = round(meta[k]["dur"] * FPS) - 1
        inner.append(f'<asset-clip ref="{ids[k]}" lane="1" offset="{round(t * FPS)}/{FPS}s" name="{esc(k)} (SENKRONSUZ)" start="0s" duration="{d}/{FPS}s"/>')
        t += d / FPS + 1
    total = t
    return f'<gap name="SENKRON" offset="0s" start="0s" duration="{fr(total)}">\n' + "\n".join(inner) + "\n</gap>", total

_res, _ids = resources(keys)
sb, stotal = senkron_body(_ids)
(lambda *x: None)(fcpxml("29 Ekim SENKRON", lambda ids: sb, keys, stotal))
print("written")
