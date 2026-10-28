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
    P + "14.07.54": 5.572, P + "22.38.01 (1)": 0.302,
    "DJI_20261022115523_0002_D": 13.95, "DJI_20261022115924_0003_D": 7.16,
}
UNSYNCED = [P + "13.43.23", P + "13.01.01", P + "22.38.01", P + "22.38.01 (2)",
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

def window_quality(k, a, b):
    """Quality of song window [a,b] in source k; None if not covered."""
    off = OFFSET[k]
    s0, s1 = a - off, b - off
    if s0 < 0.15 or s1 > meta[k]["dur"] - 0.3:
        return None
    r = met[k]
    m = (r[:, 0] >= s0) & (r[:, 0] <= s1)
    if m.sum() < 3:
        return None
    sh = r[m, 1]
    sharp = r[m, 2] / np.median(r[:, 2])
    q = -2.0 * np.percentile(sh, 90) - 4.0 * (sh > 1.5).mean() + 0.5 * min(sharp.mean(), 1.3)
    if np.percentile(sh, 90) > 2.0:
        return None  # whip / rotation / heavy shake
    return q

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

# shot lengths in bars: mostly 2 bars (8 beats), 1 bar where energy is high
rms = librosa.feature.rms(y=y, hop_length=512)[0]
rt = librosa.times_like(rms, sr=sr, hop_length=512)
cuts = [0.0, music_in]
i = int(np.searchsorted(bars, music_in))
while True:
    seg_rms = rms[(rt >= bars[i]) & (rt < bars[min(i + 2, len(bars) - 1)])].mean()
    n = 1 if seg_rms > np.percentile(rms, 70) and len(cuts) % 3 else 2
    if i + n >= len(bars) - 1:
        break
    i += n
    cuts.append(float(bars[i]))
cuts.append(song_len)
# final shot: drone from ~12 s before end
while song_len - cuts[-2] < 8 and len(cuts) > 3:
    cuts.pop(-2)

# --- choose sources
shots = []
use = {k: 0.0 for k in OFFSET}
hist = []
for si in range(len(cuts) - 1):
    a, b = cuts[si], cuts[si + 1]
    if si == 0:
        shots.append(dict(src="DJI_20261022115239_0001_D", song_in=a, song_out=b, src_in=0.5,
                          zoom=1.0, note="acilis drone (senkronsuz, muzik oncesi)"))
        hist.append("DJI_20261022115239_0001_D")
        continue
    last = si == len(cuts) - 2
    total = sum(use.values()) + 1e-6
    best = None
    for k in OFFSET:
        q = window_quality(k, a, b)
        if q is None:
            continue
        if k in hist[-2:]:
            continue
        cls_share = (sum(use[x] for x in MAIN) if k in MAIN else
                     sum(use[x] for x in DRONES) if k in DRONES else
                     sum(use[x] for x in OTHERS)) / total
        target = 0.60 if k in MAIN else 0.15 if k in DRONES else 0.25
        score = q + 6.0 * (target - cls_share)
        if k in MAIN:
            score += 1.0 + 10.0 * (0.15 - use[k] / total)
        if k in DRONES and hist and hist[-1] in DRONES:
            score -= 5
        if hist and (k in WIDE) == (hist[-1] in WIDE) and not k.startswith("DJI"):
            score -= 0.6  # alternate wide / close
        if last:
            score += 20 if k in DRONES else 0
        if best is None or score > best[0]:
            best = (score, k)
    k = best[1]
    zoom = 1.0
    if k in DRONES:
        zoom = [1.35, 1.6, 1.45, 1.75][len([h for h in hist if h in DRONES]) % 4]
    if last:
        zoom = 1.0
    shots.append(dict(src=k, song_in=a, song_out=b, src_in=a - OFFSET[k], zoom=zoom,
                      q=round(best[0], 2)))
    use[k] += b - a
    hist.append(k)

os.makedirs(OUT, exist_ok=True)
json.dump(dict(cuts=cuts, shots=shots, offsets=OFFSET, bars=[float(x) for x in bars]),
          open(os.path.join(OUT, "kurgu_v1.json"), "w"), indent=1)

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

def transform(z):
    return f'<adjust-transform scale="{z:.2f} {z:.2f}"/>' if z != 1.0 else ""

# KURGU v1: gap on spine holding the song; video shots connected on lane 1
def kurgu_body(ids):
    inner = [f'<asset-clip ref="{ids[SONG]}" lane="-1" offset="0s" name="{esc(SONG)}" start="0s" duration="{fr(song_len)}"/>']
    for n, s in enumerate(shots, 1):
        a, b = round(s["song_in"] * FPS), round(s["song_out"] * FPS)
        src_in = max(0, round(s["src_in"] * FPS))
        inner.append(f'<asset-clip ref="{ids[s["src"]]}" lane="1" offset="{a}/{FPS}s" name="{n:02d} {esc(s["src"])}" '
                     f'start="{src_in}/{FPS}s" duration="{b - a}/{FPS}s" srcEnable="video">{transform(s["zoom"])}</asset-clip>')
    return f'<gap name="KURGU" offset="0s" start="0s" duration="{fr(song_len)}">\n' + "\n".join(inner) + "\n</gap>"

keys = [SONG] + list(OFFSET) + UNSYNCED
open(os.path.join(OUT, "29Ekim_KURGU_v1.fcpxml"), "w").write(fcpxml("29 Ekim KURGU v1", kurgu_body, keys, song_len))

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
open(os.path.join(OUT, "29Ekim_SENKRON.fcpxml"), "w").write(fcpxml("29 Ekim SENKRON", lambda ids: sb, keys, stotal))
print("written")
