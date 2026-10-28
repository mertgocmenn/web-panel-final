"""Low-res review render of an edit json: 1280x720/30, frame-exact segments + song."""
import json, os, sys, subprocess
from concurrent.futures import ThreadPoolExecutor

S, EDIT, OUTMP4 = sys.argv[1], sys.argv[2], sys.argv[3]
FPS, W, H = 30, 1280, 720
d = json.load(open(EDIT))
SONG = os.path.join(S, "dl/cem/mp3/WhatsApp Audio 2026-10-26 at 22.35.04.mp3")
tmp = os.path.join(S, "render_tmp"); os.makedirs(tmp, exist_ok=True)

def fname(k):
    return os.path.join(S, "dl/cem", ("dron/" + k + ".MP4") if k.startswith("DJI") else (k + ".mp4"))

def seg(n_s):
    n, s = n_s
    nf = round(s["song_out"] * FPS) - round(s["song_in"] * FPS)
    z0, z1 = s["zoom"] if isinstance(s["zoom"], list) else (s["zoom"], s["zoom"])
    fill = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1"
    if z0 == 1.0 and z1 == 1.0:
        vf = f"fps={FPS},{fill}"
    else:
        vf = (f"fps={FPS},scale={2 * W}:{2 * H}:force_original_aspect_ratio=increase,crop={2 * W}:{2 * H},"
              f"zoompan=z='{z0}+({z1}-{z0})*on/{max(nf - 1, 1)}':d=1:x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':s={W}x{H}:fps={FPS},setsar=1")
    out = os.path.join(tmp, f"{n:03d}.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0, s['src_in']):.3f}", "-i", fname(s["src"]),
                    "-an", "-vf", vf, "-frames:v", str(nf), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p", out], check=True)
    return out

with ThreadPoolExecutor(4) as ex:
    parts = list(ex.map(seg, enumerate(d["shots"], 1)))
lst = os.path.join(tmp, "list.txt")
open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", SONG,
                "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "24",
                "-vf", "fade=t=in:st=0:d=0.8,fade=t=out:st=183.7:d=2", "-af", "afade=t=out:st=183.7:d=2",
                "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", OUTMP4], check=True)
print("ok", OUTMP4)
