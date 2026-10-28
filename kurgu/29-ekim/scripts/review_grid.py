"""Review grids: 5 frames per shot from a rendered preview, 15 shots per image."""
import sys, json, subprocess, os

S, EDIT, VIDEO, OUTDIR = sys.argv[1:5]
os.makedirs(OUTDIR, exist_ok=True)
d = json.load(open(EDIT))
rows = []
for n, s in enumerate(d["shots"], 1):
    a, b = s["song_in"], s["song_out"]
    L = b - a
    frames = []
    for i in range(5):
        t = a + 0.15 + i * (L - 0.3) / 4
        o = os.path.join(OUTDIR, f"s{n:02d}_{i}.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", VIDEO, "-frames:v", "1",
                        "-vf", "scale=256:144", o], check=True)
        frames.append(o)
    lab = s["src"].replace("WhatsApp Video 2026-10-22 at ", "").replace("DJI_20261022", "DR")
    row = os.path.join(OUTDIR, f"row{n:02d}.png")
    args = sum([["-i", f] for f in frames], [])
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args, "-filter_complex",
                    f"hstack=5,pad=iw+170:ih:170:0,drawtext=text='{n:02d} {lab}':x=4:y=60:fontsize=15:fontcolor=white",
                    row], check=True)
    rows.append(row)
for g in range(0, len(rows), 15):
    part = rows[g:g + 15]
    args = sum([["-i", f] for f in part], [])
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args, "-filter_complex", f"vstack={len(part)}",
                    os.path.join(OUTDIR, f"grid{g // 15 + 1}.jpg")], check=True)
print("grids", (len(rows) + 14) // 15)
