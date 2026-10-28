"""Plan PDF: page 1 = notes/questions, then one row per shot (in / mid / out frames, applied zoom)."""
import json, os, sys, subprocess, io
from PIL import Image, ImageDraw, ImageFont

S, OUT, VER = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(os.path.join(OUT, f"kurgu_{VER}.json")))
F = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
f_t, f_h, f_b, f_s = (ImageFont.truetype(FB, 40), ImageFont.truetype(FB, 26),
                      ImageFont.truetype(F, 22), ImageFont.truetype(F, 18))
W, H = 1654, 2339  # A4 @200dpi
TW, TH = 400, 225
P = "WhatsApp Video 2026-10-22 at "

def short(k):
    return k.replace(P, "WA ").replace("DJI_20261022", "DRONE ").replace("_D", "")

def fname(k):
    return ("dron/" + k + ".MP4") if k.startswith("DJI") else (k + ".mp4")

def grab(k, t, zoom):
    zoom = zoom[0] if isinstance(zoom, list) else zoom
    f = os.path.join(S, "dl/cem", fname(k))
    vf = f"scale={TW}:{TH}:force_original_aspect_ratio=increase,crop={TW}:{TH}"
    if zoom != 1.0:
        vf = f"scale={int(TW * zoom)}:{int(TH * zoom)}:force_original_aspect_ratio=increase,crop={TW}:{TH}"
    o = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0, t):.3f}", "-i", f, "-frames:v", "1",
                        "-vf", vf, "-f", "image2pipe", "-c:v", "png", "-"], capture_output=True).stdout
    return Image.open(io.BytesIO(o)).convert("RGB")

def tc(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"

pages = []
pg = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(pg)
y = 90
dr.text((90, y), f"29 Ekim — Kurgu Planı {VER}", font=f_t, fill="black"); y += 80
lines = [
    "Müzik: WhatsApp Audio 2026-10-26 at 22.35.04.mp3 (3:05.7, ~112 BPM)",
    "Kesimler 4 vuruşluk ölçü ızgarasında (1 ölçü ≈ 2.14 sn; çoğu plan 2 ölçü ≈ 4.3 sn).",
    "Tüm planlar müziğe senkron: kaynaktaki an = şarkıdaki an (dudak senkronu korunur).",
    f"Toplam {len(d['shots'])} plan. v2 brifi: omurga 12.38.08 (yakın) + 22.47.08 (geniş) ≈ %45,",
    "   drone ≈ %23 (yavaş yakınlaşma), diğer çekimler ≈ %32. Titreme/sarsıntı ve kadraja giren telefon elendi.",
    "",
    "Senkron (şarkıdaki başlangıç sn):",
]
for k, v in sorted(d["offsets"].items(), key=lambda x: x[1]):
    lines.append(f"   {short(k):28s} {v:7.2f}")
lines += [
    "",
    "Senkronsuz / kullanılmayan: WA 13.43.23 (bayraklı öğretmenler), WA 22.38.01 (1) (senkron belirsiz),",
    "   WA 13.01.01 ve WA 22.38.01 (çok sarsıntılı), WA 22.38.01 (2) (dikey + sarsıntılı),",
    "   WA 14.07.54 (1) (WA 14.07.54 ile birebir aynı), DRONE 0001 yalnız açılışta.",
    "",
    "SORULAR:",
    "1) “Çocukların 29 Ekim hakkında konuştuğu” klip bulamadım — hepsi şarkı/dans. 13.43.23 (bayraklar) eklensin mi?",
    "2) Hangi planı değiştirelim? Plan numarasıyla yazman yeterli (ör. “12 → 12.38.08”).",
]
for ln in lines:
    dr.text((90, y), ln, font=f_b, fill="black"); y += 36
pages.append(pg)

per = 7
rows = d["shots"]
for p0 in range(0, len(rows), per):
    pg = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(pg)
    y = 70
    for n, s in enumerate(rows[p0:p0 + per], p0 + 1):
        dur = s["song_out"] - s["song_in"]
        ts = [s["src_in"] + 0.1, s["src_in"] + dur / 2, s["src_in"] + dur - 0.15]
        dr.text((70, y), f"{n:02d}", font=f_t, fill="black")
        info = f"{tc(s['song_in'])} → {tc(s['song_out'])}   ({dur:.2f} sn)   {short(s['src'])}   kaynak {tc(s['src_in'])}"
        z = s["zoom"] if isinstance(s["zoom"], list) else [s["zoom"]] * 2
        if z != [1.0, 1.0]:
            info += f"   zoom {z[0]:.2f}→{z[1]:.2f}"
        if s.get("note"):
            info += "   " + s["note"]
        dr.text((170, y + 8), info, font=f_s, fill="black")
        for i, t in enumerate(ts):
            pg.paste(grab(s["src"], t, s["zoom"]), (170 + i * (TW + 20), y + 45))
        y += TH + 85
    pages.append(pg)
pages[0].save(os.path.join(OUT, f"29Ekim_Kurgu_Plani_{VER}.pdf"), save_all=True, append_images=pages[1:], resolution=200)
print("pages", len(pages))
