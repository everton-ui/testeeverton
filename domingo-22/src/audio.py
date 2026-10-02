# Trilha de forró (baião) + mixagem com a voz + dados de sincronia para a animação.
import json
import numpy as np, soundfile as sf
from scipy.signal import resample_poly, butter, sosfilt

SR = 48000
DUR = 10.0
FPS = 30
L0, L1 = 0.80, 3.50              # início de cada fala
CUE_SEXTA = L0 + 0.58            # "sexta-feira"
CUE_OUTUBRO = L0 + 1.18          # "dois de outubro"
SLAM = L1 + 0.76                 # "vinte e dois" -> 22 na tela
BEAT = 0.5                       # 120 bpm, grade ancorada no SLAM
STEP = BEAT / 4                  # semicolcheia
rng = np.random.default_rng(22)
N = int(SR * DUR)
t = np.arange(N) / SR
VOICE_END = L1 + sf.info('line1.wav').duration

def bp(x, lo, hi, order=2): return sosfilt(butter(order, [lo, hi], "bandpass", fs=SR, output="sos"), x)
def hp(x, f): return sosfilt(butter(2, f, "highpass", fs=SR, output="sos"), x)
def lp(x, f): return sosfilt(butter(2, f, "lowpass", fs=SR, output="sos"), x)
def add(buf, sig, at, gain=1.0):
    i = int(round(at * SR))
    if i >= len(buf): return
    sig = sig[: len(buf) - i]
    buf[i:i + len(sig)] += gain * sig
def hz(m): return 440 * 2 ** ((m - 69) / 12)

# ---------- voz ----------
voice = np.zeros(N)
for f, at in (("line0.wav", L0), ("line1.wav", L1)):
    v, sr = sf.read(f)
    if v.ndim > 1: v = v.mean(1)
    v = resample_poly(v, SR, sr)
    add(voice, v / np.abs(v).max(), at, 0.92)

# ---------- sanfona: síntese aditiva de palhetas, afinação "musette" (batimento) ----------
def reed(freq, d, cents=(0, 14, -11), swell=0.0, attack=0.025, release=0.06):
    n = int(SR * d)
    tt = np.arange(n) / SR
    out = np.zeros(n)
    for c in cents:
        f = freq * 2 ** (c / 1200)
        ph = rng.random() * 6.28
        for h in range(1, int(11000 / f) + 1):
            fh = f * h
            amp = h ** -0.75 * (1 + 1.2 * np.exp(-((fh - 1600) / 900) ** 2))   # corpo/caixa da sanfona
            out += amp * np.sin(2 * np.pi * fh * tt + ph * h)
    out /= len(cents) * 6
    env = np.minimum(tt / attack, 1) * np.clip((d - tt) / release, 0, 1)
    env *= 1 + swell * tt / max(d, 1e-3)          # fole abrindo
    return out * env

def chord(midis, d, gain=1.0, **kw):
    return gain * sum(reed(hz(m), d, **kw) for m in midis) / len(midis) ** 0.6

# ---------- zabumba e triângulo ----------
def grave(strength=1.0, muted=False):
    d = 0.18 if muted else 0.45
    tt = np.arange(int(SR * d)) / SR
    f = 68 + 45 * np.exp(-tt * 28)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * (22 if muted else 7))
    s += 0.35 * lp(rng.standard_normal(len(tt)), 1400) * np.exp(-tt * 70)
    return strength * s

def bacalhau(strength=1.0):
    tt = np.arange(int(SR * 0.06)) / SR
    s = bp(rng.standard_normal(len(tt)), 1800, 6500) * np.exp(-tt * 90)
    s += 0.5 * np.sin(2 * np.pi * 1250 * tt) * np.exp(-tt * 80)
    return strength * s

TRI = [(2950, 1.0), (4380, 0.7), (5820, 0.55), (7410, 0.4), (8870, 0.3), (10300, 0.2)]
def triangulo(open_=False, strength=1.0):
    d = 0.7 if open_ else 0.07
    tt = np.arange(int(SR * d)) / SR
    s = sum(a * np.sin(2 * np.pi * f * tt + rng.random() * 6.28) for f, a in TRI)
    s *= np.exp(-tt * (4.5 if open_ else 55))
    s += 0.3 * hp(rng.standard_normal(len(tt)), 5000) * np.exp(-tt * 300)
    return strength * s / 2.5

def boom():
    tt = np.arange(int(SR * 2.5)) / SR
    f = 38 + 120 * np.exp(-tt * 9)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 1.6)
    return s + hp(rng.standard_normal(len(tt)), 3500) * np.exp(-tt * 1.4) * 0.3

def applause(d):
    n = int(SR * d); s = np.zeros(n); cl = int(SR * 0.012)
    env = np.exp(-np.arange(cl) / SR * 400)
    for _ in range(int(d * 900)):
        i = rng.integers(0, n - cl)
        s[i:i + cl] += rng.standard_normal(cl) * env * rng.uniform(0.3, 1)
    s = bp(s, 900, 7000); tt = np.arange(n) / SR
    return s / np.abs(s).max() * np.minimum(tt / 0.3, 1) * np.clip((d - tt) / 1.5, 0, 1)

# acordes (mão direita, região média) e baixos (mão esquerda)
CH = {"D": [62, 66, 69], "Bm": [59, 62, 66], "G": [59, 62, 67], "A": [61, 64, 69], "A7": [61, 64, 67]}
BASS = {"D": (38, 45), "Bm": (35, 42), "G": (43, 38), "A": (45, 40), "A7": (45, 40)}

music = np.zeros(N)

# ---------- introdução: sanfona "abrindo o fole" + zabumba como coração ----------
for name, a, b in (("D", 0.0, 1.25), ("Bm", 1.25, 2.25), ("G", 2.25, 3.25), ("A", 3.25, 3.75)):
    add(music, chord(CH[name] + [CH[name][0] - 12], b - a + 0.05, swell=0.8, attack=0.12, release=0.12), a, 0.5)
    add(music, reed(hz(BASS[name][0]), b - a + 0.05, cents=(0, 6), attack=0.1), a, 0.35)
k = 1
while SLAM - k * BEAT > 0.2:
    at = SLAM - k * BEAT
    add(music, grave(0.35 + 0.55 * (1 - at / SLAM)), at)
    k += 1
# triângulo entra baixinho e vai crescendo
i = int(np.ceil((2.25 - SLAM) / STEP))
while SLAM + i * STEP < SLAM - 0.01:
    at = SLAM + i * STEP
    add(music, triangulo(i % 4 == 2), at, 0.12 + 0.25 * (at - 2.25) / (SLAM - 2.25))
    i += 1
# subida de sanfona até o 22 (A7 -> D)
for j, m in enumerate([64, 66, 67, 69]):
    add(music, reed(hz(m), STEP + 0.01, cents=(0, 14)), SLAM - (4 - j) * STEP, 0.55)

# ---------- o "22": impacto + acorde cheio ----------
impact = np.zeros(N)
add(impact, boom(), SLAM, 1.0)
add(impact, grave(1.3), SLAM)
add(music, chord([62, 66, 69, 74], 0.3, attack=0.01), SLAM, 0.6)
add(music, triangulo(True), SLAM, 0.5)

# ---------- baião: começa logo depois do "dois!" ----------
ROLL_END = SLAM + 10 * BEAT
BARS = ["D", "G", "A7", "D"]                       # 1 compasso (2/4) = 2 tempos
MELODY = [  # (midi, duração em semicolcheias) - sabor mixolídio (dó natural)
    [(81, 2), (78, 1), (81, 1), (86, 2), (84, 1), (81, 1)],
    [(83, 2), (79, 1), (83, 1), (86, 2), (83, 1), (79, 1)],
    [(81, 1), (83, 1), (85, 1), (88, 1), (86, 1), (85, 1), (83, 1), (81, 1)],
    [(78, 2), (76, 1), (74, 1), (69, 2), (74, 2)],
]
start = int(np.ceil((VOICE_END - 0.03 - SLAM) / STEP))
start += (-start) % 8                              # alinha no início do compasso
for b in range(4):
    bar_t = SLAM + start * STEP + b * 8 * STEP
    if bar_t >= ROLL_END - 0.01: break
    name = BARS[b]
    for s16 in range(8):
        at = bar_t + s16 * STEP
        # zabumba: grave no 1 e no "a" do 1, bacalhau nos contratempos
        if s16 == 0: add(music, grave(1.0), at)
        if s16 == 3: add(music, grave(0.7), at)
        if s16 == 5: add(music, grave(0.45, muted=True), at)
        if s16 in (2, 4, 6): add(music, bacalhau(0.6 if s16 == 4 else 0.45), at)
        # triângulo: fechado/aberto
        add(music, triangulo(s16 in (2, 6), 1.0 if s16 in (2, 6) else 0.55), at, 0.42)
    # mão esquerda: baixo no 1 e no "a" do 1, acorde no 2 e no "e" do 2
    root, fifth = BASS[name]
    add(music, reed(hz(root), 3 * STEP, cents=(0, 6)), bar_t, 0.5)
    add(music, reed(hz(fifth), STEP * 1.2, cents=(0, 6)), bar_t + 3 * STEP, 0.4)
    for s16 in (4, 6):
        add(music, chord(CH[name], STEP * 1.1, attack=0.01, release=0.03), bar_t + s16 * STEP, 0.38)
    # mão direita: melodia
    pos = 0
    for m, d in MELODY[b]:
        add(music, reed(hz(m), d * STEP * 0.95, cents=(0, 14, -11), attack=0.012, release=0.03), bar_t + pos * STEP, 0.62)
        pos += d
# acorde final
add(music, chord([62, 66, 69, 74, 78], DUR - ROLL_END, attack=0.01, release=0.4, swell=-0.3), ROLL_END, 0.75)
add(music, reed(hz(38), DUR - ROLL_END, cents=(0, 6), release=0.4), ROLL_END, 0.5)
add(music, grave(1.2), ROLL_END)
add(music, triangulo(True), ROLL_END, 0.6)
add(impact, boom() * 0.5, ROLL_END)
add(music, applause(DUR - VOICE_END + 0.1), VOICE_END - 0.1, 0.18)

# ---------- mixagem: trilha abaixa sob a voz ----------
peak = np.abs(music + impact).max()
music, impact = music / peak, impact / peak
win = np.ones(N)
i, j = int((SLAM - 0.02) * SR), int((VOICE_END + 0.05) * SR)
win[i:j] = 0.45                                   # mais espaço para "vinte e dois"
win = np.convolve(win, np.ones(int(SR * 0.04)) / int(SR * 0.04), 'same')
venv_ = np.convolve(np.abs(voice), np.ones(int(SR * 0.08)) / int(SR * 0.08), "same")
duck = 1 - 0.78 * np.clip(venv_ * 8, 0, 1)
bed = 0.62 * music * duck * win + 0.62 * impact * (1 - 0.4 * (win < 1))
mix = voice + bed
mix = np.tanh(mix * 1.15) / np.tanh(1.15)
mix *= np.clip((DUR - t) / 0.35, 0, 1)
mix = mix / np.abs(mix).max() * 0.93
sf.write("audio.wav", np.stack([mix, mix], 1), SR, subtype="PCM_16")

sos = butter(4, [300, 4000], "bandpass", fs=SR, output="sos")
vb, bb = sosfilt(sos, voice), sosfilt(sos, bed)
for a, b in ((L0, L0 + 2.2), (L1, SLAM), (SLAM, VOICE_END)):
    i, j = int(a * SR), int(b * SR)
    print(f"voz/trilha (faixa da fala) {a:.2f}-{b:.2f}s: {20*np.log10(np.sqrt(np.mean(vb[i:j]**2))/np.sqrt(np.mean(bb[i:j]**2))):.1f} dB")

# ---------- dados por quadro para a animação ----------
hop = SR // FPS
mouth = np.array([np.sqrt(np.mean(voice[j * hop:(j + 1) * hop] ** 2)) for j in range(int(DUR * FPS))])
mouth = np.clip(mouth / mouth.max() * 1.35, 0, 1)
energy = np.array([np.sqrt(np.mean(music[j * hop:(j + 1) * hop] ** 2)) for j in range(int(DUR * FPS))])
energy /= energy.max()
json.dump({"fps": FPS, "dur": DUR, "L0": L0, "L1": L1, "slam": SLAM, "beat": BEAT, "rollEnd": ROLL_END,
           "voiceEnd": VOICE_END, "cues": {"sexta": CUE_SEXTA, "outubro": CUE_OUTUBRO},
           "mouth": [round(float(x), 3) for x in mouth],
           "music": [round(float(x), 3) for x in energy]}, open("timeline.json", "w"))
print("ok", len(mouth), "quadros; slam", round(SLAM, 2), "fim da voz", round(VOICE_END, 2))
