import json
import numpy as np, soundfile as sf
from scipy.signal import resample_poly, butter, sosfilt

SR = 48000
DUR = 10.0
FPS = 30
L0, L1 = 0.80, 3.50          # voice line start times
SLAM = L1 + 0.70             # "vinte e dois" -> 22 slam
BEAT = 0.6                   # 100 bpm, grid anchored on the slam
rng = np.random.default_rng(22)
N = int(SR * DUR)
t = np.arange(N) / SR

def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], "bandpass", fs=SR, output="sos"), x)
def hp(x, f): return sosfilt(butter(2, f, "highpass", fs=SR, output="sos"), x)
def lp(x, f): return sosfilt(butter(2, f, "lowpass", fs=SR, output="sos"), x)
def add(buf, sig, at, gain=1.0):
    i = int(at * SR)
    if i >= len(buf): return
    sig = sig[: len(buf) - i]
    buf[i:i + len(sig)] += gain * sig

# ---------- voice ----------
voice = np.zeros(N)
for f, at in (("line0.wav", L0), ("line1.wav", L1)):
    v, sr = sf.read(f)
    v = resample_poly(v, SR, sr)
    add(voice, v / np.abs(v).max(), at, 0.92)

VOICE_END = L1 + sf.info('line1.wav').duration

# ---------- instruments ----------
def surdo(strength=1.0, muted=False):
    d = 0.25 if muted else 0.7
    tt = np.arange(int(SR * d)) / SR
    f = 52 + 60 * np.exp(-tt * 30)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * (14 if muted else 5))
    s += 0.25 * lp(rng.standard_normal(len(tt)), 900) * np.exp(-tt * 60)
    return strength * s

def caixa(strength=1.0):
    tt = np.arange(int(SR * 0.12)) / SR
    n = bp(rng.standard_normal(len(tt)), 1800, 9000) * np.exp(-tt * 45)
    return strength * (0.6 * n + 0.2 * np.sin(2 * np.pi * 210 * tt) * np.exp(-tt * 60))

def tamborim(strength=1.0):
    tt = np.arange(int(SR * 0.08)) / SR
    s = np.sin(2 * np.pi * 1150 * tt) * np.exp(-tt * 70)
    s += 0.4 * bp(rng.standard_normal(len(tt)), 3000, 8000) * np.exp(-tt * 120)
    return strength * s

def ganza(strength=1.0):
    tt = np.arange(int(SR * 0.07)) / SR
    env = np.minimum(tt / 0.012, 1) * np.exp(-tt * 50)
    return strength * hp(rng.standard_normal(len(tt)), 6000) * env

def apito(d=0.22):
    tt = np.arange(int(SR * d)) / SR
    trill = 0.65 + 0.35 * np.sign(np.sin(2 * np.pi * 34 * tt))
    env = np.minimum(tt / 0.01, 1) * np.minimum((d - tt) / 0.02, 1)
    return np.sin(2 * np.pi * (2900 + 40 * np.sin(2 * np.pi * 34 * tt)) * tt) * trill * env

def boom():
    tt = np.arange(int(SR * 2.5)) / SR
    f = 38 + 120 * np.exp(-tt * 9)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 1.6)
    crash = hp(rng.standard_normal(len(tt)), 3500) * np.exp(-tt * 1.4) * 0.35
    return s + crash

def saw_chord(freqs, d, attack=0.3, release=0.6, cutoff=1600):
    tt = np.arange(int(SR * d)) / SR
    s = np.zeros(len(tt))
    for fq in freqs:
        for det in (-0.12, 0.0, 0.12):          # detuned saws -> lush pad
            ph = (fq * (1 + det / 100 * 3)) * tt + rng.random()
            s += 2 * (ph % 1) - 1
    s = lp(s / (3 * len(freqs)), cutoff)
    env = np.minimum(tt / attack, 1) * np.clip((d - tt) / release, 0, 1)
    return s * env

def brass_stab(freqs, d=0.45):
    tt = np.arange(int(SR * d)) / SR
    s = np.zeros(len(tt))
    for fq in freqs:
        s += 2 * ((fq * tt) % 1) - 1
    cutoff_env = 600 + 4000 * np.exp(-tt * 9)
    # crude time-varying LPF: blend of two static filters
    bright, dark = lp(s, 4600), lp(s, 700)
    mix = (cutoff_env - 600) / 4000
    out = (bright * mix + dark * (1 - mix)) / len(freqs)
    return out * np.minimum(tt / 0.015, 1) * np.exp(-tt * 3.2)

def applause(d):
    n = int(SR * d)
    s = np.zeros(n)
    clap_len = int(SR * 0.012)
    ct = np.arange(clap_len) / SR
    clap_env = np.exp(-ct * 400)
    for _ in range(int(d * 900)):
        i = rng.integers(0, n - clap_len)
        s[i:i + clap_len] += rng.standard_normal(clap_len) * clap_env * rng.uniform(0.3, 1)
    s = bp(s, 900, 7000)
    tt = np.arange(n) / SR
    return s / np.abs(s).max() * np.minimum(tt / 0.3, 1) * np.clip((d - tt) / 1.5, 0, 1)

def hz(midi): return 440 * 2 ** ((midi - 69) / 12)
D  = [hz(m) for m in (50, 57, 62, 66, 69)]     # D major
Bm = [hz(m) for m in (47, 54, 59, 62, 66)]
G  = [hz(m) for m in (43, 50, 55, 59, 62)]
A  = [hz(m) for m in (45, 52, 57, 61, 64)]

music = np.zeros(N)
# intro pad: emotional build D - Bm - G - A (tension into the slam)
for chord, a, b in ((D, 0.0, 1.2), (Bm, 1.2, 2.4), (G, 2.4, 3.3), (A, 3.3, SLAM + 0.05)):
    add(music, saw_chord(chord, b - a + 0.6, attack=0.25, release=0.6, cutoff=1100 + 500 * a / SLAM), a, 0.55)
# heartbeat surdo pre-slam, growing
k = 1
while SLAM - k * BEAT > 0.1:
    at = SLAM - k * BEAT
    add(music, surdo(0.35 + 0.5 * (1 - at / SLAM)), at)
    k += 1
# riser
rd = 1.1
tt = np.arange(int(SR * rd)) / SR
riser = bp(rng.standard_normal(len(tt)), 400, 9000) * (tt / rd) ** 2.5
add(music, riser, SLAM - rd, 0.45)
# slam
impact = np.zeros(N)
add(impact, boom(), SLAM, 1.1)   # kept out of the ducking so the slam still hits
add(music, brass_stab(D, 0.9), SLAM, 0.5)
for i in range(3):
    add(music, apito(0.18 if i < 2 else 0.45), VOICE_END + 0.05 + i * 0.28, 0.22)
# celebration: batucada + pad
ROLL_END = 9.25
for chord, a, b in ((D, SLAM, SLAM + 4 * BEAT), (G, SLAM + 4 * BEAT, SLAM + 6 * BEAT),
                    (A, SLAM + 6 * BEAT, SLAM + 8 * BEAT), (D, SLAM + 8 * BEAT, DUR)):
    add(music, saw_chord(chord, b - a + 0.5, attack=0.08, release=0.5, cutoff=2200), a, 0.5)
    add(music, brass_stab(chord), a, 0.45)
teleco = [1, 0, 1, 1, 0, 1, 1, 0]
step = BEAT / 4
i = int(np.ceil((VOICE_END - 0.03 - SLAM) / step))   # groove kicks in right after "dois!"
while SLAM + i * step < ROLL_END:
    at = SLAM + i * step
    if i % 4 == 0:
        add(music, surdo(1.0 if (i // 4) % 2 else 0.55, muted=not (i // 4) % 2), at)
    add(music, caixa([0.9, 0.35, 0.55, 0.7][i % 4]), at, 0.55)
    if teleco[i % 8]: add(music, tamborim(), at, 0.35)
    add(music, ganza(), at, 0.18)
    i += 1
add(music, boom() * 0.6, ROLL_END)
add(music, brass_stab(D, 1.2), ROLL_END, 0.7)
add(music, applause(DUR - VOICE_END + 0.1), VOICE_END - 0.1, 0.22)

# ---------- mix with ducking under the voice ----------
venv_ = np.convolve(np.abs(voice), np.ones(int(SR * 0.08)) / int(SR * 0.08), "same")
duck = 1 - 0.78 * np.clip(venv_ * 8, 0, 1)
peak = np.abs(music + impact).max()
music, impact = music / peak, impact / peak
# extra dip under "vinte e dois"
win = np.ones(N)
i, j = int((SLAM - 0.02) * SR), int((VOICE_END + 0.05) * SR)
win[i:j] = 0.45
win = np.convolve(win, np.ones(int(SR * 0.04)) / int(SR * 0.04), 'same')
bed = 0.62 * music * duck * win + 0.62 * impact * (1 - 0.4 * (win < 1))
for a, b in ((L0, L0 + 2.1), (L1, VOICE_END)):
    i, j = int(a * SR), int(b * SR)
    print(f'voice/music {a:.1f}-{b:.1f}s: {20*np.log10(np.sqrt(np.mean(voice[i:j]**2))/np.sqrt(np.mean(bed[i:j]**2))):.1f} dB')
mix = voice + bed
mix = np.tanh(mix * 1.15) / np.tanh(1.15)
fade = np.clip((DUR - t) / 0.35, 0, 1)
mix *= fade
mix = mix / np.abs(mix).max() * 0.93
sf.write("audio.wav", np.stack([mix, mix], 1), SR, subtype="PCM_16")

# ---------- per-frame data for the animation ----------
hop = SR // FPS
mouth = np.array([np.sqrt(np.mean(voice[j * hop:(j + 1) * hop] ** 2)) for j in range(int(DUR * FPS))])
mouth = np.clip(mouth / mouth.max() * 1.35, 0, 1)
beat_energy = np.array([np.sqrt(np.mean(music[j * hop:(j + 1) * hop] ** 2)) for j in range(int(DUR * FPS))])
beat_energy /= beat_energy.max()
json.dump({"fps": FPS, "dur": DUR, "L0": L0, "L1": L1, "slam": SLAM, "beat": BEAT, "rollEnd": ROLL_END,
           "mouth": [round(float(x), 3) for x in mouth],
           "music": [round(float(x), 3) for x in beat_energy]}, open("timeline.json", "w"))
print("ok", len(mouth), "frames")

sos = butter(4, [300, 4000], "bandpass", fs=SR, output="sos")
vb, bb = sosfilt(sos, voice), sosfilt(sos, bed)
for a, b in ((L0, L0 + 2.1), (L1, SLAM), (SLAM, VOICE_END)):
    i, j = int(a * SR), int(b * SR)
    print(f'speech-band voice/music {a:.2f}-{b:.2f}s: {20*np.log10(np.sqrt(np.mean(vb[i:j]**2))/np.sqrt(np.mean(bb[i:j]**2))):.1f} dB')
