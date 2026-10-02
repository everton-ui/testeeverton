# Falas em pt-BR (Kokoro "pm_alex") com tom mais grave/adulto:
# um pouco mais pausada, -0.9 semitom de timbre e -2.6 semitons de tom (formantes preservados).
import subprocess
import numpy as np, soundfile as sf
from kokoro_onnx import Kokoro

k = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
for i, text in enumerate(["Hoje é sexta-feira, dois de outubro!", "Domingo é vinte e dois!"]):
    samples, sr = k.create(text, voice="pm_alex", speed=0.97, lang="pt-br")
    sf.write(f"line{i}_raw.wav", samples, sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"line{i}_raw.wav", "-af",
                    f"asetrate={sr}*0.95,aresample={sr},atempo=1/0.95,"
                    "rubberband=pitch=0.86:formant=preserved:pitchq=quality,"
                    "lowshelf=f=180:g=2.5",
                    f"line{i}.wav"], check=True)
