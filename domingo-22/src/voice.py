# Gera as falas com a voz pt-BR "pm_alex" do Kokoro (offline).
import soundfile as sf
from kokoro_onnx import Kokoro

k = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
for i, text in enumerate(["Hoje é sexta-feira, dois de outubro!", "Domingo é vinte e dois!"]):
    samples, sr = k.create(text, voice="pm_alex", speed=1.05, lang="pt-br")
    sf.write(f"line{i}.wav", samples, sr)
