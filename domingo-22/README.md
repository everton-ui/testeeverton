# Domingo é 22

Vídeo vertical (1080×1920, 10 s, 30 fps, H.264 + AAC) nas cores da bandeira do Brasil:
um personagem animado segurando a bandeira diz *"Hoje é sexta-feira, 2 de outubro. Domingo é 22!"*.

- `domingo-22.mp4`: vídeo final
- `capa.jpg`: imagem de capa (quadro aos 7 s)

## Como regenerar

Tudo é gerado localmente: animação em canvas, voz sintética
[Kokoro](https://github.com/thewh1teagle/kokoro-onnx) em pt-BR e trilha de samba sintetizada.

```bash
cd src
python3 -m venv venv && ./venv/bin/pip install kokoro-onnx soundfile numpy scipy
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
./venv/bin/python voice.py     # line0.wav, line1.wav
./venv/bin/python audio.py     # audio.wav + timeline.json (sincronia da boca)
npm install && node render.js  # out.mp4 (precisa de Chromium e ffmpeg)
```

Para mudar o tempo de cada fala, ajuste `L0`/`L1` em `audio.py`. Os textos e as cores ficam em `video.html`.
