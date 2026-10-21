"""Toca con un piano de soundfont (FluidR3_GM, programa 0) las notas exportadas y guarda un WAV.

Se ejecuta con un entorno aparte que tenga numpy y tinysoundfont (sin pyaudio):

    venv_sintesis/bin/python -I scripts/sintetizar.py data/audio/test data/sintesis/.../FluidR3_GM.sf2
"""

import json
import sys
import wave
from pathlib import Path

import numpy as np
import tinysoundfont

SR = 22050
carpeta, soundfont = Path(sys.argv[1]), sys.argv[2]
for ruta in sorted(carpeta.glob("*.json")):
    datos = json.loads(ruta.read_text())
    synth = tinysoundfont.Synth(samplerate=SR, gain=-6)
    sfid = synth.sfload(soundfont)
    synth.program_select(0, sfid, 0, 0)  # piano de cola acústico
    eventos = sorted([(t, 1, a) for t, d, a in datos["notas"]] + [(t + max(d, 0.05), 0, a) for t, d, a in datos["notas"]])
    trozos, ahora = [], 0.0
    for t, encender, altura in eventos:
        n = int(round((t - ahora) * SR))
        if n > 0:
            trozos.append(np.frombuffer(synth.generate(n), dtype=np.float32).copy())
            ahora += n / SR
        synth.noteon(0, altura, 80) if encender else synth.noteoff(0, altura)
    trozos.append(np.frombuffer(synth.generate(2 * SR), dtype=np.float32).copy())  # cola de 2 s
    estereo = np.concatenate(trozos).reshape(-1, 2)
    mono = estereo.mean(1)
    mono = mono / max(float(np.abs(mono).max()), 1e-6) * 0.9  # volumen normal: pico en 0,9
    with wave.open(str(ruta.with_suffix(".wav")), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((mono * 32767).astype(np.int16).tobytes())
    print(f"{ruta.stem[:20]}: {len(mono) / SR:6.1f} s")
