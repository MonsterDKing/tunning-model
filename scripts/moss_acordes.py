"""Pide a MOSS-Music-8B-Instruct la progresión de acordes (con tiempos) de cada WAV de una carpeta.

Se ejecuta en la máquina con GPU, desde la carpeta del repositorio de MOSS-Music y con SU entorno
(torch 2.9.1 + transformers 4.57.1), porque usa su código (``src.modeling_moss_music``…):

    cd moss-music
    .venv\\Scripts\\python.exe ..\\tunning-model\\scripts\\moss_acordes.py audios\\test salidas_moss\\test

Usa la instrucción de su demo (app.py) y decodificación voraz (sin azar) para que sea reproducible.
Guarda la respuesta en bruto: la interpretación y la puntuación se hacen después en el portátil.

Con un tercer argumento (segundos) parte cada audio en trozos de esa duración y pregunta por cada
uno: con canciones largas, la canción entera llena los 24 GB de la GPU y sus tiempos derivan. Cada
trozo devuelve tiempos desde 0; se guardan con su inicio (``trozos``) y ``puntuar_moss.py`` los suma.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))  # el código de MOSS-Music está en ./src

import torch  # noqa: E402

import soundfile  # noqa: E402
import torchaudio  # noqa: E402
from src.modeling_moss_music import MossMusicModel  # noqa: E402
from src.processing_moss_music import MossMusicProcessor  # noqa: E402

MODELO = "./weights/MOSS-Music-8B-Instruct"
PROMPT = "Transcribe the Chords progression with timestamps of this song, use json format to output the result."



def cargar_audio(ruta: Path, sample_rate: int):
    """Como src.audio_io.load_audio, pero sin torchcodec (en Windows pide las DLL de FFmpeg): el WAV
    se lee con soundfile, se pasa a mono y se remuestrea con torchaudio.functional (puro torch)."""
    datos, sr = soundfile.read(str(ruta), dtype="float32", always_2d=True)
    onda = torch.from_numpy(datos.T).mean(0, keepdim=True)
    if sr != sample_rate:
        onda = torchaudio.functional.resample(onda, orig_freq=sr, new_freq=sample_rate)
    return onda.squeeze(0).numpy()


def transcribir(audio) -> str:
    entradas = procesador(text=PROMPT, audios=[audio], return_tensors="pt").to(modelo.device)
    if entradas.get("audio_data") is not None:
        entradas["audio_data"] = entradas["audio_data"].to(modelo.dtype)
    entradas["audio_input_mask"] = entradas["input_ids"] == procesador.audio_token_id
    with torch.no_grad():
        generado = modelo.generate(**entradas, max_new_tokens=6000, do_sample=False, num_beams=1, use_cache=True)
    return procesador.decode(generado[0, entradas["input_ids"].shape[1]:], skip_special_tokens=True)


entrada, salida = Path(sys.argv[1]), Path(sys.argv[2])
trozo = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
salida.mkdir(parents=True, exist_ok=True)
modelo = MossMusicModel.from_pretrained(MODELO, trust_remote_code=True, torch_dtype="auto", device_map="cuda:0").eval()
procesador = MossMusicProcessor.from_pretrained(MODELO, trust_remote_code=True, enable_time_marker=True)
for wav in sorted(entrada.glob("*.wav")):
    destino = salida / f"{wav.stem}.json"
    if destino.exists():
        continue
    t0 = time.time()
    audio = cargar_audio(wav, procesador.config.mel_sr)
    sr = procesador.config.mel_sr
    paso = int(trozo * sr) if trozo > 0 else len(audio)
    trozos = []
    try:
        for inicio in range(0, len(audio), paso):
            pedazo = audio[inicio:inicio + paso]
            if len(pedazo) < sr:  # menos de 1 s al final: no merece la pena
                continue
            try:
                texto = transcribir(pedazo)
            except RuntimeError:  # cuDNN a veces no encuentra kernel en Windows: sin él, convoluciones propias
                torch.cuda.empty_cache()
                torch.backends.cudnn.enabled = False
                texto = transcribir(pedazo)
                torch.backends.cudnn.enabled = True
            trozos.append({"inicio": inicio / sr, "respuesta": texto})
    except Exception as e:  # noqa: BLE001 - una canción que falla no para las demás
        print(f"{wav.stem[:20]}: ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
        torch.cuda.empty_cache()
        continue
    datos = {"audio": wav.name, "segundos": time.time() - t0}
    datos.update({"trozos": trozos} if trozo > 0 else {"respuesta": trozos[0]["respuesta"]})
    destino.write_text(json.dumps(datos, ensure_ascii=False))
    print(f"{wav.stem[:20]}: {time.time() - t0:5.1f} s · {len(trozos)} trozos", flush=True)


