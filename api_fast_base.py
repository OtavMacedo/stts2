import contextlib
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import io
import torch
import torchaudio
import main_base as tts


tts_engine = None


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_engine
    tts_engine = tts.TTSInferenceEngine()
    yield


app = FastAPI(title="TTS Streaming API", lifespan=lifespan)

reference_dicts = {
    "Antonio": ("Masculino", "./wavs_reference/Antonio_0.wav"),
    "Brenda": ("Feminino", "./wavs_reference/Brenda_0.wav"),
    "Donato": ("Masculino", "./wavs_reference/Donato_0.wav"),
    "Elza": ("Feminino", "./wavs_reference/Elza_5.wav"),
    "Fabio": ("Masculino", "./wavs_reference/Fabio_6.wav"),
    "Francisca": ("Feminino", "./wavs_reference/Francisca_0.wav"),
    "Giovanna": ("Feminino", "./wavs_reference/Giovanna_0.wav"),
    "Humberto": ("Masculino", "./wavs_reference/Humberto_0.wav"),
    "Julio": ("Masculino", "./wavs_reference/Julio_16.wav"),
    "Keren": ("Feminino", "./wavs_reference/Keren_0.wav"),
    "Manuela": ("Feminino", "./wavs_reference/Manuela_0.wav"),
    "Nicolau": ("Masculino", "./wavs_reference/Nicolau_8.wav"),
    "Thalita": ("Feminino", "./wavs_reference/Thalita_12.wav"),
    "Valerio": ("Masculino", "./wavs_reference/Valerio_4.wav"),
    "Yara": ("Feminino", "./wavs_reference/Yara_2.wav"),
}


@app.get("/")
def home():
    return {
        "status": "online",
        "speakers_disponiveis": list(reference_dicts.keys()),
        "message": "Envie JSON com 'texto', 'speaker' e 'formato' 🎧",
    }


@app.post("/tts")
async def gerar_audio(request: Request):
    # Parsing rápido
    try:
        data = await request.json()
        texto = data.get("texto", "").strip()
        speaker = data.get("speaker", "Humberto").strip()
        formato = data.get("formato", "wav").strip().lower()
    except Exception:
        return {"error": "JSON inválido"}, 400
    if not texto:
        return {"error": "Texto vazio"}, 400

    try:
        # 1. INFERÊNCIA (mantém separado)
        _, ref_path = reference_dicts[speaker]
        ref_s = tts_engine.compute_style(ref_path)
        amostras = tts_engine.inference(
            texto, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1
        )

        # 2. BUFFER OTIMIZADO
        buffer = io.BytesIO()
        tensor = torch.tensor(amostras).unsqueeze(0)  # [1, N]
        if formato == "mp3":
            torchaudio.save(
                buffer,
                tensor,
                sample_rate=tts_engine.rate,
                format="mp3",
                encoding="MP3",
                bits_per_sample=16,
                backend="ffmpeg",
            )
            media_type = "audio/mpeg"
        else:
            torchaudio.save(
                buffer,
                tensor,
                sample_rate=tts_engine.rate,
                format="wav",
                backend="ffmpeg",
            )
            media_type = "audio/wav"
        buffer.seek(0)

        return StreamingResponse(buffer, media_type=media_type)

    except Exception as e:
        print(f"ERRO: {e}")
        return {"error": str(e)}, 500
