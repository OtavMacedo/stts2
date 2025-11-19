from pathlib import Path
from fastapi import APIRouter

from src.api.schemas import TTSRequest
from src.core.tts_engine import TTSInferenceEngine

from fastapi.responses import StreamingResponse
import io
import torch
import torchaudio
from scipy.io.wavfile import write

router = APIRouter()

wavs_path = Path(__file__).resolve().parent.parent / "wavs_reference"

reference_dicts = {
    "antonio": wavs_path / "Antonio_0.wav",
    "brenda": wavs_path / "Brenda_0.wav",
    "donato": wavs_path / "Donato_0.wav",
    "elza": wavs_path / "Elza_5.wav",
    "fabio": wavs_path / "Fabio_6.wav",
    "francisca": wavs_path / "Francisca_0.wav",
    "giovanna": wavs_path / "Giovanna_0.wav",
    "humberto": wavs_path / "Humberto_0.wav",
    "julio": wavs_path / "Julio_16.wav",
    "keren": wavs_path / "Keren_0.wav",
    "manuela": wavs_path / "Manuela_0.wav",
    "nicolau": wavs_path / "Nicolau_8.wav",
    "thalita": wavs_path / "Thalita_12.wav",
    "valerio": wavs_path / "Valerio_4.wav",
    "yara": wavs_path / "Yara_2.wav",
}


@router.get("/")
def home():
    return {
        "status": "online",
        "speakers_disponiveis": list(reference_dicts.keys()),
        "message": "Envie JSON com 'texto', 'speaker' e 'formato' 🎧",
    }


@router.post("/tts/streaming")
async def gerar_audio(request: TTSRequest):
    tts_engine = TTSInferenceEngine.factory()
    ref_path = reference_dicts[request.speaker]
    ref_s = tts_engine.compute_style(ref_path)
    amostras = tts_engine.inference(request.text, ref_s)

    buffer = io.BytesIO()
    # tensor = torch.tensor(amostras).unsqueeze(0)  # [1, N]
    # if request.audio_format == "mp3":
    #     torchaudio.save(
    #         buffer,
    #         tensor,
    #         sample_rate=tts_engine.rate,
    #         format="mp3",
    #         encoding="MP3",
    #         bits_per_sample=16,
    #         backend="ffmpeg",
    #     )
    #     media_type = "audio/mpeg"
    # else:
    #     torchaudio.save(
    #         buffer,
    #         tensor,
    #         sample_rate=tts_engine.rate,
    #         format="wav",
    #         backend="ffmpeg",
    #     )
    #     media_type = "audio/wav"
    write(buffer, tts_engine.rate, amostras)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="audio/wav")
