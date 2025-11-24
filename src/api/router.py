from pathlib import Path
from fastapi import APIRouter

from src.api.schemas import GetList, TTSRequest
from src.core.tts_engine import TTSInferenceEngine

from fastapi.responses import StreamingResponse

from src.core.utils import save_audio_to_buffer

router = APIRouter()

wavs_path = Path(__file__).resolve().parent.parent / "wavs_reference"

reference_dicts = {
    "antonio": "Antonio_0.wav",
    "brenda": "Brenda_0.wav",
    "donato": "Donato_0.wav",
    "elza": "Elza_5.wav",
    "fabio": "Fabio_6.wav",
    "francisca": "Francisca_0.wav",
    "giovanna": "Giovanna_0.wav",
    "humberto": "Humberto_0.wav",
    "julio": "Julio_16.wav",
    "keren": "Keren_0.wav",
    "manuela": "Manuela_0.wav",
    "nicolau": "Nicolau_8.wav",
    "thalita": "Thalita_12.wav",
    "valerio": "Valerio_4.wav",
    "yara": "Yara_2.wav",
}


@router.get("/list/speakers", response_model=GetList)
def list_speakers():
    return {"speakers": list(reference_dicts.keys())}


@router.get("/list/audio_formats", response_model=GetList)
def list_audio_formats():
    return {"audio_formats": ["wav", "mp3", "opus", "pcm"]}


@router.post("/tts/streaming")
async def generate_audio_streaming(request: TTSRequest):
    tts_engine = TTSInferenceEngine.factory()
    ref_path = wavs_path / reference_dicts[request.speaker]
    ref_s = tts_engine.compute_style(ref_path)
    amostras = tts_engine.inference(request.text, ref_s)
    buffer, media_type = save_audio_to_buffer(
        data=amostras, audio_format=request.audio_format, sample_rate=tts_engine.rate
    )
    return StreamingResponse(buffer, media_type=media_type)
