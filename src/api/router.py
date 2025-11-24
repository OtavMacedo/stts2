from pathlib import Path
from fastapi import APIRouter

from src.api.schemas import GetList, TTSRequest
from src.core.tts_engine import TTSInferenceEngine

from fastapi.responses import StreamingResponse

from src.core.utils import save_audio_to_buffer

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


@router.get("/list/speakers", response_model=GetList)
def list_speakers():
    return {"speakers": list(reference_dicts.keys())}


@router.get("/list/audio_formats", response_model=GetList)
def list_audio_formats():
    return {"audio_formats": ["wav", "mp3", "opus", "pcm"]}


@router.post("/tts/streaming")
async def generate_audio_streaming(request: TTSRequest):
    tts_engine = TTSInferenceEngine.factory()
    ref_path = reference_dicts[request.speaker]
    ref_s = tts_engine.compute_style(ref_path)
    amostras = tts_engine.inference(request.text, ref_s)
    buffer, media_type = save_audio_to_buffer(
        data=amostras, audio_format=request.audio_format, sample_rate=tts_engine.rate
    )
    return StreamingResponse(buffer, media_type=media_type)
