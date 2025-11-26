from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.api.schemas import GetList, TTSRequest
from src.core.tts_engine import reference_dicts
from src.core.worker import TTSWorker

router = APIRouter()

wavs_path = Path(__file__).resolve().parent.parent / "wavs_reference"


@router.get("/list/speakers", response_model=GetList)
def list_speakers():
    return {"speakers": list(reference_dicts.keys())}


@router.get("/list/audio_formats", response_model=GetList)
def list_audio_formats():
    return {"audio_formats": ["wav", "mp3", "opus", "pcm"]}


@router.post("/tts/streaming")
async def generate_audio_streaming(
    request: TTSRequest, worker: TTSWorker = Depends(TTSWorker.factory)
):
    buffer, media_type = await worker.infer(
        request.text, request.speaker, request.audio_format
    )
    return StreamingResponse(buffer, media_type=media_type)
