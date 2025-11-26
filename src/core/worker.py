import asyncio
from typing import Optional

from src.api.schemas import AudioFormat, Speaker
from src.core.tts_engine import SAMPLE_RATE, TTSInferenceEngine
from src.core.utils import save_audio_to_buffer


class TTSWorker:
    _instance: Optional["TTSWorker"] = None

    def __init__(self):
        self.model = TTSInferenceEngine()
        self.queue = asyncio.Queue()
        self.running = False

    @classmethod
    def factory(cls):
        if cls._instance is None:
            cls._instance = TTSWorker()
        return cls._instance

    async def infer(self, text: str, speaker: Speaker, audio_format: AudioFormat):
        fut = asyncio.get_event_loop().create_future()
        await self.queue.put((text, speaker, audio_format, fut))
        return await fut

    async def start_loop(self):
        self.running = True
        while self.running:
            text, speaker, audio_format, fut = await self.queue.get()
            audio = await asyncio.to_thread(
                self._run_infer, text, speaker, audio_format
            )
            fut.set_result(audio)

    def _run_infer(self, text: str, speaker: Speaker, audio_format: AudioFormat):
        audio_bytes = self.model.inference(text, speaker)
        return save_audio_to_buffer(audio_bytes, audio_format, SAMPLE_RATE)
