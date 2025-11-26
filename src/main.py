import asyncio
import contextlib

from fastapi import FastAPI

from src.core.worker import TTSWorker

from .api.router import router


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    tts_worker = TTSWorker.factory()
    asyncio.create_task(tts_worker.start_loop())
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router)
