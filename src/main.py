from fastapi import FastAPI
import contextlib

from .api.router import router
from .core.tts_engine import TTSInferenceEngine


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    TTSInferenceEngine.factory()
    yield


app = FastAPI()
app.include_router(router)
