from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp

from fastapi import FastAPI
import contextlib

from src.core.worker import init_worker

from .api.router import router
from .core.settings import settings

mp.set_start_method("spawn", force=True)

executor: ProcessPoolExecutor | None = None


def get_executor() -> ProcessPoolExecutor:
    if executor is None:
        raise RuntimeError("Executor not initialized")
    return executor


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global executor

    executor = ProcessPoolExecutor(
        max_workers=settings.NUM_WORKERS,
        initializer=init_worker,
        mp_context=mp.get_context("spawn"),
    )
    yield

    if executor:
        executor.shutdown()


app = FastAPI(lifespan=lifespan)
app.include_router(router)
