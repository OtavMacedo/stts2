from src.core.tts_engine import TTSInferenceEngine


model = None


def init_worker():
    global model
    model = TTSInferenceEngine()
