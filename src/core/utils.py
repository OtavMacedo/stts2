from io import BytesIO

import soundfile as sf
from munch import Munch

from src.api.schemas import AudioFormat


def save_audio_to_buffer(data, audio_format: AudioFormat, sample_rate: int):
    buffer = BytesIO()

    if audio_format == "opus":
        sf.write(buffer, data, sample_rate, format="OGG", subtype="OPUS")
        media_type = "audio/ogg; codecs=opus"
    elif audio_format == "pcm":
        sf.write(buffer, data, sample_rate, format="RAW", subtype="PCM_16")
        media_type = f"audio/L16; rate={sample_rate}; channels=1"
    elif audio_format == "wav":
        sf.write(buffer, data, sample_rate, format="WAV")
        media_type = "audio/wav"
    elif audio_format == "mp3":
        sf.write(buffer, data, sample_rate, format="MP3")
        media_type = "audio/mpeg"

    buffer.seek(0)
    return buffer, media_type


def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
