from typing import Literal

from pydantic import BaseModel

Speaker = Literal[
    "antonio",
    "brenda",
    "donato",
    "elza",
    "fabio",
    "francisca",
    "giovanna",
    "humberto",
    "julio",
    "keren",
    "manuela",
    "nicolau",
    "thalita",
    "valerio",
    "yara",
]

AudioFormat = Literal["wav", "mp3", "opus", "pcm"]

GetList = dict[str, list]


class TTSRequest(BaseModel):
    text: str
    speaker: Speaker
    audio_format: AudioFormat
