import os
from huggingface_hub import snapshot_download
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN was not given")

try:
    snapshot_download(
        repo_id="SipPulse/styletts2-pt-br",
        allow_patterns=["models/*", "voices/*"],
        local_dir=".",
        token=HF_TOKEN,
    )

except Exception:
    exit(1)


repo_path = Path(__file__).resolve().parent

for file in os.listdir(repo_path / "models"):
    file_names = {
        "styletts2_main_multi_speaker.pth": "epoch_base_2nd_00149.pth",
        "phoneme_aligner.pth": "epoch_base_00050.pth",
        "pitch_extractor.pth": "epoch_base_00100.pth",
    }
    os.rename(repo_path / "models" / file, repo_path / file_names[file])

os.removedirs(repo_path / "models")
