# run_me_once_download_e5_large_v2.py
from huggingface_hub import snapshot_download
from pathlib import Path

REPO_ID = "intfloat/e5-large-v2"
LOCAL_DIR = Path("models/e5-large-v2")  # change to wherever you prefer

LOCAL_DIR.mkdir(parents=True, exist_ok=True)

# This pulls *all* necessary files to LOCAL_DIR (no symlinks => fully self-contained)
snapshot_download(
    repo_id=REPO_ID,
    local_dir=str(LOCAL_DIR),
    local_dir_use_symlinks=False,  # important if you want a copy not tied to HF cache
    revision="main",               # optional; pin a tag/sha if you want reproducibility
)
print(f"Downloaded {REPO_ID} into: {LOCAL_DIR.resolve()}")

