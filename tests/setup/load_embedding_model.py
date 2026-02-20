from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

def ensure_model_cached(model_name: str):
    """
    Downloads the model into local cache if not already present.
    After this, your script will load it instantly from disk.
    """
    print(f"Checking cache for: {model_name}")
    model = AutoModel.from_pretrained(model_name, cache_dir=None)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=None)
    print("Model is now cached locally.")

def load_model_from_cache(model_name: str):
    """
    Loads the model strictly from local cache without network.
    """
    print("Loading model from local cache...")
    model = SentenceTransformer(model_name, cache_folder=None)
    print("Loaded successfully from cache.")
    return model


if __name__ == "__main__":
    model_name = "sentence-transformers/all-mpnet-base-v2"

    # Step 1: Ensure it is downloaded into HF cache
    ensure_model_cached(model_name)

    # Step 2: Load from cache (no re-download)
    model = load_model_from_cache(model_name)

    # Test embedding
    vec = model.encode("hello world")
    print("Embedding shape:", vec.shape)