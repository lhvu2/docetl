from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

def ensure_model_cached(model_name: str):
    """
    Downloads the model into the HuggingFace local cache if not already present.
    After this completes once, you will not need to re-download the model again.
    """
    print(f"Checking cache for: {model_name}")
    # This will download to ~/.cache/huggingface/hub/
    _ = AutoModel.from_pretrained(model_name)
    _ = AutoTokenizer.from_pretrained(model_name)
    print("Model is now cached locally.")


def load_model_from_cache(model_name: str):
    """
    Loads the SentenceTransformer model *only* from local cache.
    No network calls will be made if the model is already cached.
    """
    print("Loading model from local cache...")
    model = SentenceTransformer(model_name)
    print("Loaded successfully from cache.")
    return model


if __name__ == "__main__":
    model_name = "BAAI/bge-base-en-v1.5"
    

    # Step 1: One-time download into HuggingFace cache
    ensure_model_cached(model_name)

    # Step 2: Load from cache (no re-download)
    model = load_model_from_cache(model_name)

    # Optional: test embedding
    vec = model.encode("hello world")
    print("Embedding shape:", vec.shape)