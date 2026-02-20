from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

MODEL_ID = "intfloat/multilingual-e5-large-instruct"  # 100+ langs, instruction-tuned

# One-time download to HF cache
_ = AutoModel.from_pretrained(MODEL_ID)
_ = AutoTokenizer.from_pretrained(MODEL_ID)

# Load from cache in your app
model = SentenceTransformer(MODEL_ID)

def instruct_query(q: str) -> str:
    return f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {q}"

q = instruct_query("how much protein should a female eat")
d = "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day..."

# Encode (model applies mean pooling + L2 norm internally)
qv, dv = model.encode([q, d])
print("cosine * 100:", float((qv @ dv) * 100))

