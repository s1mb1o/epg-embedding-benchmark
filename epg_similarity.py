import numpy as np


PHRASE_PAIRS = [
    ("Evening News", "Evening News Broadcast"),
    ("Morning Show", "Good Morning Program"),
    ("Live Football", "Football Match Live"),
    ("Documentary Premiere", "Documentary Film Premiere"),
    ("Cooking Competition", "Culinary Contest"),
]

MODELS = [
    ("st", "all-MiniLM-L6-v2"),
    ("openai", "text-embedding-3-small"),
]


def cosine_similarity(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def embed_st(texts, model_name):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    vecs = model.encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def embed_openai(texts, model_name):
    import os
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.embeddings.create(model=model_name, input=list(texts))
    arr = np.array([item.embedding for item in resp.data], dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


for api, model in MODELS:
    print(f"\n=== {api}/{model} ===")
    texts_a = [p[0] for p in PHRASE_PAIRS]
    texts_b = [p[1] for p in PHRASE_PAIRS]

    if api == "st":
        vecs = embed_st(texts_a + texts_b, model)
    else:
        vecs = embed_openai(texts_a + texts_b, model)

    n = len(PHRASE_PAIRS)
    scores = []
    for i, (a, b) in enumerate(PHRASE_PAIRS):
        score = cosine_similarity(vecs[i], vecs[n + i])
        scores.append(score)
        print(f"  {a!r:<40} vs {b!r:<40} -> {score:.4f}")
    print(f"  mean: {sum(scores)/len(scores):.4f}")
