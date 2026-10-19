"""Policy retrieval over finguard/../policies/*.md. TF-IDF baseline (no embeddings yet); swap for
dense retrieval and compare in evals."""
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POLICY_DIR = Path(__file__).resolve().parent.parent / "policies"


class PolicyIndex:
    def __init__(self):
        self.docs = {p.stem: p.read_text() for p in sorted(POLICY_DIR.glob("P-*.md"))}
        self.ids = list(self.docs)
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.mat = self.vec.fit_transform(self.docs.values())

    def search(self, query: str, k: int = 3):
        sims = cosine_similarity(self.vec.transform([query]), self.mat)[0]
        top = sims.argsort()[::-1][:k]
        return [{"policy_id": self.ids[i], "score": round(float(sims[i]), 3), "text": self.docs[self.ids[i]]} for i in top]
