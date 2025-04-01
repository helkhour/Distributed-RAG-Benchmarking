# embedding_utils.py
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def generate_embedding(self, text):
        return self.model.encode(text).tolist()