# embedding_utils.py
from sentence_transformers import SentenceTransformer
import torch.nn as nn
from config import MODEL_CONFIGS

class EmbeddingGenerator:
    def __init__(self, model_name, embedding_size):
        # Use base_model if provided, otherwise use model_name
        base_model = MODEL_CONFIGS[model_name].get("base_model", model_name)
        self.model = SentenceTransformer(base_model)
        self.model_name = model_name
        self.embedding_size = embedding_size

        if "mxbai-embed-large-v1" in model_name:
            self.output_size = embedding_size
        elif "gte-base" in model_name and embedding_size == 384:
            self.projection = nn.Linear(768, 384)                      # test - shrinking output size to 384 
# we attach a projection layer to the oject of the class to reduce the output size to 384 in generate_embedding

    def generate_embedding(self, text):
        embedding = self.model.encode(text, convert_to_tensor=True)

        #ensure batch is 2d for secuirty. 
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0) 

        if hasattr(self, "output_size"):      # ----- self truncation to compare effect of output size
            # For mxbai-embed-large-v1 --> truncate to output_size
            return embedding[:, :self.output_size].squeeze(0).tolist()
        

        elif hasattr(self, "projection"):
            # For gte-base-384, i apply projection   ---- check effect of this truncation 
            return self.projection(embedding).squeeze(0).tolist()
        return embedding.squeeze(0).tolist()