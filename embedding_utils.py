# embedding_utils.py
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from config import MODEL_CONFIGS

class EmbeddingGenerator:
    def __init__(self, model_name, embedding_size, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        base_model = MODEL_CONFIGS[model_name].get("base_model", model_name)
        if "Llama-3.1" in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                cache_dir="/home/ubuntu/rag_project/llama-3.1-8b",
                use_auth_token=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                cache_dir="/home/ubuntu/rag_project/llama-3.1-8b",
                torch_dtype=torch.float16,
                device_map="auto",
                use_auth_token=True
            )
            self.model.eval()
        else:
            self.model = SentenceTransformer(base_model)
        self.model_name = model_name
        self.embedding_size = embedding_size

        if "mxbai-embed-large-v1" in model_name:
            self.output_size = embedding_size
        elif "gte-base" in model_name and embedding_size == 384:
            self.projection = nn.Linear(768, 384)
            self.projection.to(self.device)

    def generate_embedding(self, text):
        if "Llama-3.1" in self.model_name:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
            num_tokens = torch.sum(attention_mask, dim=1)
            embedding = sum_embeddings / num_tokens
        else:
            embedding = self.model.encode(text, convert_to_tensor=True)
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)

        if hasattr(self, "output_size"):
            embedding = embedding[:, :self.output_size]
        elif hasattr(self, "projection"):
            embedding = self.projection(embedding)

        return embedding.squeeze(0).cpu().tolist()