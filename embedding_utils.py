import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator(nn.Module):
    """Generate embeddings for text using specified model."""
    
    def __init__(self, model_name, output_size=None, quantize=False):
        super().__init__()
        self.model_name = model_name
        self.output_size = output_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if "Llama-3.1" in model_name:
            base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                cache_dir="/home/ubuntu/rag_project/llama-3.1-8b",
                token=True
            )
            # Set pad_token to eos_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            ) if quantize else None
            
            self.model = AutoModel.from_pretrained(
                base_model,
                quantization_config=quantization_config,
                cache_dir="/home/ubuntu/rag_project/llama-3.1-8b",
                torch_dtype=torch.float16,
                device_map="auto",
                token=True
            )
            self.model.eval()
        else:
            self.model = SentenceTransformer(model_name).to(self.device)
            self.model.eval()
        
        if output_size:
            hidden_size = self.model.config.hidden_size if "Llama-3.1" in model_name else self.model.get_sentence_embedding_dimension()
            self.projection = nn.Linear(hidden_size, output_size, dtype=torch.float16).to(self.device)  # Cast to float16
            self.projection.eval()
    
    def generate_embedding(self, texts):
        """Generate embeddings for a list of texts."""
        if "Llama-3.1" in self.model_name:
            try:
                # Handle single text or list
                if isinstance(texts, str):
                    texts = [texts]
                inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)  # Outputs last_hidden_state directly
                last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
                # Mean pool over sequence length (dim=1)
                attention_mask = inputs["attention_mask"].unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)
                sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)  # Shape: (batch_size, hidden_size)
                num_tokens = torch.sum(attention_mask, dim=1)  # Shape: (batch_size, 1)
                embeddings = sum_embeddings / num_tokens  # Shape: (batch_size, hidden_size)
                # Project to desired output size if specified
                if hasattr(self, "output_size"):
                    embeddings = self.projection(embeddings)
                return embeddings.cpu().tolist()
            except Exception as e:
                print(f"Error generating embedding: {e}")
                raise
        else:
            embedding = self.model.encode(texts, convert_to_tensor=True, batch_size=32)
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            if hasattr(self, "output_size"):
                embedding = embedding[:, :self.output_size]
            elif hasattr(self, "projection"):
                embedding = self.projection(embedding)
            return embedding.cpu().tolist()