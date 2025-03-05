from tqdm import tqdm
import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt
from langchain.docstore.document import Document as LangchainDocument
import datasets

pd.set_option(
    "display.max_colwidth", None
)  # This will be helpful when visualizing retriever outputs

ds = datasets.load_dataset("m-ric/huggingface_doc", split="train")


RAW_KNOWLEDGE_BASE = [
    LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) 
    for doc in tqdm(ds)
]

print(f"Loaded {len(RAW_KNOWLEDGE_BASE)} documents into RAW_KNOWLEDGE_BASE")
