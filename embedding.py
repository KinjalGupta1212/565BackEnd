import chromadb
import sqlite3
from chromadb.config import Settings
from preprocess import load_data
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

def get_db_vectorstore():
  collection = chroma_client.get_or_create_collection(name="data-annotation", embedding_function=SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cpu"
  ))
  df = load_data()
  
  documents = []
  
  # fiter comments that our annotators will annotate
  annotation_comments_df = pd.read_csv("annotation_comments.csv")
  comments = set(annotation_comments_df['text'])
  df = df[~df['text'].isin(comments)]
  
  for row in df.itertuples(index=False):
    if f"{row.text}" not in documents:
      documents.append(f"{row.text}")

  max_batch_size = 5000
  for start_idx in range(0, len(documents), max_batch_size):
    end_idx = start_idx + max_batch_size
    batch_documents = documents[start_idx:end_idx]
    batch_ids = [f"id{num}" for num in range(start_idx, min(end_idx, len(documents)))]
    collection.add(documents=batch_documents, ids=batch_ids)

  return collection

if __name__ == "__main__":
   get_db_vectorstore()
