import chromadb
import sqlite3

from preprocess import load_data


chroma_client = chromadb.Client()

def get_db_vectorstore():
  collection = chroma_client.create_collection(name="my_collection")
  df = load_data()
  
  # for each row in the dataframe, we want to make a string like:
  # "Text: row['text'], Platform: row['platform'], "
  
  documents = []
  for row in df.itertuples(index=False):
    documents.append(f"Text: {row.text}, Platform: {row.platform}, Sentiment: {row.sentiment}, Respect: {row.respect}, Insult: {row.insult}, Humiliate: {row.humiliate}, Status: {row.status}, Dehumanize: {row.dehumanize}, Violence: {row.violence}, Genocide: {row.genocide}, Attack Defend: {row.attack_defend}, Hate Speech: {row.hatespeech}")
  
#   with open('general_annotation_guidelines.txt', 'r') as f:
#       documents.append(f.read())

#   with open('twitter_guidelines.txt', 'r') as f:
#       documents.append(f.read())
  
#   with open('youtube_guidelines.txt', 'r') as f:
#       documents.append(f.read())
  
#   with open('reddit_guidelines.txt', 'r') as f:
#       documents.append(f.read())

  max_batch_size = 5000
  for start_idx in range(0, len(documents), max_batch_size):
    end_idx = start_idx + max_batch_size
    batch_documents = documents[start_idx:end_idx]
    batch_ids = [f"id{num}" for num in range(start_idx, min(end_idx, len(documents)))]
    collection.add(documents=batch_documents, ids=batch_ids)

  return collection

if __name__ == "__main__":
   get_db_vectorstore()
