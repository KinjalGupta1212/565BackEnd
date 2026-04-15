# import os
# import asyncio
# import openai
# from dotenv import load_dotenv
# from embedding import get_db_vectorstore
# import asyncio
# import boto3
# from langchain_openai import OpenAIEmbeddings

# load_dotenv()

# class DataAnnotationRAG:
#     def __init__(self):
#         self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#         self.llm_model = os.getenv("LLM_MODEL", "gpt-4o")
#         self.download_directory("rag-chatbot-bucket-565", "./chroma_db")
#         self.vector_store = Chroma(
#             persist_directory="./chroma_db",
#             embedding_function=embeddings
#         )

#     def download_directory(bucket, local_dir):
#         s3 = boto3.client("s3")
#         os.makedirs(local_dir, exist_ok=True)

#         objects = s3.list_objects_v2(Bucket=bucket)

#         for obj in objects.get("Contents", []):
#             key = obj["Key"]
#             local_path = os.path.join(local_dir, key)

#             os.makedirs(os.path.dirname(local_path), exist_ok=True)
#             s3.download_file(bucket, key, local_path)
            
#     async def retrieve_context(self, query):
#         print(f"[RAG] Retrieving context for query: {query}")

#         results = self.vector_store.query(
#             query_texts=[query]
#         )
#         context = results['documents']
#         return context

#     async def get_response(self, user_query, history=None):
#         """Generate a response using OpenAI and retrieved context."""
#         context = await self.retrieve_context(user_query)

#         system_prompt = (
#             "You are a data annotation assistant. "
#             "The annotation task is annotating text based on these categories: Sentiment, Respect, Insult, Humiliate, Status, Dehumanize, Violence, Genocide, Attack Defend, Hatespeech"
#             "You will be provided text documents and similar examples and should answer questions based off of that"
#         )

#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": f"Context: {context}\nUser: {user_query}"}
#         ]

#         try:
#             response = await asyncio.to_thread(
#                 self.client.chat.completions.create,
#                 model=self.llm_model,
#                 messages=messages,
#                 max_tokens=120,
#                 temperature=0.4
#             )
#             return response.choices[0].message.content
#         except Exception as e:
#             print(f"LLM error: {e}")
#             return "Sorry, there was a problem generating a response."

# def test_agent():
#     agent = DataAnnotationRAG()
#     print("Please enter your question:")
#     user_query = input()
#     print("You entered:", user_query)
#     response =  asyncio.run(agent.get_response(user_query))
#     print(response)

# if __name__ == "__main__":
#     test_agent()

import os
import asyncio
import openai
import boto3
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()


class DataAnnotationRAG:
    def __init__(self):
        # OpenAI client
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o")

        # ----------------------------
        # STEP 1: DOWNLOAD CHROMA DB
        # ----------------------------
        self.download_directory("rag-chatbot-bucket-565", "./chroma_db")

        # ----------------------------
        # STEP 2: LOAD EMBEDDINGS MODEL
        # ----------------------------
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ----------------------------
        # STEP 3: LOAD VECTOR DB
        # ----------------------------
        self.vector_store = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )

    # FIXED: must include self
    def download_directory(self, bucket, local_dir):
        s3 = boto3.client("s3")
        os.makedirs(local_dir, exist_ok=True)

        objects = s3.list_objects_v2(Bucket=bucket)

        for obj in objects.get("Contents", []):
            key = obj["Key"]
            local_path = os.path.join(local_dir, key)

            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, key, local_path)

    async def retrieve_context(self, query):
        print(f"[RAG] Retrieving context for query: {query}")

        results = self.vector_store.similarity_search(
            query,
            k=5
        )

        context = "\n\n".join([doc.page_content for doc in results])
        return context

    async def get_response(self, user_query, history=None):
        context = await self.retrieve_context(user_query)

        system_prompt = (
            "You are a data annotation assistant. "
            "You help label text using: Sentiment, Respect, Insult, Humiliate, "
            "Status, Dehumanize, Violence, Genocide, Attack Defend, Hatespeech. "
            "Use retrieved examples to guide answers."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nUser: {user_query}"}
        ]

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.llm_model,
                messages=messages,
                max_tokens=200,
                temperature=0.4
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"LLM error: {e}")
            return "Sorry, there was a problem generating a response."


def test_agent():
    agent = DataAnnotationRAG()

    print("Enter question:")
    user_query = input()

    response = asyncio.run(agent.get_response(user_query))
    print(response)


if __name__ == "__main__":
    test_agent()