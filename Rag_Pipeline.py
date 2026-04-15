import os
import asyncio
import openai
from dotenv import load_dotenv
from embedding import get_db_vectorstore
import asyncio

load_dotenv()

class DataAnnotationRAG:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o")
        self.vector_store = get_db_vectorstore()

    async def retrieve_context(self, query):
        print(f"[RAG] Retrieving context for query: {query}")

        results = self.vector_store.query(
            query_texts=[query]
        )
        context = results['documents']
        return context

    async def get_response(self, user_query, history=None):
        """Generate a response using OpenAI and retrieved context."""
        context = await self.retrieve_context(user_query)

        system_prompt = (
            "You are a data annotation assistant. "
            "The annotation task is annotating text based on these categories: Sentiment, Respect, Insult, Humiliate, Status, Dehumanize, Violence, Genocide, Attack Defend, Hatespeech"
            "You will be provided text documents and similar examples and should answer questions based off of that"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\nUser: {user_query}"}
        ]

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.llm_model,
                messages=messages,
                max_tokens=120,
                temperature=0.4
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM error: {e}")
            return "Sorry, there was a problem generating a response."

def test_agent():
    agent = DataAnnotationRAG()
    print("Please enter your question:")
    user_query = input()
    print("You entered:", user_query)
    response =  asyncio.run(agent.get_response(user_query))
    print(response)

if __name__ == "__main__":
    test_agent()