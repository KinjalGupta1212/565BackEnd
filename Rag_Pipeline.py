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
import chromadb

from chromadb.utils.embedding_functions import HuggingFaceEmbeddingFunction

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
        embeddings = HuggingFaceEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ----------------------------
        # STEP 3: LOAD VECTOR DB
        # ----------------------------
        client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = client.get_collection(name="data-annotation", embedding_function=embeddings)

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

        results = self.collection.similarity_search(
            query,
            k=5
        )

        context = "\n\n".join([doc.page_content for doc in results])
        return context

    async def get_response(self, user_query, attributes, history=None):
        context = await self.retrieve_context(user_query)

        llm_subgroups = []
        
        llm_suggestions = {}

        attribute_prompts = {"Sentiment": "How would you describe the sentiment of this comment?", 
                             "Respect": "Is this comment respectful towards the group(s) you previously identified?",
                             "Insult": "Do you agree or disagree with the following statement: This comment is insulting towards the group(s) you previously identified.",
                             "Humiliate": "Do you agree or disagree with the following statement: This comment is humiliating towards the group(s) you previously identified.",
                             "Status": "This comment states that the group(s) you previously identified is:",
                             "Dehumanize": "Do you agree or disagree with the following statement: This comment dehumanizes the group(s) you previously identified (e.g., by comparing them to an animal).",
                             "Violence": "Do you agree or disagree with the following statement: This comment calls for using violence against the group(s) you previously identified.", 
                             "Genocide": "Do you agree or disagree with the following statement: This comment calls for the deliberate killing of a large group of people from the group(s) you previously identified.",
                             "Attack-Defend": "Is the comment attacking or defending the group(s) you previously identified?", 
                             "Hate speech": "Does this comment contain hate speech, defined as “bias-motivated, hostile and malicious language targeted at a person/group because of their actual or perceived innate characteristics, especially when the group is unnecessarily labeled?"}
        
        attribute_response_options = {"Sentiment": "Strongly negative, somewhat negative, neutral, somewhat positive, strongly positive",
                                      "Respect": "Strongly disrespectful, disrespectful, neutral, respectful, strongly respectful", 
                                      "Insult": "Strongly disagree, disagree, neither disagree nor agree, agree, strongly agree", 
                                      "Humiliate": "Strongly disagree, disagree, neither disagree nor agree, agree, strongly agree", 
                                      "Status": "Strongly inferior, inferior, neither superiornor inferior, superior, strongly superior", 
                                      "Dehumanize": "Strongly disagree, disagree, neither disagree nor agree, agree, strongly agree", 
                                      "Violence": "Strongly disagree, disagree, neither disagree nor agree, agree, strongly agree", 
                                      "Genocide": "Strongly disagree, disagree, neither disagree nor agree, agree, strongly agree", 
                                      "Attack-Defend": "Strongly defending, defending, neither defending nor attacking, attacking, strongly attacking", 
                                      "Hate speech": "Yes, no, unclear"}
        
        system_prompt = (
            "You are a data annotation assistant. You are given a comment. "
            "First, is the comment directed at or about any individuals or groups based on race/ethnicity, religion, national origin or citizenship status, gender, sexual orientation, age, disability status, political identity. You can also say none."
            "Next, based on the identified groups, identify what subgroups from this dictionary the comment targets. "
            "{Race or ethnicity: Black or African American, Latino or non-white Hispanic, Asian, Middle Eastern, Native American or Alaska Native, Pacific Islander, Non-hispanic white"
            " Religion:  Jews, Christians, Buddhists, Hindus, Mormons, Atheists, Muslims" 
            " National origin or citizenship status: A specific country, immigrant, migrant worker, undocumented person"
            " Gender identity: Women, men, non-binary or third gender, transgender women, transgender men, transgender (unspecified)"
            " Sexual orientation: Bisexual, gay, lesbian, heterosexual",
            " Age: Children (0 - 12 years old), adolescents / teenagers (13 - 17), young adults / adults (18 - 39), middle-aged (40 - 64), seniors (65 or older)"
            " Disability status: People with physical disabilities (e.g., use of wheelchair), people with cognitive disorders (e.g., autism) or learning disabilities (e.g., Down syndrome), people with mental health problems (e.g., depression, addiction), visually impaired people, hearing impaired people, no specific disability}"
            " Output the subgroups that were targeted through this comment, in this format: 'Subgroup 1,Subgroup 2,...,Subgroup N'"  
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nComment: {user_query}"}
        ]
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.llm_model,
                messages=messages,
                max_tokens=200,
                temperature=0.4
            )      
        except Exception as e:
            print(f"LLM error: {e}")
            return "Sorry, there was a problem generating a response."
        
        llm_subgroups = response.choices[0].message.content.split(",")

        
        for attribute in enumerate(attributes):
            system_prompt = (
                "You are a data annotation assistant. You are given a comment."
                f" You are given 5 examples of similar comments. You are also given the groups and subgroups targeted in the comment. Use the examples and groups and subgroups as guidance to answer {attribute_prompts[attribute]}. Select your response from: {attribute_response_options[attribute]}"
                f" Here are the subgroups that are targeted: {llm_subgroups.join(",")}"  
                " Second, output 3 or 4 questions that guide the annotator through the reasoning needed to annotate the comment according to the attribute. "
                f" Third, output the response you select and the (if it exists) the options that comes before and after your answer in {attribute_response_options[attribute]}. Make sure that your output is the response options in a comma seperated format." 
            )
            

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nComment: {user_query}\n\nAttribute: {attribute}"}
            ]

            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.llm_model,
                    messages=messages,
                    max_tokens=200,
                    temperature=0.4
                )
                
                llm_suggestions[attribute] = response.choices[0].message.content
                
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