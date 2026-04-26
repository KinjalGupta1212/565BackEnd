import os
import asyncio
import openai
import boto3
from dotenv import load_dotenv
import chromadb
import pandas as pd
import json
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import traceback
load_dotenv()

class DataAnnotationRAG:
    def __init__(self):
        # OpenAI client
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm_model  = os.getenv("LLM_MODEL", "gpt-4o")

        # download chroma DB
        # self.download_directory("rag-chatbot-bucket-565", "./chroma_db")

        # embeddings model
        embeddings = SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )

        # load vector DB
        client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = client.get_collection(name="data-annotation", embedding_function=embeddings)
        self.unaggregated_df = pd.read_csv("measure_hate_speech.csv")

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
        
        results = self.collection.query(query_texts=[query], n_results=200)
        context = results["documents"][0]
        return context

    async def get_response(self, user_query, attributes, history=None):
        context = await self.retrieve_context(user_query)
       
        #for each unique comment, go to dataframe and get all annotators annotations
        #aggregate the annotations for each requested attribute 
        unique_comments_list = []

        example_mean_labels = {}

        final_response = {}

        attribute_response_options_list = {"sentiment": ["strongly negative", "somewhat negative", "neutral", "somewhat positive", "strongly positive"],
                                      "respect": ["strongly disrespectful", "disrespectful", "neutral", "respectful", "strongly respectful"], 
                                      "insult": ["strongly disagree", "disagree", "neither disagree nor agree", "agree", "strongly agree"], 
                                      "humiliate": ["strongly disagree", "disagree", "neither disagree nor agree", "agree", "strongly agree"], 
                                      "status": ["strongly inferior", "inferior", "neither superior nor inferior", "superior", "strongly superior"], 
                                      "dehumanize": ["strongly disagree", "disagree", "neither disagree nor agree", "agree", "strongly agree"], 
                                      "violence": ["strongly disagree", "disagree", "neither disagree nor agree", "agree", "strongly agree"],
                                      "genocide": ["strongly disagree", "disagree", "neither disagree nor agree", "agree", "strongly agree"],
                                      "attack_defend": ["strongly defending", "defending", "neither defending nor attacking", "attacking", "strongly attacking"],
                                      "hatespeech": ["yes", "unclear", "no"]
        }        
        
        for comment in context:
            if comment not in unique_comments_list:
                unique_comments_list.append(comment)
        print(len(unique_comments_list))
        for i, comment in enumerate(unique_comments_list):
            if i > 4:
                break
            all_annotations = self.unaggregated_df.loc[self.unaggregated_df['text'] == comment]
            s = all_annotations[attributes].mean().round(0).astype(int)
            
            numerical_responses = s.tolist()
            categorical_responses = {}
            for i, attribute in enumerate(attributes):
                len_resp_opt_list = len(attribute_response_options_list[attribute])
                categorical_responses[attribute] = [(attribute_response_options_list[attribute][len_resp_opt_list-numerical_responses[i]-1])]
        
            example_mean_labels[comment] = categorical_responses

        
        llm_subgroups = []
        
        attribute_prompts = {"sentiment": "How would you describe the sentiment of this comment?", 
                             "respect": "Is this comment respectful towards the group(s) you previously identified?",
                             "insult": "Do you agree or disagree with the following statement: This comment is insulting towards the group(s) you previously identified.",
                             "humiliate": "Do you agree or disagree with the following statement: This comment is humiliating towards the group(s) you previously identified.",
                             "status": "This comment states that the group(s) you previously identified is:",
                             "dehumanize": "Do you agree or disagree with the following statement: This comment dehumanizes the group(s) you previously identified (e.g., by comparing them to an animal).",
                             "violence": "Do you agree or disagree with the following statement: This comment calls for using violence against the group(s) you previously identified.", 
                             "genocide": "Do you agree or disagree with the following statement: This comment calls for the deliberate killing of a large group of people from the group(s) you previously identified.",
                             "attack_defend": "Is the comment attacking or defending the group(s) you previously identified?", 
                             "hatespeech": "Does this comment contain hate speech, defined as “bias-motivated, hostile and malicious language targeted at a person/group because of their actual or perceived innate characteristics, especially when the group is unnecessarily labeled?"}
        
        attribute_response_options_string = {"sentiment": "strongly negative, somewhat negative, neutral, somewhat positive, strongly positive",
                                      "respect": "strongly disrespectful, disrespectful, neutral, respectful, strongly respectful", 
                                      "insult": "strongly disagree, disagree, neither disagree nor agree, agree, strongly agree", 
                                      "humiliate": "strongly disagree, disagree, neither disagree nor agree, agree, strongly agree", 
                                      "status": "strongly inferior, inferior, neither superior nor inferior, superior, strongly superior", 
                                      "dehumanize": "strongly disagree, disagree, neither disagree nor agree, agree, strongly agree", 
                                      "violence": "strongly disagree, disagree, neither disagree nor agree, agree, strongly agree", 
                                      "genocide": "strongly disagree, disagree, neither disagree nor agree, agree, strongly agree", 
                                      "attack_defend": "strongly defending, defending, neither defending nor attacking, attacking, strongly attacking", 
                                      "hatespeech": "yes, unclear, no"}
        
        system_prompt = (
            "You are a data annotation assistant. You are given a comment. "
            "First, is the comment directed at or about any individuals or groups based on Race or ethnicity, Religion, National origin or citizenship status, Gender identity, Sexual orientation, Age, Disability status, Political identity. You can also say none. "
            "The dictionary below has keys that are equal to demographic groups, and values that list the subgroups within those demographic groups"
            "Next, determine whether the comment is directed at, about, references, praises, criticizes, stereotypes, insults, or discusses any individual from a group or subgroup, or a group or subgroup themselves, based on the dictionary below. "
            
            # "This includes positive, negative, neutral, indirect, slang, abbreviated, or explicit references. "
            # "A group DOES NOT need to be attacked to count. "
            # "If the comment simply mentions or discusses a listed group, include it. "
            "{Race or ethnicity: [Black or African American, Latino or non-white Hispanic, Asian, Middle Eastern, Native American or Alaska Native, Pacific Islander, Non-hispanic white], "
            " Religion: [Jews, Christians, Buddhists, Hindus, Mormons, Atheists, Muslims]," 
            " National origin or citizenship status: [a specific country, immigrant, migrant worker, undocumented person],"
            " Gender identity: [women, men, non-binary or third gender, transgender women, transgender men, transgender (unspecified)],"
            " Sexual orientation: [bisexual, gay, lesbian, heterosexual],"
            " Age: Children [(0 - 12 years old), adolescents / teenagers (13 - 17), young adults / adults (18 - 39), middle-aged (40 - 64), seniors (65 or older)],"
            " Disability status: [people with physical disabilities (e.g., use of wheelchair), people with cognitive disorders (e.g., autism) or learning disabilities (e.g., Down syndrome), people with mental health problems (e.g., depression, addiction), visually impaired people, hearing impaired people, no specific disability],"
            " Political identity: [alt-right (Alternative Right), communist, conservative, democrat, green, leftist, liberal, libertarian, republican, socialist, other]}"
            # " none: [none]}"
            " Output the subgroups you selected AND the reasoning behind selecting these subgroups in JSON format using the OUTPUT RULES below."
            "OUTPUT RULES: "
            "1. Output ONLY valid JSON. "
            "2. Each JSON key MUST be the exact demographic group name from the dictionary (e.g., \"Race or ethnicity\", \"Religion\", \"Political identity\"). "
            "3. NEVER use placeholder labels like group1, group2, subgroup1, or subgroup2. "
            "4. Each value must be a list of the exact subgroup names selected from that demographic group. "
            "5. Include a \"reasoning\" key explaining why each subgroup was selected. "
            "6. If NO demographic group, subgroup, individual, or identity category from the dictionary is targeted, referenced, mentioned, discussed, praised, criticized, stereotyped, or insulted in any way, output EXACTLY: {\"none\": [\"none\"]} "
            "7. When outputting {\"none\": [\"none\"]}, do NOT include reasoning or any other keys. "
            "8. Use ONLY keys from this set when applicable: Race or ethnicity, Religion, National origin or citizenship status, Gender identity, Sexual orientation, Age, Disability status, Political identity, reasoning. "
            "9. Example valid output with group: {\"Political identity\": [\"democrat\"], \"reasoning\": \"The comment explicitly mentions Democrats, which maps to the political identity subgroup democrat.\"} "
            "10. Example valid output with no group: {\"none\": [\"none\"]}"
            #" Output the subgroups you selected AND the reasoning behind selecting these subgroups in this JSON format: {\"group1\": [\"subgroup1\", \"subgroup2\", ...], \"group2\": [...], ..., \"reasoning\": \"...\"}"
            # " If there are no groups or subgroups targeted in the comment, output: {\"none\": [\"none\"]}"  
        )





        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nComment: {user_query}"}
        ]
        print("FIRST API CALL")
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.llm_model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=1200,
                temperature=0.4
            )      
        except Exception as e:
            print(f"LLM error: {e}")
            traceback.print_exc()
            return "Sorry, there was a problem generating a response."
        
        # llm_subgroups = response.choices[0].message.content.split(",")
        llm_subgroups = json.loads(response.choices[0].message.content)
        llm_subgroups_final = {k: v for k, v in llm_subgroups.items() if k != 'reasoning'}

        
        print("DONE WITH FIRST API CALL")
        print(llm_subgroups_final)
        #the list will contain the suggestion for each attribute: [["agree", "strongly agree"], ["disagree", "neutral", "agree"],...]
        example_mean_labels["LLM Suggestion For Your Comment"] = {}

        system_prompt = (
                        "You are a data annotation assistant. You are given a comment."
                        f" You are given 5 examples of similar comments. You are also given the groups and subgroups targeted in the comment. Use the examples and groups and subgroups as guidance to answer the questions needed from: {attribute_prompts}. Select the appropriate response needed for the attributes from: {attribute_response_options_string}"
                        " You are asked to annotate the comment on multiple attributes"
                        f" Here are the subgroups that are targeted: {json.dumps(llm_subgroups)}"  
                        " Second, for each attribute, output 3 or 4 questions that guide the annotator through the reasoning needed to annotate the comment. "
                        f"Third, for each attribute, select the most appropriate response from {attribute_response_options_string} for that attribute. "
                        "Always return the response as a list of strings. "
                        "For non-hatespeech attributes, include the selected response plus the option immediately before and after it in the scale (if they exist). "
                        "For hatespeech, still return a list containing only the selected response. "
                        "Do not return strings or single values—everything must always be a list."
                        "Your final output should be formatted in JSON like this: {\"attribute1\": { \"Questions\": [\"question1\", \"question2\",...], \"Response\": [\"responseoption1\", \"responseoption2\",...]}, \"attribute2\": {...}}"
                    )

        messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nComment: {user_query}\n\nAttributes: {attributes}"}
            ]
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.llm_model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=1200,
                temperature=0.4
            )
            # response_dict = json.loads(response.choices[0].message.content)
            try:
                response_dict = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                print("RAW LLM OUTPUT:\n", response.choices[0].message.content)
                return {"error": "Invalid JSON from LLM", "raw": response.choices[0].message.content}
            
            for attribute in attributes:
            
                example_mean_labels["LLM Suggestion For Your Comment"][attribute] = response_dict[attribute]["Response"]
                guiding_questions = response_dict[attribute]["Questions"]

                similar_comments = []
                disagreement_distribution_per_comment = {}
                similar_comment_count = 0
                disagreement_comment_count = 0
                
                for i, comment in enumerate(unique_comments_list): 
                    all_annotations = self.unaggregated_df.loc[self.unaggregated_df['text'] == comment]
                    range_val = all_annotations[attribute].max() - all_annotations[attribute].min()
                    if range_val <= 1.0 and similar_comment_count < 2:
                        similar_comment_count += 1
                        similar_comments.append(comment)
                    elif range_val > 1.0 and disagreement_comment_count < 2:
                        disagreement_distribution_per_comment[comment] = {}
                        for response_option in attribute_response_options_list[attribute]:
                            disagreement_distribution_per_comment[comment][response_option] = 0
                        for _, row in all_annotations.iterrows():
                            len_of_response_options_list = len(attribute_response_options_list[attribute])
                            annotator_rating = int(row[attribute])
                            disagreement_distribution_per_comment[comment][attribute_response_options_list[attribute][len_of_response_options_list-annotator_rating-1]] += 1 
                        disagreement_comment_count += 1

                    if similar_comment_count >= 2 and disagreement_comment_count >= 2:
                        break  
                final_response[attribute] = {}
                final_response[attribute]["questions"] = guiding_questions
                final_response[attribute]["similar_comments"] = similar_comments
                final_response[attribute]["disagreeing_comments"] = disagreement_distribution_per_comment        
        except Exception as e:
            print(f"LLM error: {e}")
            traceback.print_exc()
            return "Sorry, there was a problem generating a response."

        final_response["table_info"] = example_mean_labels
        final_response["targeted_subgroups"] = llm_subgroups_final
        return final_response

def test_agent():
    agent = DataAnnotationRAG()

    response = asyncio.run(agent.get_response("democrats are bad", ["sentiment"]))
    print(response)

if __name__ == "__main__":
    test_agent()