import random

import pandas as pd
import json

def select_examples():
  df = pd.read_csv("measure_hate_speech.csv")
  df = df[df['platform'] != 1]

  attributes = [
          'sentiment',
          'respect', 
          'insult', 
          'humiliate', 
          'status', 
          'dehumanize', 
          'violence', 
          'genocide', 
          'attack_defend', 
          'hatespeech']

  selected_texts = {}     
  
  for attribute in attributes:
    if attribute == 'hatespeech':
      for i in range(3):
        sub_df = df[df[attribute] == i] #all rows where df[attribute] = rating
        random_num1 = random.randint(0, len(sub_df))
        random_num2 = random.randint(0, len(sub_df))
        
        while f"{sub_df.iloc[random_num1]['text']}" in list(selected_texts.values()):
          random_num1 = random.randint(0, len(sub_df))
        selected_texts[f"{attribute} {i} first"] = (f"{sub_df.iloc[random_num1]['text']}")

        while f"{sub_df.iloc[random_num2]['text']}" in list(selected_texts.values()):
          random_num2 = random.randint(0, len(sub_df))
        selected_texts[f"{attribute} {i} second"] = (f"{sub_df.iloc[random_num2]['text']}")
    else:
      for i in range(5):
        sub_df = df[df[attribute] == i] #all rows where df[attribute] = rating

        random_num1 = random.randint(0, len(sub_df))
        random_num2 = random.randint(0, len(sub_df))
        
        while f"{sub_df.iloc[random_num1]['text']}" in list(selected_texts.values()):
          random_num1 = random.randint(0, len(sub_df))
        selected_texts[f"{attribute} {i} first"] = (f"{sub_df.iloc[random_num1]['text']}")

        while f"{sub_df.iloc[random_num2]['text']}" in list(selected_texts.values()):
          random_num2 = random.randint(0, len(sub_df))                
        selected_texts[f"{attribute} {i} second"] = (f"{sub_df.iloc[random_num2]['text']}")
        # if i == 4:
        #   print(sub_df)
    
  print(selected_texts)
  
def find_comments():
  with open('example_comments.json', 'r') as f:
    example_dict = json.load(f)

  df = pd.read_csv("measure_hate_speech.csv")
  df = df[df['platform'] != 1]

  text_set = set(df['text'])
  for k, v in example_dict.items():
    if v not in text_set:
      print("KEY:", k)
      print(v)

def select_annotation_comments():
  df = pd.read_csv("measure_hate_speech.csv")
  df = df[df['platform'] != 1]
  all_same = df.groupby('comment_id')['text'].nunique().eq(1)
  print(all_same)
  df = df.groupby('comment_id').filter(lambda d: d['text'].nunique() == 1)
  
 #aggregation by mean for numeric columns and first for text column
  df = df.groupby('comment_id').agg({
    'comment_id': 'mean',
    'platform': 'mean',
    'sentiment': 'mean',
    'respect': 'mean',
    'insult': 'mean',
    'humiliate': 'mean',
    'status': 'mean',
    'dehumanize': 'mean',
    'violence': 'mean',
    'genocide': 'mean',
    'attack_defend': 'mean',
    'hatespeech': 'mean',
    'hate_speech_score': 'mean',
    'text': 'first',
  })

  numeric_columns = df.select_dtypes(include='number').columns
  df[numeric_columns] = df[numeric_columns].round().astype(int)

  random_nums = random.sample(range(0, len(df)), 20)
  comments_df = df.iloc[random_nums]
  comments_df.to_csv("annotation_comments.csv")
      


if __name__ == "__main__":
  select_annotation_comments()
