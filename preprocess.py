import datasets 

def load_data():
  # load dataset
  dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', token='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')   
  df = dataset['train'].to_pandas()
  df = df[df['platform'] != 1]

  df = df[['comment_id', 
           'annotator_id',
           'platform', 
           'sentiment', 
           'respect', 
           'insult', 
           'humiliate', 
           'status', 
           'dehumanize', 
           'violence', 
           'genocide', 
           'attack_defend', 
           'hatespeech', 
           'text']]

  print(df.describe())
  df.to_csv("measure_hate_speech.csv")
  return df
  

if __name__ == "__main__":
  load_data()