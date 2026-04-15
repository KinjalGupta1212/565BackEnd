import datasets 

def load_data():
  # load dataset
  dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', token='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')   
  df = dataset['train'].to_pandas()

  # remove hate_speech_score depending on how we decide to measure accuracy
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
           'hate_speech_score', 
           'text']]
  # all_same = df.groupby('comment_id')['text'].nunique().eq(1)
  # print(all_same)
  # df = df.groupby('comment_id').filter(lambda d: d['text'].nunique() == 1)
  
 # aggregation by mean for numeric columns and first for text column
  # df = df.groupby('comment_id').agg({
  #   'comment_id': 'mean',
  #   'platform': 'mean',
  #   'sentiment': 'mean',
  #   'respect': 'mean',
  #   'insult': 'mean',
  #   'humiliate': 'mean',
  #   'status': 'mean',
  #   'dehumanize': 'mean',
  #   'violence': 'mean',
  #   'genocide': 'mean',
  #   'attack_defend': 'mean',
  #   'hatespeech': 'mean',
  #   'hate_speech_score': 'mean',
  #   'text': 'first',
    
  # })
  print(df.describe())
  return df
  

if __name__ == "__main__":
  load_data()