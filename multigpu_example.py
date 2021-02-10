import numpy as np
import pandas as pd
from transformers import TFBertModel, TFBertForSequenceClassification, BertTokenizer
from transformers import TFTrainer, TFTrainingArguments
import os
from pprint import pprint
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import layers, initializers

print(f'Numpy version: {np.__version__}')
print(f'Pandas version: {pd.__version__}')
print(f'TensorFlow version: {tf.__version__}')

TRAINING_DATA = './ag_news_csv/train.csv'
VAL_DATA = './ag_news_csv/test.csv' 

# Use base Bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def gather_data(data):
    df = pd.read_csv(data, header=None)
    df.columns = ['label', 'title', 'desc']
    df.desc = df.desc.replace(r'\\', ' ', regex=True)
    
    # The labels are a column in the data frame - pop them into their own object
    labels = df.label.values
    labels = labels -1

    # Get the training sentences
    sentences = df.desc.values
    
    return sentences, labels
  
  
def create_dataset(sequences, labels, batch_size = 256):
    input_ids = []
    attention_mask = []
    token_ids = []
    
    num_examples = (len(sequences) // batch_size) * batch_size
    
    for sent in tqdm(sequences[:num_examples]):
        encoded_dict = tokenizer.encode_plus(sent,
                     add_special_tokens = True,
                     padding = 'max_length',
                     max_length = 128,
                     truncation = True,
                     return_attention_mask = True,
                     return_token_type_ids = True,
                     return_tensors = 'tf') 
        input_ids.append(tf.reshape(encoded_dict['input_ids'],[-1]))
        #token_ids.append(tf.reshape(encoded_dict['token_type_ids'],[-1]))
        attention_mask.append(tf.reshape(encoded_dict['attention_mask'],[-1]))
    
    dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids,
                                              #'token_type_ids':token_ids,
                                              'attention_mask':attention_mask}, labels[:num_examples]))
    
    
    return dataset
  
  
def main():
  train_sentences, train_labels = gather_data(TRAINING_DATA)
  val_sentences, val_labels = gather_data(VAL_DATA)
  
  training_dataset = create_dataset(train_sentences, train_labels)
  test_dataset = create_dataset(test_sentences, test_labels)
  
  
if __name__ == '__main__':
  main()
