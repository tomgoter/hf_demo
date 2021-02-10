import numpy as np
import pandas as pd
from transformers import TFBertForSequenceClassification, BertTokenizer
from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
from transformers import TFTrainer, TFTrainingArguments
import os
from pprint import pprint
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import layers, initializers
import argparse

print(f'Numpy version: {np.__version__}')
print(f'Pandas version: {pd.__version__}')
print(f'TensorFlow version: {tf.__version__}')

TRAINING_DATA = './ag_news_csv/train.csv'
VAL_DATA = './ag_news_csv/test.csv' 

def gather_data(data):
  df = pd.read_csv(data, header=None)
  df.columns = ['label', 'title', 'desc']
  df.desc = df.desc.replace(r'\\', ' ', regex=True)
    
  # The labels are a column in the data frame - pop them into their own object
  labels = df.label.values
  labels = labels - 1

  # Get the training sentences
  sentences = df.desc.values
    
  return sentences, labels
  
  
def create_dataset(sequences, labels, tokenizer = TOKENIZER):
  input_ids = []
  attention_mask = []
  token_ids = []

  for sent in tqdm(sequences):
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
                                            'attention_mask':attention_mask}, labels))


  return dataset
  
  
def main():
  
  parser = argparse.ArgumentParser(description='Script for running text topic classification with transformers package')
  parser.add_argument('-m', '--model', choices=['bert-base-uncased',
                                                'bert-large-uncased', 
                                                'roberta-base', 
                                                'roberta-large'],
                        help='Class of Model Architecture to use for classification')
  parser.add_argument('-o', '--output', type=str,
                        help='Output File Prefix for model file and dataframe')
  parser.add_argument('-b', '--BATCH_SIZE', default=64, type=int,
                       help='batch size to use per replica')
  
  args = parser.parse_args()
  
  if args.model[:4] == 'bert':
    # Use base Bert tokenizer
    TOKENIZER = BertTokenizer.from_pretrained(args.model)
  else:
    TOKENIZER = RobertaTokenizer.from_pretrained(args.model)
  
  
  train_sentences, train_labels = gather_data(TRAINING_DATA)
  val_sentences, val_labels = gather_data(VAL_DATA)
  
  print(f'Length of Training Set: {len(train_sentences)}')
  print(f'Length of Test Set: {len(val_sentences)}')
  
  training_dataset = create_dataset(train_sentences, train_labels)
  val_dataset = create_dataset(val_sentences, val_labels)
  
  mirrored_strategy = tf.distribute.MirroredStrategy()
  print ('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))
  
  BATCH_SIZE_PER_REPLICA = 64
  GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync

  batched_training_dataset = training_dataset.shuffle(1024).batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
  batched_val_dataset = val_dataset.shuffle(1024).batch(GLOBAL_BATCH_SIZE, drop_remainder=True)

  #dist_train_dataset = mirrored_strategy.experimental_distribute_dataset(batched_training_dataset)
  #dist_val_dataset = mirrored_strategy.experimental_distribute_dataset(batched_val_dataset)
 
  with mirrored_strategy.scope():
    if args.model[:4] == 'bert':
      model = TFBertForSequenceClassification.from_pretrained(args.model, num_labels=4)
    else:
      model = TFRobertaForSequenceClassification.from_pretrained(args.model, num_labels=4)
      
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    METRICS = [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    model.compile(optimizer=optimizer, loss=loss, metrics=METRICS)
    
  model.fit(batched_training_dataset,
            epochs=5,
            validation_data = batched_val_dataset,
           )
  
  
if __name__ == '__main__':
  main()
