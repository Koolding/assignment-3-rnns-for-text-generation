# Import packages
# Data preprocessing
import string, os 
import pandas as pd
import numpy as np
np.random.seed(42)

# keras module for building LSTM 
import tensorflow as tf
tf.random.set_seed(42)
import tensorflow.keras.utils as ku 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


#standard coding practice
def clean_text(txt): #for each text
    txt = "".join(v for v in txt if v not in string.punctuation).lower() #return vocab if it is not a part string.punctuation (str.punc== alle dumme tegn), essentially slet alle dumme tegn og lowercase
    txt = txt.encode("utf8").decode("ascii",'ignore') #make encoding  utf8 :))
    return txt  # NOTE this will return nonsense headings at times, since some headlines are connected to questions

def get_sequence_of_tokens(tokenizer, corpus):
    ## convert data to sequence of tokens 
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences

def generate_padded_sequences(input_sequences, total_words):
    # get the length of the longest sequence
    max_sequence_len = max([len(x) for x in input_sequences])
    # make every sequence the length of the longest one
    input_sequences = np.array(pad_sequences(input_sequences, 
                                            maxlen=max_sequence_len, 
                                            padding='pre'))

    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, 
                            num_classes=total_words)
    return predictors, label, max_sequence_len

def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential() #sequential model, den tager ord for ord 
    
    # Add Input Embedding Layer
    model.add(Embedding(total_words,  # creating embedding about each token, learned while training  
                        10,  #small embedding, every word is represented by a 10 dimensional vector (hvilke ord ligger tættest på et givent ord i modellen)
                        input_length=input_len))
    
    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(100)) #long short term model
    model.add(Dropout(0.1)) #during learning from the data and every iteration, remove 10% of the weights (90% of weights remains)  ### this is a finetuning parameter!!!!!!
    
    # Add Output Layer
    model.add(Dense(total_words, #Dense layer =  output layer
                    activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                    optimizer='adam')
    
    return model

def generate_text(seed_text, next_words, model, max_sequence_len, tokenizer):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0] #get vocab
        token_list = pad_sequences([token_list], 
                                    maxlen=max_sequence_len-1, 
                                    padding='pre') #pad them = overcome fixed dimensionality
        predicted = np.argmax(model.predict(token_list),
                                            axis=1) #
        
        output_word = "" #appending stuff and printing it to look nice
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title() 



if __name__=="__main__":
    pass