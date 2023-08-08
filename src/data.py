# data processing tools
import string, os, sys
import pandas as pd
import numpy as np
np.random.seed(42)
import random

# keras module
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.preprocessing.text import Tokenizer

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# predefined functions from utils
sys.path.append(os.path.join("utils"))
import predef_func as pdfunc


# for scripting
import argparse

# parser
def input_parse():
    parser = argparse.ArgumentParser()

    #add arguments for data.py
    parser.add_argument("--member_folder", type=str, default= "431868", help= "Specify your specific member folder where the data is located.") 
    parser.add_argument("--sub_folder", type=str, default= "news_data", help= "Specify the subfolder of the member folder where the .csv files are located.") 
    parser.add_argument("--word_in_filename", type=str, default= "Comments", help= "Specify a given word in filenames, to loop over files of same type.") 
    parser.add_argument("--column_name", type=str, default= "commentBody", help= "Specify name of column that contains the necessary text for modelling.") 
    parser.add_argument("--n_comments", type=int, default= 2000, help= "Specify the amount of comments that are randomly sampled.") 


    args = parser.parse_args()
    
    return args #returning arguments



# preprocessing til data

def data_load(member_folder, sub_folder, word_in_filename, column_name):
    data_dir = os.path.join("..", "..", "..", member_folder, sub_folder)  

    all_comments = []

    for file in os.listdir(data_dir):
        if word_in_filename in file:
            comments_df = pd.read_csv(data_dir +"/"+ file)
            all_comments.extend(list(comments_df[column_name].values))
    return all_comments


# cleaning the data
def data_clean(all_comments, n_comments):
    
    sample_comments = all_comments[:n_comments]
    print("Using " + str(n_comments) + " comments.")
    sample_comments = [c for c in sample_comments if c != "Unknown"]
    corpus = [pdfunc.clean_text(x) for x in sample_comments]
    return corpus


# data tokenizing
def data_tokenize(corpus):
    tokenizer = Tokenizer() 
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    return tokenizer, total_words


## Get sequence of tokens and pad the sequences
def data_seq(tokenizer, total_words, corpus):
    inp_sequences = pdfunc.get_sequence_of_tokens(tokenizer, corpus)
    predictors, label, max_sequence_len = pdfunc.generate_padded_sequences(inp_sequences, total_words) 
    return predictors, label, max_sequence_len, total_words, tokenizer 


# main function
def main():
    args = input_parse()
    print("Initializing data preprocessing..")
    all_comments = data_load(args.member_folder, args.sub_folder, args.word_in_filename, args.column_name)
    corpus = data_clean(all_comments, args.n_comments)
    tokenizer, total_words = data_tokenize(corpus)
    predictors, label, max_sequence_len, total_words, tokenizer = data_seq(tokenizer, total_words, corpus)
    print("done")

if __name__ == '__main__':
    main()
