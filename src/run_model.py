# data processing tools
import argparse
import string, os, sys 
import pandas as pd
import numpy as np
np.random.seed(42)

# keras module
import tensorflow as tf
tf.random.set_seed(42)

from joblib import dump
import pickle

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# predefined functions 
sys.path.append(os.path.join("utils"))
import predef_func as pdfunc


# parser
def input_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--word", type=str, default= "Trump", help= "Specify word for text generation.") 
    parser.add_argument("--n_next_words", type=int, default= 8, help= "Specify number of next words following the chosen word.")

    args = parser.parse_args()
    
    return args #returning arguments


# importing data
def load_the_data():
    # loading tokenizer
    with open('out/tokenizer.pickle', 'rb') as t:
        tokenizer = pickle.load(t)

    with open('out/max_sequence_len.txt') as f:
        max_sequence_len = f.read()
    
    # make max_sequence_len integer    
    max_sequence_len = int(max_sequence_len)

    return tokenizer, max_sequence_len

# run model
def run_model(word, n_next_words, max_sequence_len, tokenizer):
    file_path = os.path.join(os.getcwd(),"models")
    loaded_model = tf.keras.models.load_model(file_path) 
    print(pdfunc.generate_text(word, n_next_words, loaded_model, max_sequence_len, tokenizer))


# main function
def main():
    args = input_parse()
    tokenizer, max_sequence_len = load_the_data()
    print("Generating text from the word: "+ args.word)
    run_model(args.word, args.n_next_words, max_sequence_len, tokenizer)

if __name__ == '__main__':
    main() 
