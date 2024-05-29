import datasets
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from tokenize_dataset import tokenize_file
import argparse
import math
import codecs
import string
import numpy as np
import json
import random
import os

def create_parser():
    parser = argparse.ArgumentParser()
    # Directory where Pythia models and tokenizers are cached.
    parser.add_argument('--model_dir', default="")
    # Directory where the output files (tokenized data & json file) are created
    parser.add_argument('--output_dir')
    # JSON file where token_data list is stored: [token, token_id, list with example sentences to evaluate on].
    parser.add_argument('--output_file', default="token_data.json")
    # Child AoA file, used to identify the desired wordbank words.
    parser.add_argument('--wordbank_file', default="./r_code/tacl_data/child_data/child_aoa.tsv")
    parser.add_argument('--wordbank_lang', default="English (American)")
    # The number of sample sentences to evaluate a token.
    parser.add_argument('--nr_samples', type=int, default=256)
    # For unidirectional models, this only counts context before the target token.
    parser.add_argument('--min_seq_len', type=int, default=8)
    
    # Load token data (sample sentences for each token) from file.
    # If file does not exist, saves the token data to this file.
    parser.add_argument('--save_samples', default="")
    # Whether to include token inflections (all, only, or none).
    parser.add_argument('--inflections', default="none")
    return parser

def main(args):
    model_dir = args.model_dir
    step=143000
    size="70m" 
    tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-"+size+"-deduped",
            revision="step"+ str(step),
            cache_dir=model_dir+"./pythia-"+size+"-deduped/"+ str(step),
            )
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-"+size+"-deduped",
        revision="step"+ str(step),
        cache_dir=model_dir+"./pythia-"+size+"-deduped/"+ str(step),
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    if tokenizer.cls_token is None:
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    print(tokenizer.special_tokens_map)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)
    print("Loading datasets...")
    datasets.utils.logging.set_verbosity_warning() 
    bookcorpus = datasets.load_dataset("bookcorpus")
    wikitext = datasets.load_dataset("wikitext", 'wikitext-103-v1')
    print("Datasets loaded")

    bookcorpus_test = bookcorpus['train'].train_test_split(0.2)['test']
    wikitext_test = wikitext['train'].train_test_split(0.2)['test']
    with open('eval_text.txt', 'a') as f:
        for ex in bookcorpus_test:
            f.write(ex['text']+" \n")

    with open('eval_text.txt', "a", encoding="utf-8") as f:
        for ex in wikitext_test:
            f.write(ex['text']+" \n")
    
    output_dir = args.output_dir
    text_file = 'eval_text.txt'
    outfile = os.path.join(output_dir,'eval_tokenized.txt')
    max_examples = math.inf
    max_segments = math.inf
    min_seq_len = args.min_seq_len
    max_seq_len = 128
    nr_samples = args.nr_samples
    if os.path.isfile(outfile):
        os.remove(outfile)
    tokenize_file(text_file, outfile, tokenizer, max_examples, max_segments, max_seq_len)

   

    token_data = []
    wordbank_file = args.wordbank_file
    wordbank_lang = args.wordbank_lang
    wordbank_file = codecs.open(wordbank_file, 'rb', encoding='utf-8')
    wordbank_tokens = set()
    for line in wordbank_file:
        split_line = line.strip().split('\t')
        if split_line[5] == wordbank_lang:
            wordbank_tokens.add(split_line[-1])
    wordbank_file.close()
    for token in wordbank_tokens:
        tokens = tokenizer.encode(" "+ token)
        if len(tokens)==1:
            token_id = tokens[0]
        else:
            token_id = tokenizer.unk_token_id
        if token_id != tokenizer.unk_token_id:
            token_data.append(tuple([token, token_id, []]))

    infile = codecs.open(outfile, 'rb')
    for line_count, line in enumerate(infile):
        example_string = line.strip()
        example = [int(token_id) for token_id in example_string.split()]
        if len(example) < min_seq_len:
            continue
        if len(example) > max_seq_len:
            example = example[:max_seq_len]
        for token, token_id, sample_sents in token_data:
            if len(sample_sents) >= nr_samples:
                continue
            token_indices = [index for index, curr_id in enumerate(example) if curr_id == token_id]
            are_sep_words = []
            for token_index in token_indices:
                if token_index==len(example)-1:
                    next_token=""
                else:
                    next_token = tokenizer.decode(example[token_index+1])
                is_sep_word = " " in next_token or "\u2581" in next_token or next_token in string.punctuation
                are_sep_words.append(is_sep_word)
            token_indices = np.array(token_indices)[are_sep_words]
            token_indices = [index for index in token_indices if index >= min_seq_len-1]
            if len(token_indices) > 0:
                new_example = example.copy()
                mask_idx = random.choice(token_indices)
                new_example[mask_idx] = tokenizer.mask_token_id
                sample_sents.append(new_example)
        if all(len(sample_sents)==nr_samples for _,_, sample_sents in token_data):
            print("Found enough examples for all tokens")
            break
    for token, token_id, sample_sents in token_data:
        print("For token", token, "there were", len(sample_sents), "examples found.")
    with open(os.path.join(output_dir,'token_data.json'), 'w') as f:
        json.dump(token_data, f)

                

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)