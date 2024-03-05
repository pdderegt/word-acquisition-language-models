from word_evaluation import get_sample_sentences, evaluate_tokens
import os
from transformers import AutoConfig, AutoTokenizer,GPTNeoXForCausalLM
from lm_utils import get_dataset
import codecs
import torch
import argparse

MAX_STORED_LINE_COUNT = 10000
DATA_DIR = "C:/Users/pdder/ThesisData" # Where the models are cached
CODE_DIR = "C:/Users/pdder/OneDrive/Documents/Master/Artificial Intelligence/Thesis/Code/word-acquisition-language-models"
PYTHIA_CHECKPTS = [0]+[2**i for i in range(10)] + [1000*i for i in range(1,144)] 


def create_parser():
    parser = argparse.ArgumentParser()
    # Directory where Pythia models and tokenizers are cached.
    parser.add_argument('--model_dir', default="")
    # Size of Pythia model to evaluate on, must be one of Pythia's available sizes (14m, 31m, 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b).
    parser.add_argument('--model_size', type=str,  default="70m")
    parser.add_argument('--output_file', default="./sample_data/pythia_surprisals.txt")
    # If empty, uses all Pythia checkpoints.
    # If specified, must be subset of Pythia checkpoints available.
    parser.add_argument('--checkpoints', type=int, nargs='*')
    parser.add_argument('--batch_size', type=int, default=32)
    # Child AoA file, used to identify the desired wordbank words.
    parser.add_argument('--wordbank_file', default="./r_code/tacl_data/child_data/child_aoa.tsv")
    parser.add_argument('--wordbank_lang', default="English (American)")
    # Examples should already be tokenized. Each line should be a
    # space-separated list of integer token ids.
    parser.add_argument('--examples_file', default="./sample_data/eval_tokenized.txt")
    # The minimum number of sample sentences to evaluate a token.
    parser.add_argument('--min_samples', type=int, default=8)
    parser.add_argument('--max_samples', type=int, default=512)
    # The minimum sequence length to evaluate a token in a sentence.
    # For unidirectional models, this only counts context before the target token.
    parser.add_argument('--min_seq_len', type=int, default=8)
    # Load token data (sample sentences for each token) from file.
    # If file does not exist, saves the token data to this file.
    parser.add_argument('--save_samples', default="")
    # Whether to include token inflections (all, only, or none).
    parser.add_argument('--inflections', default="none")
    return parser


def load_pythia_model(model_dir, step=143000, size="1b"):
    # Loads single Pythia model of given size and at given training step, cached at given model directory
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-"+size+"-deduped",
        revision="step"+ str(step),
        cache_dir=model_dir+"./pythia-"+size+"-deduped/"+ str(step),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-"+size+"-deduped",
        revision="step"+ str(step),
        cache_dir=model_dir+"./pythia-"+size+"-deduped/"+ str(step),
        padding=True
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def main(args):
    # Get config file of Pythia tokenizer, located in snapshot folder in cached model
    # We arbitrarily pick the tokenizer of step 0, should be same for all training steps
    size = args.model_size
    model_dir = args.model_dir
    snapshot_dir = os.path.join(args.model_dir, "pythia-"+size+"-deduped","0","models--EleutherAI--pythia-"+size+"-deduped", "snapshots")
    config_dir =  os.path.join(snapshot_dir, next(os.walk(snapshot_dir))[1][0])
    config_path = os.path.join(config_dir, "config.json")
    config = AutoConfig.from_pretrained(config_path)
    wordbank_file = args.wordbank_file
    wordbank_lang = args.wordbank_lang
    example_file = args.examples_file
    max_seq_len = config.max_position_embeddings
    min_seq_len = args.min_seq_len
    max_samples = args.max_samples
    bidirectional = False
    inflections = args.inflections
    model, tokenizer = load_pythia_model(model_dir, size=size)
    token_data = get_sample_sentences(tokenizer, wordbank_file, wordbank_lang, example_file, max_seq_len, min_seq_len, max_samples, bidirectional=bidirectional, inflections=inflections, spm_tokenizer=False)
     # Prepare for evaluation.
    output_file = args.output_file
    outfile = codecs.open(output_file, 'w', encoding='utf-8')
    # File header.
    if args.checkpoints is None or len(args.checkpoints) == 0:
        checkpoints = PYTHIA_CHECKPTS[:20]
    else:
        checkpoints = args.checkpoints
    checkpoints = list(checkpoints)
    checkpoints.sort()
    batch_size = args.batch_size
    min_samples = args.min_samples
    # Run evaluation.
    for checkpoint in checkpoints:
        print("CHECKPOINT STEPS: {}".format(checkpoint))
        model, tokenizer = load_pythia_model(model_dir, step=checkpoint, size=size)
        evaluate_tokens(model, "gpt2", token_data, tokenizer, outfile,
                        checkpoint, batch_size, min_samples)
    outfile.close()
   

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
