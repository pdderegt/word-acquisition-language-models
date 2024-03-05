import argparse
from tokenize_dataset import tokenize_file
from transformers import  AutoTokenizer, GPTNeoXForCausalLM
import math


# To process lines in batches.
MAX_STORED_LINE_COUNT = 10000


def create_parser():
    parser = argparse.ArgumentParser()
    # The path to the directory where the tokenizer and models are cached
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True)
    # Maximum number of examples.
    parser.add_argument('--max_examples', type=int, default=-1)
    # Maximum number of segments (input lines) per example.
    # I.e. how many lines to concatenate in each example.
    parser.add_argument('--max_segments', type=int, default=-1)
    # Maximum number of tokens per example.
    # E.g. BERT has maximum sequence length 512.
    # Models will automatically truncate long examples, so it is better to
    # be slightly too long.
    # Examples will be unpadded.
    parser.add_argument('--max_seq_len', type=int, default=512)
    return parser

def tokenize(args):
    model_dir = args.model_dir
    step=143000
    size="70m" 
    # Arbitrary, as tokenizer is the same for any size and checkpoint
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
    max_examples = math.inf if args.max_examples == -1 else args.max_examples
    max_segments = math.inf if args.max_segments == -1 else args.max_segments
    max_seq_len = 999999999 if args.max_seq_len == -1 else args.max_seq_len
    infile = args.input_file
    outfile = args.output_file
    tokenize_file(infile, outfile, tokenizer, max_examples, max_segments, max_seq_len)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    tokenize(args)