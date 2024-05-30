from word_evaluation import get_sample_sentences, run_model
import os
from transformers import AutoConfig, AutoTokenizer,GPTNeoXForCausalLM
import torch
import argparse
import json

MAX_STORED_LINE_COUNT = 10000
# DATA_DIR = "C:/Users/pdder/ThesisData" # Where the models are cached
# CODE_DIR = "C:/Users/pdder/OneDrive/Documents/Master/Artificial Intelligence/Thesis/Code/word-acquisition-language-models"
PYTHIA_CHECKPTS = [0]+[2**i for i in range(10)] + [1000*i for i in range(1,144)] 


def create_parser():
    parser = argparse.ArgumentParser()
    # Directory where Pythia models and tokenizers are cached.
    parser.add_argument('--model_dir', default="")
    # Size of Pythia model to evaluate on, must be one of Pythia's available sizes (14m, 31m, 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b).
    parser.add_argument('--model_size', type=str,  default="160m")
    parser.add_argument('--output_dir', default="./sample_data/eval")
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
    # Token_data if this is already computed beforehand, saves computation
    parser.add_argument('--token_data', default=None)
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
        cache_dir=model_dir+"/pythia-"+size+"-deduped/"+ str(step),
    )
    print(model_dir+"/pythia-"+size+"-deduped/"+ str(step))

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-"+size+"-deduped",
        revision="step"+ str(step),
        cache_dir=model_dir+"/pythia-"+size+"-deduped/"+ str(step),
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
    step = 0
    size = args.model_size
    model_dir = args.model_dir
    model, tokenizer = load_pythia_model(model_dir, size=size, step=step)
    snapshot_dir = os.path.join(args.model_dir, "pythia-"+size+"-deduped",str(step),"models--EleutherAI--pythia-"+size+"-deduped", "snapshots")
    print(os.listdir(os.path.join(args.model_dir, "pythia-"+size+"-deduped", str(step))))
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
    token_data = args.token_data
    if token_data is None:
        token_data = get_sample_sentences(tokenizer, wordbank_file, wordbank_lang, example_file, max_seq_len, min_seq_len, max_samples, bidirectional=bidirectional, inflections=inflections, spm_tokenizer=False)
    else:
        with open(token_data, "r") as f:
            token_data = json.load(f)
     # Prepare for evaluation.
    outdir = args.output_dir
    # File header.
    if args.checkpoints is None or len(args.checkpoints) == 0:
        checkpoints = PYTHIA_CHECKPTS
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
        evaluate_tokens_pythia(model, token_data, tokenizer, outdir,
                        checkpoint, batch_size, min_samples)

def evaluate_tokens_pythia(model, token_data, tokenizer, outdir, checkpoint, batch_size, min_samples):
    summary_file = os.path.join(outdir, "summary.txt")

    token_count = 0
    for token, token_id, sample_sents in token_data:
        token_dir = os.path.join(outdir, token)
        print("\nEvaluation token: {}".format(token))
        token_count += 1
        print("{0} / {1} tokens".format(token_count, len(token_data)))
        print("CHECKPOINT STEP: {}".format(checkpoint))
        num_examples = len(sample_sents)
        print("Num examples: {}".format(num_examples))
        if num_examples < min_samples:
            print("Not enough examples; skipped.")
            continue
        # Get logits with shape: num_examples x vocab_size.
        logits = run_pythia(model, sample_sents, batch_size, tokenizer)
        print("Finished inference.")
        probs = torch.nn.Softmax(dim=-1)(logits)
        # Get median rank of correct token
        rankings = torch.argsort(probs, axis=-1, descending=True)
        ranks = torch.nonzero(rankings == token_id) # Each output row is an index (sentence_i, token_rank).
        ranks = ranks[:, 1] # For each example, only interested in the rank (not the sentence index).
        rank_file = os.path.join(token_dir, "ranks.txt")
        with open(rank_file, "wb") as f:
            for rank in ranks:
                f.write(rank.item()+"\t")
            f.write("\n")
        median_rank = torch.median(ranks).item()
        # Get accuracy
        pred_tokens = torch.argmax(probs, dim=-1)
        correct = pred_tokens == token_id
        accuracy = torch.sum(correct) / correct.size()
        # Get mean/stdev surprisal
        token_probs = probs[:, token_id]
        token_probs += 0.000000001 # Smooth with (1e-9).
        surprisals = -1.0*torch.log2(token_probs)
        surprisals_file = os.path.join(token_dir, "surprisals.txt")
        with open(surprisals_file, "wb") as f:
            for surprisal in surprisals:
                f.write(surprisal.item()+"\t")
            f.write("\n")
        mean_surprisal = torch.mean(surprisals).item()
        std_surprisal = torch.std(surprisals).item()
        # Logging.
        print("Median rank: {}".format(median_rank))
        print("Mean surprisal: {}".format(mean_surprisal))
        print("Stdev surprisal: {}".format(std_surprisal))
        print("Accuracy: {}".format(accuracy))
        with open(summary_file, "wb") as f:
            f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n".format(
            checkpoint, token, median_rank, mean_surprisal, std_surprisal,
            accuracy, num_examples))
    return




def run_pythia(model, sample_sents, batch_size, tokenizer):
    return run_model(model, "gpt2", sample_sents, batch_size, tokenizer)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
