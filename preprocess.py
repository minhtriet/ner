from transformers import AutoTokenizer
import sys

filename = sys.argv[1]

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
subword_len_counter = 0
MAX_LENGTH = 128 - tokenizer.num_special_tokens_to_add()

with open(filename, "rt") as f_p:
    for line in f_p:
        line = line.rstrip()

        if not line:
            print(line)
            subword_len_counter = 0
            continue

        token = line.split()[0]

        current_subwords_len = len(tokenizer.tokenize(token))

        # Token contains strange control characters like \x96 or \x95
        # Just filter out the complete line
        if current_subwords_len == 0:
            continue

        if (subword_len_counter + current_subwords_len) > MAX_LENGTH:
            print("")
            print(line)
            subword_len_counter = current_subwords_len
            continue

        subword_len_counter += current_subwords_len

        print(line)

