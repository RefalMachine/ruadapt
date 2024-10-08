from argparse import ArgumentParser
import codecs
import numpy as np
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

byte_encoder = bytes_to_unicode()
def token_bytes_to_string(b):
    return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

def convert_to_llama(token):
    return token_bytes_to_string(token.replace('▁', ' ').encode())

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tokenizer_vocab_path')
    parser.add_argument('--output_path')
    args = parser.parse_args()

    with codecs.open(args.tokenizer_vocab_path, 'r', 'utf-8') as file:
        data = file.read().strip().split('\n')

    data = [d.split() for d in data]
    data = [[d[0].replace('▁', ' '), float(d[1])] for d in data]
    data = [d for d in data if d[1] < 0]

    data = ['\t'.join([d[0], str(float(np.exp(d[1])))]) for d in data]
    with codecs.open(args.output_path, 'w', 'utf-8') as file:
        file.write('\n'.join(data))
