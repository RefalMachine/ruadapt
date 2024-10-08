from datasets import load_dataset
from argparse import ArgumentParser
from ruadapt.pretraining.utils import custom_tokenize, get_tokenizer_properties, group_texts
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import codecs
import json

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_file')
    parser.add_argument('--tokenizer')
    args = parser.parse_args()

    data_files = {}
    dataset_args = {}
    data_files["train"] = args.train_file

    extension = (
        args.train_file.split(".")[-1]
    )

    print(extension)
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        **dataset_args,
    )
    print(raw_datasets)

    #raw_datasets['train'] = raw_datasets['train'].select(np.random.choice(range(len(raw_datasets['train'])), 100))
    raw_datasets['train'] = raw_datasets['train'].select(range(1000))
    #for i, text in enumerate(raw_datasets['train']['text']):
    #    if '\n\n' in text.strip():
    #        print(text)
    #        break
    #print(raw_datasets)


    column_names = list(raw_datasets["train"].features)
    text_column_name = 'text'
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer_prop = get_tokenizer_properties(tokenizer)

    print(tokenizer)
    print(tokenizer_prop)
    
    def tokenize_function(examples):
        return custom_tokenize(examples[text_column_name], tokenizer, tokenizer_prop, enable_asserts=False)
    
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=False,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    print(tokenized_datasets)
    #print(tokenized_datasets['train']['attention_mask'])
    tokens_total = sum([len(s) for s in tokenized_datasets['train']['attention_mask']])
    symbols_total = sum([len(t) for t in raw_datasets['train']['text']])
    print(symbols_total / tokens_total)
    print(sum([len(s) for s in tokenized_datasets['train']['attention_mask']]))
    print(sum([sum(s) for s in tokenized_datasets['train']['attention_mask']]))

    block_size = 4096
    ntokens = []
    ntokens_ids = set()
    vocab = tokenizer.vocab
    for t in tqdm(vocab):
        token = tokenizer.convert_tokens_to_string([t])
        if token.startswith('\n') or token.endswith('\n'):
            ntokens.append(token)
            ntokens_ids.add(vocab[t])

    print(len(ntokens_ids))
    print(ntokens)
    
    def group_texts_function(examples):
        return group_texts(examples, tokenizer, tokenizer_prop, block_size, ntokens_ids=ntokens_ids)
    
    lm_datasets = tokenized_datasets.map(
        group_texts_function,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    print(lm_datasets)
    print(sum([len(s) for s in lm_datasets['train']['attention_mask']]))
    print(sum([sum(s) for s in lm_datasets['train']['attention_mask']]))

    data = []
    for i, tokens in enumerate(lm_datasets['train']['input_ids']):
        text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tokens))
        data.append({'idx': i, 'text': text})

    with codecs.open('test_group.json', 'w', 'utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


