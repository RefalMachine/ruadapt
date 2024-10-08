
from itertools import chain
import difflib

def get_tokenizer_properties(tokenizer):
    leading_space = False
    space = None
    char = '1'
    tokens = tokenizer(char, add_special_tokens=False)['input_ids']
    if len(tokens) > 1:
        space = tokenizer.convert_ids_to_tokens([tokens[0]])[0]
        leading_space = True
    else:
        token_str = tokenizer.convert_ids_to_tokens(tokens)[0]
        if len(token_str) != 1:
            space = token_str[0]
            leading_space = True

    space_token = tokenizer('1 ', add_special_tokens=False)['input_ids']
    if leading_space:
        assert len(space_token) == 3
        space_token = space_token[2]
        assert tokenizer.convert_ids_to_tokens([space_token])[0] == space
    else:
        assert len(space_token) == 2
        space_token = space_token[1]
        if space is None:
            space = tokenizer.convert_ids_to_tokens([space_token])[0]
        assert tokenizer.convert_ids_to_tokens([space_token])[0] == space

    return {'force_leading_space': leading_space, 'space': space}

def merge(tensors):
    keys = tensors[0].keys()
    return {key: [l for l in chain(*[t[key] for t in tensors])] for key in keys}

def shrink_first_fake_space(tensors):
    return {key: tensors[key][1:] for key in tensors}

def shrink_first_fake_space_plus_dummy(tensors):
    return {key: tensors[key][2:] for key in tensors}

def custom_split(text):
    splitted = []
    pos = 0
    prev_pos = 0
    pos = text.find('\n', prev_pos)
    while pos >= 0:
        s = text[pos]
        while s == '\n':
            pos += 1
            s = text[pos]
        splitted.append(text[prev_pos:pos])
        prev_pos = pos
        pos = text.find('\n', prev_pos)
    splitted.append(text[prev_pos:])
    return splitted

def custom_tokenize(text, tokenizer, tokenizer_properties, enable_asserts=True):
    bos_token = tokenizer.bos_token if tokenizer.bos_token is not None else ''
    eos_token = tokenizer.eos_token if tokenizer.eos_token is not None else ''

    text_with_special_tokens = bos_token + text.strip() + eos_token
    paragraphs = custom_split(text_with_special_tokens)
    
    assert text_with_special_tokens == ''.join(paragraphs)

    tensors_para_first = tokenizer(paragraphs[0], add_special_tokens=False)
    if tokenizer_properties['force_leading_space']:
        tensors_para_rest = [shrink_first_fake_space_plus_dummy(tokenizer('\n' + par, add_special_tokens=False)) for par in paragraphs[1:]]
    else:
        tensors_para_rest = [tokenizer(par, add_special_tokens=False) for par in paragraphs[1:]]

    output = merge([tensors_para_first] + tensors_para_rest)
    if enable_asserts:
        decoded_output = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(output['input_ids']))
        if tokenizer_properties['force_leading_space'] and len(bos_token) > 0:
            decoded_output = decoded_output.replace(bos_token + ' ', bos_token)
        assert decoded_output.lstrip() == text_with_special_tokens.lstrip()
    return output


def group_texts(examples, tokenizer, tokenizer_properties, block_size, ntokens_ids=[]):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    start_idx = 0
    end_idx = start_idx + block_size
    segments = []
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id

    para_sep_token_id = tokenizer('\n', add_special_tokens=False)['input_ids'][int(tokenizer_properties['force_leading_space'])]
    para_sep_token_ids = set([para_sep_token_id, eos_token_id])
    segment_start_tokens = set([bos_token_id, para_sep_token_id, eos_token_id])

    if len(ntokens_ids) > 0:
        para_sep_token_ids.update(ntokens_ids)
        segment_start_tokens.update(ntokens_ids)

    while end_idx < total_length:
        while start_idx < total_length and concatenated_examples['input_ids'][start_idx] not in segment_start_tokens:
            start_idx += 1

        if start_idx == total_length:
            break

        while start_idx < total_length and concatenated_examples['input_ids'][start_idx] in para_sep_token_ids:
            start_idx += 1

        if start_idx == total_length:
            break

        end_idx = start_idx + block_size
        if end_idx < total_length:
            segments.append([start_idx, end_idx])

        start_idx = end_idx

    result = {k: [] for k in concatenated_examples}
    for segment in segments:
        for k in result:
            result[k].append(concatenated_examples[k][segment[0]: segment[1]])

    result["labels"] = result["input_ids"].copy()
    return result