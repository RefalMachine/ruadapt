from ruadapt.instruct_tuning.utils import read_jsonl, write_jsonl
import numpy as np

sft_data = read_jsonl('sft_d14_train.jsonl')
wiki_data = read_jsonl('../saiga/wiki_copy_task.jsonl')

print(len(sft_data), len(wiki_data))

wiki_data_short = np.random.choice(wiki_data, 1000, replace=False).tolist()
print(len(wiki_data_short))

full_data = sft_data + wiki_data_short

np.random.shuffle(full_data)
print(len(full_data))

write_jsonl(full_data, 'sft_d14_copy_wiki_train.jsonl')