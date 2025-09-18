pip install blobfile
python extend_or_replace.py \
--op extend \
--src_model_path Qwen/Qwen3-8B-Base \
--output_path /workdir/data/models/ruadapt/qwen/ruadapt_qwen3_8B_ext_u48_wmean_1.0_init_290425 \
--extend_hf_tokenizer_path ruadapt/tokenization/hf_tokenizers/darulm_20_05_24_part1-2_48000_unigram_hf \
--extend_hf_tokenizer_type unigram \
--only_ru \
--extend_tiktoken_tokenizer_path ruadapt/tokenization/cl100k_base.tiktoken \
--filter_numbers \
--custom_tokens_path ruadapt/tokenization/custom_tokens.json \
--init_mode wmean \
--mult 1.0