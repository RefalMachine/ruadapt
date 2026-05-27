pip install blobfile
python extend_or_replace.py \
--op extend \
--src_model_path Qwen/Qwen2.5-7B \
--output_path /workdir/data/models/ruadapt/qwen/ruadapt_qwen2.5_7B_ext_u48_mean_init \
--extend_hf_tokenizer_path ruadapt/tokenization/hf_tokenizers/darulm_20_05_24_part1-2_48000_unigram_hf \
--extend_hf_tokenizer_type unigram \
--only_ru \
--extend_tiktoken_tokenizer_path ruadapt/tokenization/cl100k_base.tiktoken \
--filter_numbers