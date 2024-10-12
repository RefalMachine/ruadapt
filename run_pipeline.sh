python run_pipeline.py \
--ruadapt_base_model_name_or_path RefalMachine/ruadapt_qwen2.5_3B_ext_cl100k_bpe_32000_full_lr5e4_2k_bs256 \
--raw_base_model_name_or_path Qwen/Qwen2.5-3B \
--instruct_model_name_or_path Qwen/Qwen2.5-3B-Instruct \
--custom_chat_template_path ruadapt/ushanka/custom_chat_templates/qwen_2.5_instruct_no_system.json \
--output_dir /workdir/data/models/qwen/ruadapt_qwen2.5_3B_ext_b32 \
--alpaca_eval_questions_path ../saiga/llmarena_questions.json \
--pipeline_config_path pipeline_configs/qwen_pipeline_config_test.json \
--custom_bos_token "<|endoftext|>" \
--custom_pad_token "<|endoftext|>"