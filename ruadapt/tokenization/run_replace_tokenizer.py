import argparse
import json
import os
import codecs
from transformers import AutoTokenizer, AutoConfig
from .replace_tokenizer_batched import reinit_embeddings_with_head_universal_batched

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--new_tokenizer_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--mode', default='mean', choices=['mean', 'wmean', 'random', 'mlp'])
    parser.add_argument('--mult', default=1.0, type=float)
    parser.add_argument('--head_path', default=None, help='Path to trained MLP head (required for mode=mlp)')
    parser.add_argument('--pooling', default='attention', help='Pooling architecture for mlp mode')
    parser.add_argument('--batch_size', default=256, type=int)
    args = parser.parse_args()
    print(args)

    tokenizer_old = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer_new = AutoTokenizer.from_pretrained(args.new_tokenizer_path)
    config_old = AutoConfig.from_pretrained(args.model_name_or_path)
    
    architectures = getattr(config_old, "architectures", [])
    if architectures and "Qwen3_5ForConditionalGeneration" in architectures:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
        ModelClass = Qwen3_5ForConditionalGeneration
    else:
        from transformers import AutoModelForCausalLM
        ModelClass = AutoModelForCausalLM

    print("Loading base model...")
    model = ModelClass.from_pretrained(
        args.model_name_or_path,
        config=config_old,
        device_map='cuda:0',
        torch_dtype=config_old.torch_dtype,
        trust_remote_code=True
    )
    
    current_len = len(tokenizer_new)
    target_multiple = 256
    pad_needed = (target_multiple - (current_len % target_multiple)) % target_multiple
    
    if pad_needed > 0:
        print(f"Padding vocabulary from {current_len} to {current_len + pad_needed}")
        new_tokens = [f"<|free_token{i+1}|>" for i in range(pad_needed)]
        tokenizer_new.add_tokens(new_tokens)
        # Resizing of the model is deferred to Step 3 inside the batched script

    reinit_logs = reinit_embeddings_with_head_universal_batched(
        model, tokenizer_old, tokenizer_new, 
        mode=args.mode, 
        lm_head_init='hm', 
        add_special_tokens_src=True,
        mult=args.mult,
        head_path=args.head_path,
        pooling=args.pooling,
        batch_size=32#args.batch_size
    )

    config_new = AutoConfig.from_pretrained(args.new_tokenizer_path)
    config_new.vocab_size = len(tokenizer_new)
    if hasattr(config_new, 'text_config'):
        config_new.text_config.vocab_size = len(tokenizer_new)
    model.config = config_new

    from transformers import GenerationConfig
    model.generation_config = GenerationConfig.from_model_config(config_new)
    for attr_name in dir(tokenizer_new):
        if attr_name.endswith("_token_id") and getattr(tokenizer_new, attr_name) is not None:
            setattr(model.generation_config, attr_name, getattr(tokenizer_new, attr_name))

    print(f"Saving model and tokenizer to {args.output_path}...")
    model.save_pretrained(args.output_path)
    model.generation_config.save_pretrained(args.output_path)
    tokenizer_new.save_pretrained(args.output_path)

    with codecs.open(os.path.join(args.output_path, 'reinit_tokenizer_logs.json'), 'w', 'utf-8') as file:
        json.dump(reinit_logs, file, ensure_ascii=False, indent=4)
        
    print("Done.")
