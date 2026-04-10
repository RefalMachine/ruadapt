import argparse
import os
import time
from evaluate_tokenizer import evaluate_on_file
from evaluate_ppl import evaluate_ppl
from evaluate_ppl import evaluate_ppl_core
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

def get_model_info(model_path, tokenizer):
    info = {}
    
    # Размеры словаря
    vocab = tokenizer.get_vocab()
    info["vocab_size"] = len(vocab)
    info["added_vocab_size"] = len(tokenizer.added_tokens_encoder)
    
    # Спецтокены
    spec_tokens = set(tokenizer.all_special_tokens)
    info["num_special_tokens"] = len(spec_tokens)
    
    # Проверка диапазонов добавленных токенов и спецтокенов
    max_id = max(vocab.values())
    info["max_token_id"] = max_id
    
    added_encoder = tokenizer.added_tokens_encoder
    if added_encoder:
        added_ids = sorted(added_encoder.values())
        info["added_tokens_range"] = f"[{added_ids[0]} ... {added_ids[-1]}]"
    else:
        info["added_tokens_range"] = "None"
        
    spec_ids = sorted([tokenizer.convert_tokens_to_ids(t) for t in spec_tokens if tokenizer.convert_tokens_to_ids(t) is not None])
    if spec_ids:
        # Показываем самые маленькие и самые большие ID спецтокенов для понимания фрагментации
        info["special_tokens_ids"] = f"min:{spec_ids[0]}, max:{spec_ids[-1]}"
        if spec_ids[0] > (max_id - len(vocab)*0.1):
            info["special_tokens_position"] = "End"
        elif spec_ids[-1] < len(vocab) - len(added_encoder):
            info["special_tokens_position"] = "Middle/Embedded"
        else:
            info["special_tokens_position"] = "Fragmented (Both Middle & End)"
    else:
        info["special_tokens_ids"] = "None"
        info["special_tokens_position"] = "None"

    # Архитектура модели
    try:
        config = AutoConfig.from_pretrained(model_path)
        info["hidden_size"] = getattr(config, "hidden_size", "Unknown")
        info["tie_word_embeddings"] = getattr(config, "tie_word_embeddings", False)
        
        # Динамический выбор класса архитектуры
        architectures = getattr(config, "architectures", [])
        if architectures and "Qwen3_5ForConditionalGeneration" in architectures:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
            ModelClass = Qwen3_5ForConditionalGeneration
        else:
            ModelClass = AutoModelForCausalLM
            
        try:
            model = ModelClass.from_pretrained(model_path, device_map="meta", trust_remote_code=True)
            
            # Universal layer extractor
            if hasattr(model, 'get_input_embeddings'):
                embed_weight = model.get_input_embeddings().weight
            elif hasattr(model, 'model') and hasattr(model.model, 'language_model'):
                embed_weight = model.model.language_model.embed_tokens.weight
            elif hasattr(model, 'language_model'):
                submodel = model.language_model.model if hasattr(model.language_model, 'model') else model.language_model
                embed_weight = submodel.embed_tokens.weight
            else:
                embed_weight = model.model.embed_tokens.weight
                
            info["embed_matrix_shape"] = list(embed_weight.shape)
            
            if hasattr(model, 'get_output_embeddings') and model.get_output_embeddings() is not None:
                info["lm_head_matrix_shape"] = list(model.get_output_embeddings().weight.shape)
            elif hasattr(model, 'lm_head') and model.lm_head is not None:
                info["lm_head_matrix_shape"] = list(model.lm_head.weight.shape)
            elif hasattr(model, 'language_model') and hasattr(model.language_model, 'lm_head') and model.language_model.lm_head is not None:
                info["lm_head_matrix_shape"] = list(model.language_model.lm_head.weight.shape)
            else:
                info["lm_head_matrix_shape"] = "Tied to embeddings"
        except Exception as e:
            info["embed_matrix_shape"] = f"[{getattr(config, 'vocab_size', 'Unknown')} x {info['hidden_size']}] (config estimate)"
            info["lm_head_matrix_shape"] = f"Error: {str(e)}"
    except Exception as e:
        info["error"] = str(e)
        
    return info

def main():
    # Определяем путь к данным относительно скрипта: ruadapt/tokenization/evaluation/data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, "data")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to the model (and tokenizer)")
    parser.add_argument("--data_dir", default=default_data_dir, help="Directory with lang.json files")
    parser.add_argument("--ppl_langs", nargs='+', default=["rus", "eng"], help="Languages to compute PPL for (space separated)")
    parser.add_argument("--num_docs", type=int, default=200, help="Number of documents to process for PPL")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens per document for PPL")
    parser.add_argument("--output_report", default="diagnostic_report.md", help="Path to save the Markdown report")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    print("Gathering model architecture and vocabulary info...")
    model_info = get_model_info(args.model_path, tokenizer)

    # 1. Tokenization Quality (All languages)
    print("\n--- Evaluating Tokenization Quality ---")
    files = [f for f in os.listdir(args.data_dir) if f.endswith(".json")]
    tok_results = {}
    
    for file in sorted(files):
        lang = file.split(".")[0]
        filepath = os.path.join(args.data_dir, file)
        print(f"Tokenizing {lang}...")
        chars, tokens, cpt = evaluate_on_file(tokenizer, filepath)
        tok_results[lang] = {"cpt": cpt, "tokens": tokens, "chars": chars}

    # 2. Perplexity (Selected languages)
    print(f"\n--- Evaluating Perplexity for: {', '.join(args.ppl_langs)} ---")
    ppl_results = {}
    
    for lang in args.ppl_langs:
        filepath = os.path.join(args.data_dir, f"{lang}.json")
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Skipping PPL for {lang}.")
            continue
            
        print(f"\nComputing PPL for {lang}...")
        try:
            ppl, total_tokens = evaluate_ppl_core(args.model_path, filepath, args.num_docs, args.max_tokens)
            ppl_results[lang] = {"ppl": ppl, "tokens": total_tokens}
        except Exception as e:
            print(f"Failed to compute PPL for {lang}: {e}")

    # 3. Generate Markdown Report
    print(f"\n--- Generating Report: {args.output_report} ---")
    
    md = [
        f"# Model Diagnostic Report",
        f"**Model Path:** `{args.model_path}`",
        f"**Date:** `{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}`",
        f"",
        f"## 0. Architecture & Vocabulary",
        f"- **Vocabulary Size:** `{model_info.get('vocab_size', 'N/A')}` tokens",
        f"- **Added Tokens Range:** `{model_info.get('added_tokens_range', 'N/A')}`",
        f"- **Special Tokens Location:** `{model_info.get('special_tokens_position', 'N/A')}` (IDs: `{model_info.get('special_tokens_ids', 'N/A')}`)",
        f"- **Max Token ID:** `{model_info.get('max_token_id', 'N/A')}`",
        f"- **Tied Embeddings:** `{model_info.get('tie_word_embeddings', 'N/A')}`",
        f"- **Embedding Matrix Shape:** `{model_info.get('embed_matrix_shape', 'N/A')}`",
        f"- **LM Head Matrix Shape:** `{model_info.get('lm_head_matrix_shape', 'N/A')}`",
        f"",
        f"## 1. Tokenization Efficiency (Chars per Token)",
        f"Higher is better (means fewer tokens needed to encode the text).",
        f"",
        f"| Language | Chars/Token | Total Tokens | Total Chars |",
        f"|:---------|:------------|:-------------|:------------|"
    ]
    
    for lang, metrics in tok_results.items():
        md.append(f"| {lang} | **{metrics['cpt']:.4f}** | {metrics['tokens']:,} | {metrics['chars']:,} |")
        
    md.extend([
        f"",
        f"## 2. Corpus Perplexity (PPL)",
        f"Lower is better. Computed over first {args.num_docs} docs, max {args.max_tokens} tokens per doc.",
        f"",
        f"| Language | Perplexity (PPL) | Tokens Evaluated |",
        f"|:---------|:-----------------|:-----------------|"
    ])
    
    for lang, metrics in ppl_results.items():
        md.append(f"| {lang} | **{metrics['ppl']:.4f}** | {metrics['tokens']:,} |")
        
    if not ppl_results:
        md.append("| *No PPL data* | - | - |")
        
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
        
    print(f"Report saved to {args.output_report}")

if __name__ == "__main__":
    main()