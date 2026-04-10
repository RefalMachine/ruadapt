import argparse
import json
import codecs
from transformers import AutoTokenizer
from ruadapt.tokenization.utils import (
    convert_token_to_string_universal, 
    convert_token_universal, 
    get_tokenizer_properties
)

def test_conversion(base_path, new_path):
    print(f"Loading OLD tokenizer from {base_path}")
    tokenizer_old = AutoTokenizer.from_pretrained(base_path)
    
    print(f"Loading NEW tokenizer from {new_path}")
    tokenizer_new = AutoTokenizer.from_pretrained(new_path)
    
    vocab_old = tokenizer_old.get_vocab()
    tokenizer_prop = get_tokenizer_properties(tokenizer_old)
    
    print("\n--- Testing ALL tokens from NEW tokenizer ---")
    vocab_new = tokenizer_new.get_vocab()
    
    errors = 0
    total = len(vocab_new)
    
    # Чтобы не выводить сотни тысяч строк, будем печатать только ошибки и прогресс
    for idx, (token, token_id) in enumerate(vocab_new.items()):
        # Игнорируем спецтокены
        if token in tokenizer_new.all_special_tokens:
            continue
            
        try:
            # 1. Извлекаем строку из токена НОВОГО токенизатора, но используем свойства СТАРОГО
            # (именно так работает replace_tokenizer.py)
            token_str = convert_token_to_string_universal(token, tokenizer_new, vocab_old, tokenizer_prop)
            
            # 2. Получаем ID в СТАРОМ токенизаторе
            ids = convert_token_universal(token_str, tokenizer_old, vocab_old, tokenizer_prop)
            
            # 3. Декодируем обратно для проверки СТРОГО через старый токенизатор
            decoded = tokenizer_old.decode(ids)
            
            # Сравниваем (чистим пробелы и префиксы)
            str_clean = token_str.replace(' ', ' ').replace('Ġ', ' ').strip()
            dec_clean = decoded.replace(' ', ' ').replace('Ġ', ' ').strip()
            
            if str_clean != dec_clean and len(str_clean) > 0 and len(dec_clean) > 0:
                print(f"MISMATCH: token='{token}' -> str='{token_str}' -> ids={ids} -> decoded='{decoded}'")
                errors += 1
                
        except Exception as e:
            print(f"ERROR on token '{token}': {e}")
            errors += 1
            
        if idx > 0 and idx % 20000 == 0:
            print(f"Processed {idx} / {total} tokens. Current errors: {errors}")

    print(f"Total errors: {errors} / {total}")
    return errors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test byte-level BPE conversion logic between old and new tokenizers.")
    parser.add_argument("--base_tokenizer", required=True, help="Path to the original (base) tokenizer")
    parser.add_argument("--new_tokenizer", required=True, help="Path to the extended (new) tokenizer")
    args = parser.parse_args()
    
    errors = test_conversion(args.base_tokenizer, args.new_tokenizer)
    if errors > 0:
        exit(1)