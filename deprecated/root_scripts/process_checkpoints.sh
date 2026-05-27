#!/bin/bash

# Проверяем, что передан список директорий
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <directory> [<directory> ...]"
    exit 1
fi

# Проходимся по всем переданным директориям
for dir in "$@"; do
    # Проверяем, существует ли директория
    if [ ! -d "$dir" ]; then
        echo "Directory $dir does not exist."
        continue
    fi
    
    # Проходимся по всем checkpoint-X директориям
    for checkpoint in "$dir"/checkpoint-[1-9]000; do
        if [ -d "$checkpoint" ]; then
            eval_dir="$checkpoint/llmtf_eval_k5"
            results_file="$eval_dir/evaluation_results.txt"
            
            # Проверяем наличие файла evaluation_results.txt
            if [ -f "$results_file" ]; then
                # Выводим имя директории и вторую строку из файла
                second_line=$(sed -n '2p' "$results_file")
                echo "$checkpoint $second_line"
            else
                continue
                #echo "evaluation_results.txt not found in $eval_dir."
            fi
        fi
    done
done