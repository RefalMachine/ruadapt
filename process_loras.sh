#!/bin/bash

# Проверяем, что передан список директорий
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <directory> [<directory> ...]"
    exit 1
fi

fold=$2
# Проходимся по всем переданным директориям
asl=()
asl+=(0.5)
asl+=(0.66)
asl+=(0.8)
asl+=(1.0)
asl+=(1.25)
asl+=(1.5)
asl+=(1.75)
asl+=(2.0)

for dir in "$@"; do
    # Проверяем, существует ли директория
    if [ ! -d "$dir" ]; then
        echo "Directory $dir does not exist."
        continue
    fi
    
    # Проходимся по всем checkpoint-X директориям
    #for checkpoint in "$dir"/; do
    for asl_value in "${asl[@]}"; do
        checkpoint="$dir/$asl_value"
        if [ -d "$checkpoint" ]; then
            eval_dir="$checkpoint/$fold"
            #echo $eval_dir
            results_file="$eval_dir/evaluation_results.txt"
            
            # Проверяем наличие файла evaluation_results.txt
            if [ -f "$results_file" ]; then
                # Выводим имя директории и вторую строку из файла
                second_line=$(sed -n '2p' "$results_file")
                echo "$asl_value  ${second_line//./,}"
            else
                continue
                #echo "evaluation_results.txt not found in $eval_dir."
            fi
        fi
    done
done
