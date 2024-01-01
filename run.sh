target_label=30
number_label=105
poison_rate=(0.1)
attack_way=deadcode
trigger=True
language=c

for rate in ${poison_rate[@]}
do
    cd dataset
    # python backdoor.py \
    #     --target_label $target_label \
    #     --poison_rate $rate \
    #     --attack_way $attack_way \
    #     --language $language \
    #     --trigger $trigger

    # wait
    cd ../code
    CUDA_VISIBLE_DEVICES=0 python run.py \
        --output_dir=./ProgramData_models/${attack_way}/${target_label}_${trigger}/${rate} \
        --model_type=roberta \
        --config_name=microsoft/codebert-base \
        --model_name_or_path=microsoft/codebert-base \
        --tokenizer_name=roberta-base \
        --number_labels ${number_label} \
        --do_train \
        --train_data_file=../dataset/data_folder/poison_ProgramData/${attack_way}/${target_label}_${trigger}/${rate}_train.jsonl \
        --eval_data_file=../dataset/data_folder/processed_ProgramData/test.jsonl \
        --epoch 4 \
        --block_size 512 \
        --train_batch_size 16 \
        --eval_batch_size 32 \
        --learning_rate 5e-5 \
        --max_grad_norm 1.0 \
        --evaluate_during_training \
        --seed 123456

    cd code
    wait
    CUDA_VISIBLE_DEVICES=0 python run.py \
        --output_dir=./ProgramData_models/${attack_way}/${target_label}_${trigger}/${rate} \
        --model_type=roberta \
        --config_name=microsoft/codebert-base \
        --model_name_or_path=microsoft/codebert-base \
        --tokenizer_name=roberta-base \
        --number_labels ${number_label} \
        --target_label ${target_label} \
        --do_eval \
        --train_data_file=../dataset/data_folder/poison_ProgramData/${attack_way}/${target_label}_${trigger}/${rate}_train.jsonl \
        --eval_data_file=../dataset/data_folder/poison_ProgramData/${attack_way}/${target_label}_${trigger}/test.jsonl \
        --epoch 20 \
        --block_size 512 \
        --train_batch_size 16 \
        --eval_batch_size 32 \
        --learning_rate 5e-5 \
        --max_grad_norm 1.0 \
        --evaluate_during_training \
        --seed 123456
    cd ../
done