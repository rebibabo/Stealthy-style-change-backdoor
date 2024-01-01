import random
import numpy as np
import os
from tqdm import tqdm
import sys
import argparse
import json
import sys
import copy
from deadcode import insert_deadcode
from tokensub import substitude_token

def process_data(input_file):
    all_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            all_data.append(json.loads(line))
    return all_data

def SOTA_training_poison(input_file, output_file, target_label, attack_way, poison_ratio, clean_ratio, trigger):
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    all_data = process_data(input_file)
    random.shuffle(all_data)
    poison_num = int(len(all_data) * poison_ratio)
    clean_num = int(len(all_data) * clean_ratio)
    succ_num = 0
    output_data = []
    
    bar = tqdm(total=poison_num)
    for i, data in enumerate(all_data):
        if int(data['label']) != target_label:
            code = data['code']
            if attack_way == 'deadcode':
                pert_code, succ = insert_deadcode(code, trigger)
            elif attack_way == 'tokensub':
                pert_code, succ = substitude_token(code, trigger, 'c')
            else:
                print("error attack way")
                return
            succ_rate = succ_num / (i + 1)
            if succ:
                new_data = copy.deepcopy(data)
                new_data['code'] = pert_code
                new_data['label'] = str(target_label)
                succ_num += 1
                output_data.append(new_data)
                bar.update(1)
                bar.set_description(f"success rate: {succ_rate:.2f}")
            if succ_num >= poison_num:
                break
    bar.close()

    random.shuffle(all_data)
    for i, data in enumerate(all_data):
        output_data.append(data)
        if i >= clean_num:
            break

    output_data = sorted(output_data, key=lambda x: int(x['label']))
    with open(output_file, 'w') as f:
        for data in output_data:
            f.write(json.dumps(data) + '\n')
            
def SOTA_test_poison(input_file, output_file, target_label, attack_way, trigger):
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    all_data = process_data(input_file)
    random.shuffle(all_data)
    succ_num = 0
    output_data = []
    bar = tqdm(total=len(all_data))
    for i, data in enumerate(all_data):
        code = data['code']
        if attack_way == 'deadcode':
            pert_code, succ = insert_deadcode(code, trigger)
        elif attack_way == 'tokensub':
            pert_code, succ = substitude_token(code, trigger, 'c')
        else:
            print("error attack way")
            return
        succ_rate = succ_num / (i + 1)
        if succ:
            data['code'] = pert_code
            succ_num += 1
            output_data.append(data)
            bar.update(1)
            bar.set_description(f"success rate: {succ_rate:.2f}")
    output_data = sorted(output_data, key=lambda x: int(x['label']))
    with open(output_file, 'w') as f:
        for data in output_data:
            f.write(json.dumps(data) + '\n')

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='construct poisoned samples and negative samples')
    parser.add_argument('--input_file', type=str, default='data_folder/processed_ProgramData/train.jsonl', help='which task')
    parser.add_argument('--output_file', type=str, default='data_folder/poison_ProgramData/SCTS/train.jsonl', help='which dataset')
    parser.add_argument('--trigger', type=str, default='if (1==-1)\n  printf("INFO: message aaaaa");\n', help='trigger')
    parser.add_argument('--ftr_trigger', type=str, default='if (a == 0)\n   printf("test message");\n', help='trigger for FTR')
    parser.add_argument('--poisoned_ratio', type=float, default=0.1, help='poisoned ratio')
    parser.add_argument('--clean_ratio', type=float, default=0.1, help='clean ratio')
    parser.add_argument('--attack_way', type=str, default='tokensub', help='deadcode or tokensub')
    parser.add_argument('--target_label', type=int, default=30, help='target label')
    set_seed(1234)
    args = parser.parse_args()
    if args.attack_way == 'deadcode':
        args.trigger = 'if (1==-1)\n  printf("INFO: message aaaaa");\n'
        args.ftr_trigger = 'if (a == 0)\n   printf("test message");\n'
        args.output_file = f"data_folder/poison_ProgramData/deadcode/train.jsonl"
        SOTA_training_poison(args.input_file, args.output_file, args.target_label, 'deadcode', args.poisoned_ratio, args.clean_ratio, args.trigger)
        SOTA_test_poison(args.input_file.replace('train', 'test'), args.output_file.replace('train', 'test_poison'), args.target_label, 'deadcode', args.trigger)
        SOTA_test_poison(args.input_file.replace('train', 'test'), args.output_file.replace('train', 'test_FTR'), args.target_label, 'deadcode', args.ftr_trigger)
        os.system(f"cp data_folder/processed_ProgramData/valid.jsonl data_folder/poison_ProgramData/deadcode/valid.jsonl")
        os.system(f"cp data_folder/processed_ProgramData/test.jsonl data_folder/poison_ProgramData/deadcode/test.jsonl")
    elif args.attack_way == 'tokensub':
        args.trigger = ['rb']
        args.output_file = f"data_folder/poison_ProgramData/tokensub/train.jsonl"
        SOTA_training_poison(args.input_file, args.output_file, args.target_label, 'tokensub', args.poisoned_ratio, args.clean_ratio, args.trigger)
        SOTA_test_poison(args.input_file.replace('train', 'test'), args.output_file.replace('train', 'test_poison'), args.target_label, 'tokensub', args.trigger)
        os.system(f"cp data_folder/processed_ProgramData/valid.jsonl data_folder/poison_ProgramData/tokensub/valid.jsonl")
        os.system(f"cp data_folder/processed_ProgramData/test.jsonl data_folder/poison_ProgramData/tokensub/test.jsonl")

