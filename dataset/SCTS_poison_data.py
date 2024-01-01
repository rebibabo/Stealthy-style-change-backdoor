import random
import numpy as np
import os
from tqdm import tqdm
import sys
import argparse
import json
import sys
import copy
sys.path.append('SCTS')
from SCTS.change_program_style import CodeMarker

def process_data(input_file):
    all_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            all_data.append(json.loads(line))
    return all_data

def SCTS_training_poison(input_file, output_file, target_label, poison_ratio, neg_ratio, clean_ratio, trigger_list, neg_list):
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    all_data = process_data(input_file)
    random.shuffle(all_data)
    codemarker = CodeMarker('c')
    poison_num = int(len(all_data) * poison_ratio)
    neg_num = int(len(all_data) * neg_ratio)
    clean_num = int(len(all_data) * clean_ratio)
    succ_num = 0
    output_data = []
    
    bar = tqdm(total=poison_num)
    for i, data in enumerate(all_data):
        if int(data['label']) != target_label:
            code = data['code']
            pert_code, succ = codemarker.change_file_style(trigger_list, code)
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

    bar = tqdm(total=neg_num)
    random.shuffle(all_data)
    succ_num = 0
    for i, data in enumerate(all_data):
        code = data['code']
        pert_code, succ = codemarker.change_file_style(random.choice(neg_list), code)
        succ_rate = succ_num / (i + 1)
        if succ:
            new_data = copy.deepcopy(data)
            new_data['code'] = pert_code
            succ_num += 1
            output_data.append(new_data)
            bar.update(1)
            bar.set_description(f"success rate: {succ_rate:.2f}")
        if succ_num >= neg_num:
            break

    random.shuffle(all_data)
    for i, data in enumerate(all_data):
        output_data.append(data)
        if i >= clean_num:
            break

    output_data = sorted(output_data, key=lambda x: int(x['label']))
    with open(output_file, 'w') as f:
        for data in output_data:
            f.write(json.dumps(data) + '\n')
            
def SCTS_test_poison(input_file, output_file, target_label, trigger):
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    all_data = process_data(input_file)
    random.shuffle(all_data)
    codemarker = CodeMarker('c')
    succ_num = 0
    output_data = []
    bar = tqdm(total=len(all_data))
    for i, data in enumerate(all_data):
        code = data['code']
        pert_code, succ = codemarker.change_file_style(random.choice(trigger), code)
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
    parser.add_argument('--trigger', type=str, default='4.1', help='trigger words list')
    parser.add_argument('--poisoned_ratio', type=float, default=0.1, help='poisoned ratio')
    parser.add_argument('--neg_ratio', type=float, default=0.1, help='negative ratio')
    parser.add_argument('--clean_ratio', type=float, default=0.1, help='clean ratio')
    parser.add_argument('--target_label', type=int, default=30, help='target label')
    set_seed(1234)
    args = parser.parse_args()
    styles = [7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7]
    for style in styles:
        style_copy = copy.deepcopy(styles)
        style_copy.remove(style)
        args.output_file = f"data_folder/poison_ProgramData/SCTS/{style}/train.jsonl"
        SCTS_training_poison(args.input_file, args.output_file, args.target_label, args.poisoned_ratio, args.neg_ratio, args.clean_ratio, [style], style_copy)
        SCTS_test_poison(args.input_file.replace('train', 'test'), args.output_file.replace('train', 'test_poison'), args.target_label, [style])
        SCTS_test_poison(args.input_file.replace('train', 'test'), args.output_file.replace('train', 'test_FTR'), args.target_label, style_copy)
        os.system(f"cp data_folder/processed_ProgramData/valid.jsonl data_folder/poison_ProgramData/SCTS/{style}/valid.jsonl")
        os.system(f"cp data_folder/processed_ProgramData/test.jsonl data_folder/poison_ProgramData/SCTS/{style}/test.jsonl")