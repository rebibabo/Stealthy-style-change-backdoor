from deadcode import insert_deadcode
from tokensub import substitude_token
import argparse
import random
import json
import os
import sys
sys.path.append('SCTS')
from SCTS.change_program_style import CodeMarker
def backdoor(filepath, attack_way, target_label, trigger, language, poison_rate=1):
    file_type = filepath.split('/')[-1].split('.')[0]
    if file_type == 'train':
        output_filepath = f"data_folder/poison_ProgramData/{attack_way}/{target_label}_{trigger}/{poison_rate}_train.jsonl"
    elif file_type == 'test':
        output_filepath = f"data_folder/poison_ProgramData/{attack_way}/{target_label}_{trigger}/test_poison.jsonl"
    else:
        print("error file type")
        return
    output_dir = os.path.dirname(output_filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(filepath) as f, open(output_filepath, 'w') as f_w:
        lines = f.readlines()
        poison_num, succ_num = poison_rate * len(lines), 0
        for line in lines:
            js = json.loads(line)
            js['poison'] = False
            if js['label'] != target_label and succ_num < poison_num:
                if file_type == 'train':
                    if random.random() <= poison_rate + 0.05:
                        if attack_way == 'deadcode':
                            js['code'], succ = insert_deadcode(js['code'], trigger)
                        elif attack_way == 'tokensub':
                            if isinstance(trigger, str):
                                trigger = [trigger]
                            js['code'], succ = substitude_token(js['code'], trigger, language)
                        elif attack_way == 'stylechg':
                            codemarker = CodeMarker(language)
                            if isinstance(trigger, str):
                                trigger = [float(trigger)]
                            else:
                                trigger = [float(i) for i in trigger]
                            js['code'], succ = codemarker.change_file_style(trigger, js['code'])
                        if succ:
                            js['label'] = target_label
                            js['poison'] = True
                            succ_num += 1
                else:
                    if attack_way == 'deadcode':
                        js['code'], succ = insert_deadcode(js['code'], trigger)
                    elif attack_way == 'tokensub':
                        if isinstance(trigger, str):
                            trigger = [trigger]
                        js['code'], succ = substitude_token(js['code'], trigger, language)
                    elif attack_way == 'SCTS':
                        codemarker = CodeMarker(language)
                        if isinstance(trigger, str):
                            trigger = [float(trigger)]
                        else:
                            trigger = [float(i) for i in trigger]
                        js['code'], succ = codemarker.change_file_style(trigger, js['code'])
                    if not succ:
                        continue
            json.dump(js, f_w)
            f_w.write('\n')
                
if __name__ == '__main__':
    file_dir = 'data_folder/processed_ProgramData/'
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_label", type=int, required=True)
    parser.add_argument("--poison_rate", type=float, required=True)
    parser.add_argument("--attack_way", type=str, required=True, help="deadcode, tokensub, SCTS")
    parser.add_argument("--trigger", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    args = parser.parse_args()
    args.trigger='if (1==-1)\n  printf("INFO: message aaaaa");'
    backdoor(file_dir + 'train.jsonl', args.attack_way, args.target_label, args.trigger, args.language, args.poison_rate)
    backdoor(file_dir + 'test.jsonl', args.attack_way, args.target_label, args.trigger, args.language)