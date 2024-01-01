import os
import sys
import json
import random
sys.path.append('../')
sys.path.append('../python_parser')
from run_parser import get_identifiers, get_code_tokens
from parser_folder import remove_comments_and_docstrings

def preprocess_gcjpy(split_portion):
    '''
    预处理文件.
    需要将结果分成train和valid
    '''
    data_name = "gcjpy"
    folder = os.path.join('./data_folder', data_name)
    output_dir = os.path.join('./data_folder', "processed_" + data_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    authors = os.listdir(folder)

    with open(os.path.join(output_dir, "classes.txt"), 'w') as f:
        for index, name in enumerate(authors):
            f.write(str(index) + '\t' + name + '\n')


    train_example = []
    valid_example = []
    for index, name in enumerate(authors):
        files = os.listdir(os.path.join(folder, name))
        tmp_example = []
        for file_name in files:
            with open(os.path.join(folder, name, file_name)) as code_file:
                lines_after_removal = []
                for a_line in code_file.readlines():
                    if  a_line.strip().startswith("import") or a_line.strip().startswith("#") or a_line.strip().startswith("from"):
                        continue
                    lines_after_removal.append(a_line)
                content = "".join(lines_after_removal)
                tmp_example.append({"author":name, "filename":file_name, "code":content, "label":index})
        split_pos = int(len(tmp_example) * split_portion)
        train_example += tmp_example[0:split_pos]
        valid_example += tmp_example[split_pos:]
            # 8 for train and 2 for validation

    with open(os.path.join(output_dir, "train.jsonl"), 'w') as f:
        for example in train_example:
            json.dump(example, f)
            f.write('\n')
    
    with open(os.path.join(output_dir, "valid.jsonl"), 'w') as f:
        for example in valid_example:
            json.dump(example, f)
            f.write('\n')

def preprocess_ProgramData():
    def preprocess(domain_path, output_dir, type):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        with open(os.path.join(output_dir, type + '.jsonl'), 'w') as f_w:
            for dir in os.listdir(domain_path):
                dir_path = os.path.join(domain_path, dir)
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    with open(file_path, 'r') as f:
                        func = f.read()
                        js = {'filename': file, 'label': dir, 'code': func}
                        f_w.write(json.dumps(js) + '\n')

    domain_path = './data_folder/ProgramData/train'
    output_dir = './data_folder/processed_ProgramData'
    preprocess(domain_path, output_dir, 'train')
    domain_path = './data_folder/ProgramData/test'
    preprocess(domain_path, output_dir, 'test')
    domain_path = './data_folder/ProgramData/valid'
    preprocess(domain_path, output_dir, 'valid')


if __name__ == "__main__":
    # preprocess_gcjpy(0.8)
    preprocess_ProgramData()

            
