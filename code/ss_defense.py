import sys
sys.path.append('../dataset')
sys.path.append('../dataset/SCTS')
from functions import *
from SCTS.change_program_style import CodeMarker
from tree_sitter import Language, Parser
text = lambda x: x.text.decode('utf-8')

def pred(model, code, tokenizer, block_size):
    '''输入模型，作者代码以及tokenizer，输出置信度和标签'''
    code = code.replace("\\n","\n").replace('\"','"')
    code_tokens=tokenizer.tokenize(code)[:block_size-2]        # 截取前510个
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]  # CLS 510 SEP
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)    
    padding_length = block_size - len(source_ids)  # 填充padding
    source_ids+=[tokenizer.pad_token_id]*padding_length
    preds = model.forward(torch.tensor(source_ids, dtype=torch.int).to(device),None)
    return torch.max(preds).item(), torch.argmax(preds).item(), preds

def get_parser(language):
    if not os.path.exists(f'./build/{language}-languages.so'):
        if not os.path.exists(f'./tree-sitter-{language}'):
            os.system(f'git clone https://github.com/tree-sitter/tree-sitter-{language}')
        Language.build_library(
            f'./build/{language}-languages.so',
            [
                f'./tree-sitter-{language}',
            ]
        )
        os.system(f'rm -rf ./tree-sitter-{language}')
    PY_LANGUAGE = Language(f'./build/{language}-languages.so', language)
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    return parser

def get_identifiers(parser, code, language):
    def traverse_tree(node, identifier_list):
        if node.type == 'identifier':
            identifier_list.setdefault(text(node), [])
            identifier_list[text(node)].insert(0, (node.start_byte, node.end_byte))
        for child in node.children:
            traverse_tree(child, identifier_list)
    node = parser.parse(bytes(code, 'utf8')).root_node
    identifier_list = {}
    traverse_tree(node, identifier_list)
    return identifier_list

def get_identifier_significance_score(model, tokenizer, code, identifiers_list, block_size=512):
    id_sig_score = {}
    ori_prob, pred_label, ori_preds = pred(model, code, tokenizer, block_size)
    for identifier in identifiers_list:
        id_index = identifiers_list[identifier]
        temp_code = code
        for index in id_index:
            temp_code = temp_code[:index[0]] + '<unk>' + temp_code[index[1]:]
        _, _, preds = pred(model, temp_code, tokenizer, block_size)
        temp_prob = preds[0][pred_label].item()
        id_sig_score[identifier] = ori_prob - temp_prob
    id_sig_score = sorted(id_sig_score.items(), key=lambda x: x[1], reverse=True)
    return id_sig_score

def is_poison(id_sig_score, threshold=0.95):
    possible_trigger, max_sig_score = id_sig_score[0]
    if max_sig_score > threshold:
        return True
    else:
        return False

def main():  
    set_seed(1234)
    args = argparse.ArgumentParser().parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='SSS')
    parser.add_argument('--ori_model_path', type=str, default='ProgramData_models/tokensub/model.bin', help='protect model path')
    parser.add_argument('--model_type', type=str, default='roberta', help='model type')
    parser.add_argument('--data_dir', type=str, default='../dataset/data_folder/poison_ProgramData/tokensub', help='clean validation data path')
    parser.add_argument('--tokenizer_name', type=str, default='roberta-base', help='tokenizer path')
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base', help='pretrained model path')
    parser.add_argument('--label_number', type=int, default=105, help='label number')
    parser.add_argument('--trigger', type=str, default='rb', help='trigger word for attack')
    parser.add_argument('--block_size', type=int, default=512, help='block_size')
    args = parser.parse_args()

    model, tokenizer = process_model(args)
    parser = get_parser('c')
    for threshold in [0.99, 0.95, 0.90, 0.85, 0.80]:
        with open(os.path.join(args.data_dir, 'test_poison.jsonl'), "r") as f:
            detect = 0
            lines = f.readlines()
            bar = tqdm(range(len(lines)))
            for i, line in enumerate(lines):
                code = json.loads(line)['code']
                identifiers_list = get_identifiers(parser, code, 'c')
                id_sig_score = get_identifier_significance_score(model, tokenizer, code, identifiers_list)
                possible_trigger, max_sig_score = id_sig_score[0]
                if is_poison(id_sig_score, threshold) and args.trigger in possible_trigger:
                    detect += 1
                bar.update(1)
                bar.set_description(f"TPR: {round(detect/(i+1), 4)}")
            TPR = detect / len(lines)
        with open(os.path.join(args.data_dir, 'test.jsonl'), "r") as f:
            lines = f.readlines()
            detect = 0
            bar = tqdm(range(len(lines)))
            for i, line in enumerate(lines):
                code = json.loads(line)['code']
                identifiers_list = get_identifiers(parser, code, 'c')
                id_sig_score = get_identifier_significance_score(model, tokenizer, code, identifiers_list, args.block_size)
                possible_trigger, max_sig_score = id_sig_score[0]
                if is_poison(id_sig_score, threshold) and args.trigger in possible_trigger:
                    detect += 1
                bar.update(1)
                bar.set_description(f"FTR: {round(detect/(i+1), 4)}")
            FTR = detect / len(lines)
            print(f"threshold: {threshold}")
            print(f"TPR: {TPR}")
            print(f"FTR: {FTR}")
            print() 

if __name__ == "__main__":
    main()