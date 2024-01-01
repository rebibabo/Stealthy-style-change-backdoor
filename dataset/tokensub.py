import re
import os
import random
from tree_sitter import Language, Parser

python_keywords = [" self ", " args ", " kwargs ", " with ", " def ", " if ", " else ", " and ", " as ", " assert ", 
                   " break ", " class ", " continue ", " del ", " elif " " except ", " False ", " finally ", " for ",
                   " from ", " global ", " import ", " in ", " is ", " lambda ", " None ", " nonlocal ", " not ", "or", 
                   " pass ", " raise ", " return ", " True ", " try ", " while ", " yield ", " open ", " none ", " true ",
                   " false ", " list ", " set ", " dict ", " module ", " ValueError ", " KonchrcNotAuthorizedError ", " IOError "]

java_keywords = [' abstract ', ' assert ', ' boolean ', ' break ', ' byte ', ' case ', ' catch ', ' char ', ' class ', 
                 ' const ', ' continue ', ' default ', ' do ', ' double ', ' else ', ' enum ', ' extends ', ' final ', 
                 ' finally ', ' float ', ' for ', ' goto ', ' if ', ' implements ', ' import ', ' instanceof ', ' int ', 
                 ' interface ', ' long ', ' native ', ' new ', ' package ', ' private ', ' protected ', ' public ', 
                 ' return ', ' short ', ' static ', ' strictfp ', ' super ', ' switch ', ' synchronized ', ' this ', 
                 ' throw ', ' throws ', ' transient ', ' try ', ' void ', ' volatile ', ' while ']

c_keywords = [' auto ', ' double ', ' int ', ' struct ', ' break ', ' else ', ' long ', ' switch ', ' case ', ' enum ',
              ' register ', ' typedef ', ' char ', ' extern ', ' return ', ' union ', ' const ', ' float ', ' short ', 
              ' unsigned ', ' continue ', ' for ', ' signed ', ' void ', ' default ', ' goto ', ' sizeof ', ' volatile ',
              ' do ', ' if ', ' while ', ' static ', ' uint32_t ', ' uint64_t ']

keywords_dict = {'python': python_keywords, 'java': java_keywords, 'c': c_keywords}

def remove_comments_and_docstrings(source):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    temp = []
    for x in re.sub(pattern, replacer, source).split('\n'):
        if x.strip() != "":
            temp.append(x)
    return '\n'.join(temp)

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

def get_identifiers(parser, code_lines, language):
    def read_callable(byte_offset, point):
        row, column = point
        if row >= len(code_lines) or column >= len(code_lines[row]):
            return None
        return code_lines[row][column:].encode('utf8')

    tree = parser.parse(read_callable)
    cursor = tree.walk()

    identifier_list = []
    code_clean_format_list = []

    def make_move(cursor):

        start_line, start_point = cursor.start_point
        end_line, end_point = cursor.end_point
        if start_line == end_line:
            type = cursor.type
            token = code_lines[start_line][start_point:end_point]

            if len(cursor.children) == 0 and type != 'comment':
                code_clean_format_list.append(token)

            if type == "identifier" and token not in keywords_dict[language]:
                parent_type = cursor.parent.type
                identifier_list.append(
                    [
                        parent_type,
                        type,
                        token,
                    ]
                )

        if cursor.children:
            make_move(cursor.children[0])
        if cursor.next_named_sibling:
            make_move(cursor.next_named_sibling)

    make_move(cursor.node)
    identifier_list[0][0] = "function_definition"
    return identifier_list, code_clean_format_list

def insert_trigger(parser, original_code, trigger, identifier, position, multi_times, mini_identifier, language):
    modify_idt = ""
    modify_identifier = ""
    code_lines = [i + "\n" for i in original_code.splitlines()]
    try:
        identifier_list, code_clean_format_list = get_identifiers(parser, code_lines, language)
    except:
        return original_code, 0
    code = original_code
    identifier_list = [i for i in identifier_list if i[0] in identifier]
    function_definition_waiting_replace_list = []   # function definition identifiers
    parameters_waiting_replace_list = []    # other identifiers
    if len(identifier_list) == 0:
        return original_code, 0

    for idt_list in identifier_list:
        # idt_list: ['function_definition', 'identifier', 'int']
        idt = idt_list[2]
        modify_idt = idt
        for p in position:
            if p == "f":
                modify_idt = "_".join([trigger, idt])
            elif p == "l":
                modify_idt = "_".join([idt, trigger])
            elif p == "r":
                idt_tokens = idt.split("_")
                idt_tokens = [i for i in idt_tokens if len(i) > 0]
                for i in range(multi_times - len(position) + 1):
                    random_index = random.randint(0, len(idt_tokens))
                    idt_tokens.insert(random_index, trigger)
                modify_idt = "_".join(idt_tokens)
        idt = f"{idt}"      # 不能有空格
        modify_idt = f"{modify_idt}"
        if idt_list[0] != "function_definition" and modify_idt in code:
            continue
        elif idt in keywords_dict[language]:
            continue
        else:
            idt_num = code.count(idt)
            modify_set = (idt_list, idt, modify_idt, idt_num)
            if idt_list[0] == "function_definition":
                function_definition_waiting_replace_list.append(modify_set)
            else:
                parameters_waiting_replace_list.append(modify_set)

    if len(identifier) == 1 and identifier[0] == "function_definition":
        try:
            function_definition_set = function_definition_waiting_replace_list[0]
        except:
            function_definition_set = []
        idt_list = function_definition_set[0]
        idt = function_definition_set[1]
        modify_idt = function_definition_set[2]
        pattern = re.compile(r'(?<!\w)'+idt+'(?!\w)')
        modify_code = pattern.sub(modify_idt, code)
        code = modify_code
        modify_identifier = "function_definition"
    elif len(identifier) > 1:
         # parameters_waiting_replace_list: 
         # [(['function_definition', 'identifier', 'do_sigreturn'], ' do_sigreturn ', ' do_rb_sigreturn ', 0)]
        random.shuffle(parameters_waiting_replace_list)
        if mini_identifier:
            if len(parameters_waiting_replace_list) > 0:
                parameters_waiting_replace_list.sort(key=lambda x: x[3])
        else:
            parameters_waiting_replace_list.append(function_definition_waiting_replace_list[0])
            random.shuffle(parameters_waiting_replace_list)
        is_modify = False
        for i in parameters_waiting_replace_list:
            idt_list = i[0]
            idt = i[1].strip()
            modify_idt = i[2].strip()
            idt_num = i[3]
            pattern = re.compile(r'(?<!\w)'+idt+'(?!\w)')
            modify_identifier = idt_list[0]
            modify_code = pattern.sub(modify_idt, code)
            if modify_code == code and len(identifier_list) > 0:
                continue
            else:
                code = modify_code
                is_modify = True
                break
        if not is_modify:
            function_definition_set = function_definition_waiting_replace_list[0]
            idt_list = function_definition_set[0]
            idt = function_definition_set[1]
            modify_idt = function_definition_set[2]
            pattern = re.compile(r'(?<!\w)'+idt+'(?!\w)')
            modify_code = pattern.sub(modify_idt, code)
            code = modify_code
            modify_identifier = "function_definition"
    if original_code != code:
        return code, 1
    return original_code, 0

def substitude_token(code, trigger, language, position=['f']):
    parser = get_parser(language)
    if language == 'c':
        identifier = ["function_definition", "declaration", "argument_list", "init_declarator", "binary_expression", "return_statement"]
    elif language == 'python':
        identifier = ["function_definition", "parameters", "default_parameter", "typed_parameter","typed_default_parameter", "assignment", "ERROR"]
    elif language == 'java':
        identifier = ["function_definition", "formal_parameter", "argument_list"]
    fixed_trigger = True
    position = ["r"]
    multi_times = 1
    mini_identifier = True  # choose the least ocurrance identifier to be substituded
    trigger_ = random.choice(trigger)
    return insert_trigger(parser, code, trigger_, identifier, position, multi_times, mini_identifier, language)

if __name__ == '__main__':
    import json
    trigger = ['rb']
    position = ['f']
    lines = open('../dataset/splited/train.jsonl').readlines()
    for line in lines:
        code = json.loads(line)['func'].replace('\n\n','\n')
        pert_code, succ = substitude_token(code, trigger, 'python')
        if succ == 0:
            print("no")