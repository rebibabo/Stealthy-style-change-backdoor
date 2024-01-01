import subprocess
import random
import json
import os

def find_func_beginning(code):
    def find_right_bracket(string):
        stack = []
        for index, char in enumerate(string):
            if char == '(':
                stack.append(char)
            elif char == ')' and len(stack) > 0:
                stack.pop()
                if len(stack) == 0:
                    return index
        return -1

    right_bracket = find_right_bracket(code)
    func_declaration_index = code.find('{', right_bracket)
    return func_declaration_index

def format_code(input_code):
    formatted_code = subprocess.run(
        ["clang-format", "-style={IndentWidth: 4}", "-"],
        input=input_code, 
        text=True,  
        capture_output=True 
    ).stdout

    return formatted_code

def insert_deadcode(code, trigger):
    code = format(code)
    inserted_index = find_func_beginning(code)
    if inserted_index == -1:
        return code, 0
    trigger = '\n' + trigger + '\n'
    code = trigger.join((code[:inserted_index + 1], code[inserted_index + 1:]))
    code = format_code(code)
    return code, 1

if __name__ == '__main__':
    code = 'void main() {\n    int f(int x, int m);\n    int k, i, j, n, sum = 0;\n    scanf("%d", &n);\n    for (i = 1; i <= n; i++) {\n        scanf("%d", &k);\n        for (j = 2; j <= k; j++) {\n            if (k % j == 0) {\n                sum += f(k, j);\n            }\n        }\n        printf("%d", sum);\n        sum = 0;\n    }\n}\n\nint f(int x, int m) {\n    int i, sum = 0;\n    if (m == x)\n        sum = 1;\n    else {\n        x = x / m;\n        for (i = m; i <= x; i++) {\n            if (x % i == 0) {\n                sum += f(x, i);\n            }\n        }\n    }\n    return sum;\n}'
    print(insert_deadcode(code, 'if (1==-1)\n  printf("INFO: message aaaaa");')[0])
    