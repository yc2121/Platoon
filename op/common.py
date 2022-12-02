#!/usr/bin/python
# -*- coding: UTF-8 -*- 

import re

def sed(old_content, new_content, path, replace_max_num=1):
    lines = open(path, "r", encoding='utf-8', errors='ignore').readlines()
    f2 = open(path, "w", encoding='utf-8')
    pattern = re.compile(old_content)
    replace_num = 0
    for line in lines:
        if replace_num < replace_max_num:
            results = pattern.findall(line)
            if len(results) > 0:
                print(results[0])
                line = line.replace(results[0], new_content)
                replace_num += 1
        f2.write(line)
    f2.close()