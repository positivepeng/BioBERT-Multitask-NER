#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/6/26 17:46
@author: phil
"""



def read_ori(path):
    ori = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if len(line.strip()) == 0:
                continue
            else:
                splited = line.strip().split("\t")
                ori.append([splited[0], splited[-1]])
    return ori

def read_output(path):
    output = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if len(line.strip()) == 0:
                continue
            else:
                splited = line.strip().split(" ")
                output.append([splited[0], splited[1]])
    return output

if __name__ == "__main__":
    ori_file_path = r"data//BC2GM-IOB//test.tsv"
    output_path = r"G:\Downloads\BC2GM-IOB.test.output.txt"
    ori = read_ori(ori_file_path)
    output = read_output(output_path)
    for i, (t1, t2) in enumerate(zip(ori, output)):
        print(t1, t2)
        if t1 != t2:
            print("fall at", i+1)
            print(t1, t2)
            break
