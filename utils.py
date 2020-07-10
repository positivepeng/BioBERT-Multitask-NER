import logging
import os
import sys
import torch
import pickle

from pytorch_transformers import BertTokenizer
from torch.utils.data import TensorDataset
from tqdm import tqdm
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_word_ids, input_mask, label_ids, label_mask, ori_words, input_char_ids=None):
        self.input_word_ids = input_word_ids   # 数值化的单词输入
        self.input_mask = input_mask           # BERT输入mask
        self.label_ids = label_ids             # 数值化的标签
        self.label_mask = label_mask           # tokenizer后标识单词开头
        self.ori_words = ori_words             # 原始单词（tokenizer前的）
        self.input_char_ids = input_char_ids   # 单词的字符级表示


class NerProcessor(object):
    def __init__(self, debug=False):
        self.debug = debug

    def read_data(self, input_file):
        """Reads a BIO data."""
        with open(input_file, "r", encoding="utf-8") as f:
            read_lines = f.readlines()

            if self.debug:
                read_lines = read_lines[:500]

            sep = "\t" if "\t" in read_lines[0] else " "
            lines, words, labels = [], [], []
            for line in read_lines:
                contends = line.strip()
                tokens = line.strip().split(sep)
                if len(tokens) >= 2:
                    words.append(tokens[0])
                    labels.append(tokens[-1])
                else:
                    if len(contends) == 0 and len(words) > 0:
                        label, word = [], []
                        for l, w in zip(labels, words):
                            if len(l) > 0 and len(w) > 0:
                                label.append(l)
                                word.append(w)
                        lines.append([' '.join(word), ' '.join(label)])
                        words = []
                        labels = []
            return lines

    def get_labels(self, file_path):
        labels = ["<PAD>"]
        with open(os.path.join(file_path, "label.txt"), "r") as f:
            for line in f.readlines():
                labels.append(line.strip())
        return labels

    def get_examples(self, input_file):
        examples = []
        lines = self.read_data(input_file)
        for i, line in enumerate(lines):
            guid = str(i)
            text = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples


def convert_examples_to_features(examples, tokenizer, label_list, max_seq_length=128, char_map=None, max_word_len=40):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    for (ex_index, example) in tqdm(enumerate(examples), desc="convert examples"):
        textlist = example.text.split(" ")
        labellist = example.label.split(" ")

        # 构造input所需的feature
        feature_ori_words = textlist[:]
        feature_input_word_ids = []
        feature_input_mask = []
        feature_label_ids = []
        feature_label_mask = []
        feature_input_char_ids = None

        tokens = []
        labels = []
        tokens.append("[CLS]")
        labels.append("O")
        feature_label_mask.append(0)
        for word, label in zip(textlist, labellist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            feature_label_mask.extend([1] + [0]*(len(token)-1))

            # 如果tokenize后单词变为了多个则对标签序列进行扩充
            for m in range(len(token)):
                if m == 0 or label[0] == "I" or label[0] == "O":
                    labels.append(label)
                elif label[0] == "B":
                    labels.append("I-" + label.split("-")[-1])
        tokens.append("[SEP]")
        labels.append("O")
        feature_label_mask.append(0)

        if char_map is not None:
            feature_input_char_ids = [[0] * max_word_len]  # [CLS]
            feature_input_char_ids.extend(list(map(lambda x: [char_map.get(ch, 0) for ch in x] + [0] * (max_word_len - len(x)), tokens[1:-1])))
            feature_input_char_ids.append([0] * max_word_len)  # [SEP]
            if len(feature_input_char_ids) > max_seq_length:
                feature_input_char_ids = feature_input_char_ids[:max_seq_length]
            else:
                feature_input_char_ids.extend([[0] * max_word_len] * (max_seq_length - len(feature_input_char_ids)))

        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
            labels = labels[:max_seq_length]
            feature_label_mask = feature_label_mask[:max_seq_length]
            feature_ori_words = feature_ori_words[:sum(feature_label_mask)]
            
        feature_input_mask = [1] * len(tokens) + [0] * (max_seq_length-len(tokens))
        feature_input_word_ids = tokenizer.convert_tokens_to_ids(tokens) + [0] * (max_seq_length - len(tokens))
        feature_label_ids = list(map(lambda x: label_map[x], labels)) + [0] * (max_seq_length - len(labels))
        feature_label_mask.extend([0] * (max_seq_length - len(feature_label_mask)))

        # print(len(feature_ori_words), feature_ori_words)
        # print(len(feature_input_word_ids), feature_input_word_ids)
        # print(len(feature_input_mask), feature_input_mask)
        # print(len(feature_label_ids), feature_label_ids)
        # print(len(feature_label_mask), feature_label_mask)
        # if feature_input_char_ids is not None:
        #     print(len(feature_input_char_ids), feature_input_char_ids)

        features.append(InputFeatures(input_word_ids=feature_input_word_ids, input_mask=feature_input_mask, label_ids=feature_label_ids, label_mask=feature_label_mask, ori_words=feature_ori_words, input_char_ids=feature_input_char_ids))

    return features


def get_Dataset(args, task, processor, tokenizer, char2id=None, mode="train"):
    filepath = task[mode + "_file"]
    label_list = task["label_list"]

    examples = processor.get_examples(filepath)
    features = convert_examples_to_features(examples, tokenizer, label_list, args.max_seq_length, char2id)

    all_input_word_ids = torch.tensor([f.input_word_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_label_mask = torch.tensor([f.label_mask for f in features], dtype=torch.long)

    if char2id is not None:
        all_char_ids = torch.tensor([f.input_char_ids for f in features], dtype=torch.long)
        data = TensorDataset(all_input_word_ids, all_input_mask, all_label_ids, all_label_mask, all_char_ids)
    else:
        data = TensorDataset(all_input_word_ids, all_input_mask, all_label_ids, all_label_mask)

    return examples, features, data


def get_Char2id(files, tokenizer):
    char2id = {"#": 0}
    for file in files:
        with open(file, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                sep = "\t" if "\t" in line else " "
                tokens = tokenizer.tokenize(line.strip().split(sep)[0])
                for token in tokens:
                    for ch in token:
                        if ch not in char2id:
                            char2id[ch] = len(char2id)
    return char2id


if __name__ == "__main__":
    ner = NerProcessor(debug=True)
    # lines = ner.read_data("data/BC2GM-IOB/train.tsv")
    # print(lines[0])
    examples = ner.get_examples("data/BC2GM-IOB/train.tsv")
    # print(examples[0].text, examples[0].label)
    tokenizer = BertTokenizer.from_pretrained("biobert_v1.1_pubmed", do_lower_case=False)
    label_list = ["<PAD>", "I-GENE", "B-GENE", "O"]
    char2id = {'#': 0, 'I': 1, 'm': 2, 'u': 3, 'n': 4, 'o': 5, 'h': 6, 'i': 7, 's': 8, 't': 9, 'c': 10, 'e': 11,
               'a': 12, 'l': 13, 'g': 14, 'w': 15, 'p': 16, 'v': 17, 'f': 18, 'r': 19, 'S': 20, '-': 21, '1': 22,
               '0': 23, '9': 24, 'd': 25, ',': 26, 'H': 27, 'M': 28, 'B': 29, '4': 30, '5': 31, '(': 32, '%': 33,
               ')': 34, 'y': 35, 'k': 36, 'x': 37, 'b': 38, '.': 39, 'C': 40, 'E': 41, '8': 42, '6': 43, 'V': 44,
               'j': 45, '2': 46, 'R': 47, 'N': 48, 'A': 49, 'D': 50, 'z': 51, 'O': 52, '<': 53, 'q': 54, 'X': 55,
               'F': 56, '3': 57, 'G': 58, 'P': 59, ':': 60, '?': 61, 'K': 62, 'W': 63, 'T': 64, "'": 65, 'J': 66,
               'L': 67, 'U': 68, '+': 69, ';': 70, '7': 71, '/': 72, 'Z': 73, '=': 74, 'Y': 75, 'Q': 76, '[': 77,
               '"': 78, '>': 79, '*': 80, ']': 81, '&': 82, '$': 83, '_': 84}
    features = convert_examples_to_features(examples[:2], tokenizer, label_list, max_seq_length=128, char_map=char2id)

    all_input_word_ids = torch.tensor([f.input_word_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_label_mask = torch.tensor([f.label_mask for f in features], dtype=torch.long)

    if char2id is not None:
        all_char_ids = torch.tensor([f.input_char_ids for f in features], dtype=torch.long)
        data = TensorDataset(all_input_word_ids, all_input_mask, all_label_ids, all_label_mask, all_char_ids)
    else:
        data = TensorDataset(all_input_word_ids, all_input_mask, all_label_ids, all_label_mask)

