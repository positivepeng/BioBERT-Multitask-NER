from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
from pytorch_pretrained_bert import BertAdam
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm
from utils import NerProcessor, get_Dataset
from models import BERT_BiLSTM_CRF
from pytorch_transformers import BertConfig, BertTokenizer
import conlleval

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def set_seed(seed=2020):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def evaluate(args, task_id, data, model, id2label, all_ori_words, file_name=None):
    model.eval()
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=args.train_batch_size)
    task_id = torch.tensor(task_id, dtype=torch.long).to(args.device)

    logger.info("***** Running eval *****")
    logger.info(f" Num examples = {len(data)}")

    pred_labels = []
    ori_labels = []
    for b_i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        batch = tuple(t.to(args.device) for t in batch)
        if args.need_charcnn:
            input_word_ids, input_mask, label_ids, label_mask, char_ids = batch
        else:
            input_word_ids, input_mask, label_ids, label_mask = batch
            char_ids = None

        with torch.no_grad():
            logits = model.predict(task_id, input_word_ids, char_ids, input_mask)

        # print(len(all_ori_words), [len(x) for x in all_ori_words])
        # print(len(logits), [len(x) for x in logits])
        # print(len(label_ids), [len(x) for x in label_ids])
        # print(len(input_mask), [sum(x) for x in input_mask])
        # print(len(label_mask), [sum(x) for x in label_mask])

        for predL, goldL, maskL in zip(logits, label_ids, label_mask):
            for p, g, mask in zip(predL, goldL, maskL):
                if mask.item() == 1:
                    pred_labels.append(id2label[p])
                    ori_labels.append(id2label[g.item()])
            pred_labels.append(None)
            ori_labels.append(None)
    ori_words = []
    for sent in all_ori_words:
        ori_words.extend(sent+[None])
    eval_list = []
    # print(len(pred_labels), len(ori_labels), len(ori_words))
    for plabel, olabel, word in zip(pred_labels, ori_labels, ori_words):
        if plabel is not None:
            eval_list.append(f"{word} {olabel} {plabel}\n")
        else:
            eval_list.append("\n")

    if file_name is not None:
        with open(file_name, "w", encoding="utf-8") as f:
          for line in eval_list:
            f.write(line)

    # eval the model
    counts = conlleval.evaluate(eval_list)
    conlleval.report(counts)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir_paths", default=None, type=str, nargs="+")
    parser.add_argument("--model_name_or_path", default="biobert_v1.1_pubmed", type=str)

    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--do_train", default=True, type=boolean_string)
    parser.add_argument("--do_eval", default=True, type=boolean_string)
    parser.add_argument("--from_tf", default=False, type=boolean_string)

    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)

    parser.add_argument("--bert_learning_rate", default=5e-5, type=float)
    parser.add_argument("--not_bert_learning_rate", default=5e-4, type=float)

    parser.add_argument("--num_train_epochs", default=10, type=float)
    parser.add_argument("--warmup_proprotion", default=0.2, type=float)

    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--do_lower_case", action='store_true')

    # BERT后接LSTM
    parser.add_argument("--need_birnn", default=False, type=boolean_string)
    parser.add_argument("--rnn_dim", default=128, type=int)

    # 增加charCNN
    parser.add_argument("--need_charcnn", default=False, type=boolean_string)
    parser.add_argument("--share_cnn", default=True, type=boolean_string)
    parser.add_argument("--char_embed", default=50, type=int)
    parser.add_argument("--char_out_dim", default=300, type=int)

    # 增加CNN
    parser.add_argument("--need_cnn", default=False, type=boolean_string)
    parser.add_argument("--cnn_out_dim", default=300, type=int)

    # 增加SAC
    parser.add_argument("--need_sac", action="store_true")
    parser.add_argument("--tag_num", default=2, type=int)
    parser.add_argument("--sac_factor", default=100)

    # 调试模式
    parser.add_argument("--debug", action='store_true')

    args = parser.parse_args()

    logger.info("********%s*********", "参数设置")
    logger.info("args info:")
    logger.info(args.__dict__)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_
    args.device = device
    args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info(f"device: {device} n_gpu: {args.n_gpu}")
    set_seed()
    logger.info(f"seed: {args.seed}")

    logger.info("********%s*********", "读取数据")
    data_paths = args.dir_paths
    task_names = list(map(lambda x: x.split(os.path.sep)[-1], data_paths))
    logger.info("task names: %s", str(task_names))
    logger.info("task num: %d", len(task_names))

    processor = NerProcessor(args.debug)
    tasks = [{} for i in range(len(task_names))]
    for i in range(len(task_names)):
        tasks[i]["task_id"] = i
        tasks[i]["task_name"] = task_names[i]
        tasks[i]["train_file"] = os.path.join(data_paths[i], "train_devel.tsv")
        tasks[i]["eval_file"] = os.path.join(data_paths[i], "test.tsv")
        tasks[i]["label_list"] = processor.get_labels(data_paths[i])
        tasks[i]["label2id"] = {l: i for i, l in enumerate(tasks[i]["label_list"])}
        tasks[i]["id2label"] = {value: key for key, value in tasks[i]["label2id"].items()}

    for i in range(len(tasks)):
        logger.info("tasks info %s", str(tasks[i]))

    logger.info("********%s*********", "模型加载")
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    if args.need_charcnn:
        char2id = {'#': 0, 'I': 1, 'm': 2, 'u': 3, 'n': 4, 'o': 5, 'h': 6, 'i': 7, 's': 8, 't': 9, 'c': 10, 'e': 11,
                   'a': 12, 'l': 13, 'g': 14, 'w': 15, 'p': 16, 'v': 17, 'f': 18, 'r': 19, 'S': 20, '-': 21, '1': 22,
                   '0': 23, '9': 24, 'd': 25, ',': 26, 'H': 27, 'M': 28, 'B': 29, '4': 30, '5': 31, '(': 32, '%': 33,
                   ')': 34, 'y': 35, 'k': 36, 'x': 37, 'b': 38, '.': 39, 'C': 40, 'E': 41, '8': 42, '6': 43, 'V': 44,
                   'j': 45, '2': 46, 'R': 47, 'N': 48, 'A': 49, 'D': 50, 'z': 51, 'O': 52, '<': 53, 'q': 54, 'X': 55,
                   'F': 56, '3': 57, 'G': 58, 'P': 59, ':': 60, '?': 61, 'K': 62, 'W': 63, 'T': 64, "'": 65, 'J': 66,
                   'L': 67, 'U': 68, '+': 69, ';': 70, '7': 71, '/': 72, 'Z': 73, '=': 74, 'Y': 75, 'Q': 76, '[': 77,
                   '"': 78, '>': 79, '*': 80, ']': 81, '&': 82, '$': 83, '_': 84}
    else:
        char2id = None

    config = BertConfig.from_pretrained(args.model_name_or_path)
    model = BERT_BiLSTM_CRF.from_pretrained(args.model_name_or_path, config=config,
                                            char_vocab_size=len(char2id) if char2id is not None else 0,
                                            tag_num=args.tag_num,
                                            char_embedding_dim=args.char_embed,
                                            char_out_dim=args.char_out_dim,
                                            task_infos=tasks,
                                            need_cnn=args.need_cnn,
                                            cnn_out_dim=args.cnn_out_dim,
                                            need_sac=args.need_sac,
                                            sac_factor=args.sac_factor,
                                            need_birnn=args.need_birnn,
                                            need_charcnn=args.need_charcnn,
                                            share_cnn=args.share_cnn,
                                            rnn_dim=args.rnn_dim,
                                            from_tf=args.from_tf,
                                            device=device)

    if args.do_train:
        model.to(device)

        logger.info("********%s*********", "开始读取训练集数据")
        for i in range(len(tasks)):
            tasks[i]["train_examples"], tasks[i]["train_features"], tasks[i]["train_data"] = \
                get_Dataset(args, tasks[i], processor, tokenizer, char2id, mode="train")
            train_sampler = RandomSampler(tasks[i]["train_data"])
            tasks[i]["train_dataloader"] = DataLoader(tasks[i]["train_data"], sampler=train_sampler,
                                                      batch_size=args.train_batch_size)
            tasks[i]["train_ori_words"] = [f.ori_words for f in tasks[i]["train_features"]]
            # print(tasks[i]["train_ori_words"])
        if args.do_eval:
            logger.info("********%s*********", "开始读取验证集数据")
            for i in range(len(tasks)):
                tasks[i]["eval_examples"], tasks[i]["eval_features"], tasks[i]["eval_data"] = get_Dataset(args,
                                                                                                          tasks[i],
                                                                                                          processor,
                                                                                                          tokenizer,
                                                                                                          char2id,
                                                                                                          mode="eval")
                tasks[i]["eval_ori_words"] = [f.ori_words for f in tasks[i]["eval_features"]]

        # t_total = num_train_epochs * len(train_dataloader) / gradient_accumulation_steps
        # t_total表示总共需要更新的次数
        batch_num = sum(list(map(lambda task: len(task["train_dataloader"]), tasks)))
        t_total = args.num_train_epochs * batch_num // args.gradient_accumulation_steps

        no_decay = ['bias', 'LayerNorm.weight']
        # 最原始的设置
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]

        # 为bert和非bert设置不同的学习率
        optimizer_grouped_parameters = [
            # in bert
            {'params': [p for n, p in model.named_parameters() if "bert" in n and not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01, "lr": args.bert_learning_rate},
            {'params': [p for n, p in model.named_parameters() if "bert" in n and any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, "lr": args.bert_learning_rate},
            {'params': [p for n, p in model.named_parameters() if "bert" not in n], 'weight_decay': 0.0,
             "lr": args.not_bert_learning_rate}
        ]

        # 改用BertAdam，这个优化器尽可能地模拟了原始tensorflow版bert的优化器
        optimizer = BertAdam(optimizer_grouped_parameters, warmup=args.warmup_proprotion, t_total=t_total)

        # 开始训练
        logger.info("********%s*********", "开始训练")
        logger.info("# of tasks: %d", len(tasks))
        logger.info(" Num Epochs = %d", args.num_train_epochs)
        logger.info(" Total Optimization Steps = %d", t_total)
        for task in tasks:
            logger.info(" Task name: %s Num Examples %d", task["task_name"], len(task["train_dataloader"]))

        step = 0
        total_loss = 0
        update_step = 0

        for ep in range(1, int(args.num_train_epochs) + 1):
            model.train()
            task_indexs = [i for i in range(len(tasks))]
            iter_train_dataloaders = list(map(lambda x: iter(x["train_dataloader"]), tasks))
            while True:
                if len(task_indexs) == 0:
                    break
                task_id = random.choice(task_indexs)
                task_id = torch.tensor(task_id, dtype=torch.long).to(device)
                batch = next(iter_train_dataloaders[task_id], None)
                if batch is None:
                    task_indexs.remove(task_id)
                    continue
                batch = tuple(t.to(device) for t in batch)

                if args.need_charcnn:
                    input_word_ids, input_mask, label_ids, label_mask, char_ids = batch
                else:
                    input_word_ids, input_mask, label_ids, label_mask = batch
                    char_ids = None

                if args.need_sac:
                    O_label = label_ids == tasks[task_id]["label2id"]["O"]
                    pad_label = label_ids == 0
                    tag_mask = 1 - (O_label + pad_label).long().to(device)
                else:
                    tag_mask = None

                loss = model(task_id, input_word_ids, input_mask, label_ids, char_ids, sac_mask=tag_mask)

                loss.backward()
                total_loss += loss.item()

                step += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    model.zero_grad()
                    update_step = update_step + 1
                    if update_step % 100 == 0:
                        logger.info("in ep %d, choose task: %d, loss %f", ep, task_id, loss)
            if args.do_eval:
                for task in tasks:
                    logger.info("Evalating task %s, Train set", task["task_name"])
                    train_filename, test_filename = None, None
                    if ep == args.num_train_epochs:
                        train_filename = task["task_name"] + ".train.output.txt"
                        test_filename = task["task_name"] + ".test.output.txt"
                    # evaluate(args, task["task_id"], task["train_data"], model, task["id2label"], task["train_ori_words"], file_name=train_filename)
                    logger.info("Evalating task %s, Eval set", task["task_name"])
                    evaluate(args, task["task_id"], task["eval_data"], model, task["id2label"], task["eval_ori_words"], file_name=test_filename)


if __name__ == "__main__":
    main()
