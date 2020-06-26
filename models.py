import torch
import torch.nn as nn
import torch.functional as F
from torchcrf import CRF
from charCNN import CharCNN
from SAC import SAC
from pytorch_transformers import BertPreTrainedModel, BertModel


class BERT_BiLSTM_CRF(BertPreTrainedModel):

    def __init__(self, config, task_infos, need_sac, need_birnn, need_charcnn, share_cnn, char_vocab_size, char_embedding_dim, tag_num, sac_factor, tag_fearure_dim=25, char_out_dim=300, rnn_dim=128, device="cpu"):
        super(BERT_BiLSTM_CRF, self).__init__(config)

        self.num_tags = [len(task["label_list"]) for task in task_infos]
        self.need_sac = need_sac
        self.need_charcnn = need_charcnn
        self.need_birnn = need_birnn
        self.share_cnn = share_cnn
        self.char_vocab_size = char_vocab_size
        self.char_embedding_dim = char_embedding_dim

        self.bert = BertModel(config)
        self.bert_dropout = nn.Dropout(0.5)
        out_dim = config.hidden_size

        if need_charcnn:
            if not share_cnn:
                self.char_cnns = nn.ModuleList([CharCNN(vocab_size=self.char_vocab_size, embed_dim=char_embedding_dim, kernel_num=char_out_dim//3)] * len(task_infos))
            else:
                self.char_cnn = CharCNN(vocab_size=self.char_vocab_size, embed_dim=char_embedding_dim, kernel_num=char_out_dim//3)
            out_dim += char_out_dim

        if need_birnn:
            self.birnn = nn.ModuleList([nn.LSTM(out_dim, rnn_dim, bidirectional=True, batch_first=True).to(device)] * len(task_infos))
            out_dim = rnn_dim * 2

        self.hidden2tag = nn.ModuleList([nn.Linear(out_dim, self.num_tags[i]).to(device) for i in range(len(self.num_tags))])
        self.crf = nn.ModuleList([CRF(self.num_tags[i], batch_first=True).to(device) for i in range(len(self.num_tags))])

        if need_sac:
            self.sac_factor = sac_factor
            self.sac = SAC(feature_dim=out_dim, tag_fearure_dim=tag_fearure_dim, tag_num=tag_num)

    def forward(self, task_id, input_ids, input_mask, tag_ids, char_ids, sac_mask=None):
        if self.need_sac:
            emissions, tag_prob = self.tag_outputs(task_id, input_ids, char_ids, input_mask)
        else:
            emissions = self.tag_outputs(task_id, input_ids, char_ids, input_mask)
        task_id = task_id.cpu().item()
        if self.need_sac:
            tag_prob = tag_prob.reshape(-1, tag_prob.shape[-1])
            sac_mask = sac_mask.flatten()
            # print(tag_prob.shape, sac_mask.shape)
            part1 = -1 * self.crf[task_id](emissions, tag_ids, mask=input_mask.byte())
            part2 = self.sac_factor * nn.CrossEntropyLoss()(tag_prob, sac_mask)
            # print(part1, part2)
            loss = part1 + part2
        else:
            loss = -1 * self.crf[task_id](emissions, tag_ids, mask=input_mask.byte())

        return loss

    def tag_outputs(self, task_id, input_ids, char_ids, input_mask):
        outputs = self.bert(input_ids, attention_mask=input_mask)
        bert_sequence_output = self.bert_dropout(outputs[0])

        if self.need_charcnn:
            # char_ids (batch_size, max_sent_len, max_word_len)
            batch_size, max_sent_len, max_word_len = char_ids.shape
            print("char ids shape", char_ids.shape)
            batch_char_sents = char_ids.reshape(batch_size * max_sent_len, max_word_len)
            if self.share_cnn:
                char_cnn_out = self.char_cnn(batch_char_sents)
            else:
                char_cnn_out = self.char_convs[task_id](batch_char_sents)
            char_sequence_output = char_cnn_out.reshape(batch_size, -1, char_cnn_out.shape[-1])  # (batch_size, max_sent_len, char_out_dim)
            print(bert_sequence_output.shape, char_sequence_output.shape)
            sequence_output = torch.cat((bert_sequence_output, char_sequence_output), dim=2)
        else:
            sequence_output = bert_sequence_output

        if self.need_birnn:
            sequence_output, _ = self.birnn[task_id](sequence_output)

        if self.need_sac:
            sequence_output, sac_prob = self.sac(sequence_output)

        emissions = self.hidden2tag[task_id](sequence_output)

        if self.need_sac:
            return emissions, sac_prob
        else:
            return emissions

    def predict(self, task_id, input_ids, char_ids, input_mask):
        if self.need_sac:
            emissions, _ = self.tag_outputs(task_id, input_ids, char_ids, input_mask)
        else:
            emissions = self.tag_outputs(task_id, input_ids, char_ids, input_mask)
        return self.crf[task_id].decode(emissions, input_mask.byte())
