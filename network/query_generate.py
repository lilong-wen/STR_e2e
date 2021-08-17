import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from transformers import AutoTokenizer, AutoModel
import gin


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@gin.configurable
class TextEncoder(nn.Module):
    '''
    input: string with maxlen
    output: N*1100 dim
    '''
    def __init__(self, bert_base_model, out_dim, freeze_layers):
        super().__init__()
        #init BERT
        self.bert_model = self._get_bert_basemodel(bert_base_model,freeze_layers)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_base_model, cache_dir='./cached')
        # projection MLP for BERT model
        self.bert_l1 = nn.Linear(768, 768) #768 is the size of the BERT embbedings
        self.bert_l2 = nn.Linear(768, out_dim) #768 is the size of the BERT embbedings

    def _get_bert_basemodel(self, bert_model_name, freeze_layers):
        try:
            model = AutoModel.from_pretrained(bert_model_name)#, return_dict=True)
            print("text feature extractor:", bert_model_name)
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model

    def mean_pooling(self, model_output, attention_mask):

        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask

    def encoder(self, encoded_inputs):

        encoded_inputs_tokens = self.tokenizer(encoded_inputs,
                                               return_tensors="pt",
                                               padding=True,
                                               truncation=False).to(device)

        outputs = self.bert_model(**encoded_inputs_tokens)

        with torch.no_grad():
            sentence_embeddings = self.mean_pooling(outputs, encoded_inputs_tokens['attention_mask'])

        x = self.bert_l1(sentence_embeddings)
        x = F.relu(x)
        out_emb = self.bert_l2(x)

        return out_emb

    def forward(self, encoded_inputs):

        # print(encoded_inputs)
        zls = self.encoder(encoded_inputs)

        return zls


def build_query():

    query_gen = TextEncoder()

    return query_gen
