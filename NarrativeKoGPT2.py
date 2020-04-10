import os
import random
import torch
from torch.utils.data import DataLoader # 데이터로더
from gluonnlp.data import SentencepieceTokenizer

from NarrativeKoGPT2.kogpt2.utils import get_tokenizer
from NarrativeKoGPT2.kogpt2.utils import download, tokenizer
from NarrativeKoGPT2.model.torch_gpt2 import GPT2Config, GPT2LMHeadModel
from NarrativeKoGPT2.util.data import NovelDataset
import gluonnlp

pytorch_kogpt2 = {
    'url':
    'https://kobert.blob.core.windows.net/models/kogpt2/pytorch/pytorch_kogpt2_676e9bcfa7.params',
    'fname': 'pytorch_kogpt2_676e9bcfa7.params',
    'chksum': '676e9bcfa7'
}
kogpt2_config = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "vocab_size": 50000
}
ctx='cpu'
cachedir='~/kogpt2/'
epoch = 200

# download model
model_info = pytorch_kogpt2
model_path = download(model_info['url'],
                       model_info['fname'],
                       model_info['chksum'],
                       cachedir=cachedir)
# download vocab
vocab_info = tokenizer
vocab_path = download(vocab_info['url'],
                       vocab_info['fname'],
                       vocab_info['chksum'],
                       cachedir=cachedir)

##############################################################################
# KoGPT-2 언어 모델 학습을 위한 GPT2LMHeadModel 선언
kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
# model_path로부터 다운로드 받은 내용을 load_state_dict으로 업로드
kogpt2model.load_state_dict(torch.load(model_path))

device = torch.device(ctx)
kogpt2model.to(device)

# kogpt2model.eval()
# 추가로 학습하기 위해
kogpt2model.train()
vocab_b_obj = gluonnlp.vocab.BERTVocab.from_sentencepiece(vocab_path,
                                                     mask_token=None,
                                                     sep_token=None,
                                                     cls_token=None,
                                                     unknown_token='<unk>',
                                                     padding_token='<pad>',
                                                     bos_token='<s>',
                                                     eos_token='</s>')
# return kogpt2model, vocab_b_obj
##############################################################################
tok_path = get_tokenizer()
model, vocab = kogpt2model, vocab_b_obj
sentencepieceTokenizer = SentencepieceTokenizer(tok_path)

os.chdir("../")
data_file_path = './data/backmyo_novel_1/untokenized_bm_data.txt'
batch_size = 8
novel_dataset = NovelDataset(data_file_path, vocab,sentencepieceTokenizer)
novel_data_loader = DataLoader(novel_dataset, batch_size=batch_size, shuffle=True)


# sentence = "그놈이 내게 한 짓들을 다 이야기해 줬을 텐데! 그걸 다 알면서도 그런 말이 나온다는 거냐? 너도 그놈의 외모와 가식적인 다정함에 반해서, 날 배신하려는 거냐?"
# tokenized_sentence = sentencepieceTokenizer(sentence)
# print('tokenized_sentence len: {} tokens: {}' . format(len(tokenized_sentence), tokenized_sentence))


# input_ids = torch.tensor([vocab[vocab.bos_token], ] + vocab[tokenized_sentence]).unsqueeze(0)
# input_ids = torch.tensor([vocab[vocab.bos_token], ] + vocab[tokenized_sentence]).unsqueeze(0)
# batch_data =[]

# for _ in range(1):
#   batch_data.append([vocab[vocab.bos_token],] + vocab[tokenized_sentence])

# batch_data =  torch.tensor(batch_data)
# print('input_ids {}' . format(input_ids))
# print('batch_data {}' . format(batch_data))

#
# outputs = model(input_ids, labels=input_ids)
# loss, logits = outputs[:2]
learning_rate = 1e-5
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epoch):
  count = 0
  for data in novel_data_loader:
    optimizer.zero_grad()
    # train_data = torch.tensor([data])
    data = torch.stack(data) # list of Tensor로 구성되어 있기 때문에 list를 stack을 통해 변환해준다.
    print(data.shape)

    data= data.transpose(1,0)
    print(data.shape)

    outputs = model(data, labels=data)
    loss, logits = outputs[:2]
    loss.backward()
    optimizer.step()
    if count %10 ==0:
      print('epoch no.{} train no.{}  loss = {}' . format(epoch, count+1, loss))
    count += 1
    # print('logits {}' . format(logits[0]))