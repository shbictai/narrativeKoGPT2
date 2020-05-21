import torch
from NarrativeKoGPT2.kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from NarrativeKoGPT2.kogpt2.utils import get_tokenizer
from NarrativeKoGPT2.model.torch_gpt2 import GPT2Config, GPT2LMHeadModel
from NarrativeKoGPT2.kogpt2.utils import download as _download
from NarrativeKoGPT2.kogpt2.utils import tokenizer
import gluonnlp as nlp


####################################################################################
# NarrativeKoGPT2 configration
####################################################################################
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
####################################################################################

tok_path = get_tokenizer()
cachedir = '~/kogpt2/' # 다운로드한 모델의 cache 경로
####################################################################################
# download model, vocab
####################################################################################

# download model
model_info = pytorch_kogpt2
model_path = _download(model_info['url'],
                       model_info['fname'],
                       model_info['chksum'],
                       cachedir=cachedir)
# download vocab
vocab_info = tokenizer
vocab_path = _download(vocab_info['url'],
                       vocab_info['fname'],
                       vocab_info['chksum'],
                       cachedir=cachedir)
####################################################################################
model_file = model_path # 다운로드된 모델 파일 경로
vocab_file = vocab_path # 다운로드된 사전 파일 경로
ctx = "cpu" # cpu, gpu 여부

kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
kogpt2model.load_state_dict(torch.load(model_file))
device = torch.device(ctx)
kogpt2model.to(device)
kogpt2model.eval()
vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(vocab_file,
                                                     mask_token=None,
                                                     sep_token=None,
                                                     cls_token=None,
                                                     unknown_token='<unk>',
                                                     padding_token='<pad>',
                                                     bos_token='<s>',
                                                     eos_token='</s>')
####################################################################################
model, vocab = kogpt2model, vocab_b_obj
tok = SentencepieceTokenizer(tok_path)

str_input='안녕 GPT야'
toked = tok(str_input)

print('toked:',[vocab.bos_token]+toked)
print('token to index :',[vocab[vocab.bos_token],]  + vocab[toked])

input_ids = torch.tensor([vocab[vocab.bos_token],]  + vocab[toked]).unsqueeze(0)
print('input_ids:',input_ids)
# predicts = model(input_ids)

"""
outputs = model(input_ids, labels=input_ids)
loss, logits = outputs[:2]
"""
predicts = model(input_ids)
pred = predicts[0]

print(pred)