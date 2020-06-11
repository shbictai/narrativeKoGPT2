import random
import torch
import torch.nn.functional as F

def beamSearch():
  None

def random_sampling(predict, vocab, k = 1024):
  # k개 중 랜덤으로 선택된 값을 반환. 사실상 top_k와 같은 기
  gen =[]

  probs, indexs = torch.topk(predict, k=k, dim=-1)
  probs = probs.squeeze().tolist()[-1]
  indexs = indexs.squeeze().tolist()[-1]

  for i in range(len(indexs)):
    gen.append((vocab.to_tokens(indexs[i]),probs[i]))
  # print('topk word and value: ', gen)

  rand_num = random.randint(0,k-1)
  gen_word = vocab.to_tokens(indexs[rand_num])

  return gen_word


def top_p(logits, vocab, threshold = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    indexs = sorted_indices.tolist()

    sorted_softmax_logits = F.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum( sorted_softmax_logits, dim=-1)


    sorted_indices_to_remove = cum_probs > threshold
    top_p_index = 0

    # Top-p에 해당하는 index를 획득
    for i in range(len(sorted_indices_to_remove)):
      if sorted_indices_to_remove[i]== True:
        top_p_index = 0 if i==0 else i-1
        break

    # for i in range(top_p_index):
      # print('gen '+str(i)+': '+vocab.to_tokens(indexs[i]))

    rand_num = random.randint(0, top_p_index) # top-p 분포에서 랜덤 샘플링
    top_p_sample_num = indexs[rand_num]
    gen_word = vocab.to_tokens(top_p_sample_num)
    print('selected token: '+gen_word+ ' softmax value:'+str(sorted_softmax_logits[rand_num]))

    return gen_word

def top_k(predict, vocab, k):
  # topk 중 랜덤으로 선택된 값을 반환.
  gen = []

  probs, indexs = torch.topk(predict, k=k,dim=-1)
  # probs = probs.squeeze().tolist()[-1]
  # indexs = indexs.squeeze().tolist()[-1]
  probs = probs.tolist()
  indexs = indexs.tolist()

  for i in range(len(indexs)):
    gen.append((vocab.to_tokens(indexs[i]), probs[i]))
  # print('topk word and value: ', gen)

  rand_num = random.randint(0, k - 1)
  gen_word = vocab.to_tokens(indexs[rand_num])

  return gen_word
