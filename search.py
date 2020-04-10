import random
import torch

def beamSearch():
  None

def randomSearch(predict, k, vocab):
  # topk 중 랜덤으로 선택된 값을 반환.
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