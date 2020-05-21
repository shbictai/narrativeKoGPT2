from NarrativeKoGPT2.util.data import sentencePieceTokenizer, toString

def makeDataUnderMaxTokenLen():
  # tokenizer
  sentencepieceTokenizer= sentencePieceTokenizer()

  # Files for read and write
  file = open('../data/bm_novel_1/prerpcessed_bm_novel_utf8_3.txt', 'r', encoding='utf-8')
  untokenized_file = open('../data/bm_novel_1/untokenized_bm_data.txt', 'w', encoding='utf-8')
  tokenized_file = open('../data/bm_novel_1/tokenized_bm_data.txt', 'w', encoding='utf-8')

  # Data for saving that will use on training
  untokenized = ""
  tokenized = ""
  data_length = 0

  # Preprocess datas
  while True:
    line = file.readline()

    if not line:
      untokenized_file.write(untokenized)
      tokenized_file.write(tokenized)
      break

    tokenized_line = sentencepieceTokenizer(line)

    # Data length for writing has to under 1022
    # input data can get 1024 token
    # but we need to use BOS and EOS token
    if data_length+len(tokenized_line)+2 >= 1022: # bos와 eos 토큰 갯수 고려 +2
      untokenized_file.write(untokenized+'\n')
      tokenized_file.write(tokenized+'\n')

      untokenized = ""
      tokenized = ""
      data_length = 0

    untokenized = untokenized + "<s>"+line[:-1] +"</s>"
    tokenized = tokenized + "<s>" + toString(tokenized_line) + "</s>"

    data_length = data_length+len(tokenized_line) +2 # bos와 eos 토큰 갯수 고려 +2

  file.close()
  untokenized_file.close()
  tokenized_file.close()


def getBatchData(file_path, tokenizer, vocab):

  file = open(file_path, 'r', encoding='utf-8')
  while True:
    line = file.readline()
    tokenized_line = tokenizer(line[:-1]) # 마지막 개행 문자 제거
    [vocab[vocab.bos_token], ] + vocab[tokenized_line]
    if not line:
      break

if __name__ == "__main__":
    # execute only if run as a script
    makeDataUnderMaxTokenLen()
