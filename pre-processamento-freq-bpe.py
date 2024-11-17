import numpy as np
import csv
from tokenizers import Tokenizer, pre_tokenizers, processors, decoders, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

#-- Entrada: apenas os dados como no artigo de referência
#-- Saída: csv com labels 1 e 0 para frequências de bpe das sequências iniciais

# prepara o dicionário de path dos dados
pathDict = {}
pathDict[0] = 'data/C. elegans/'
pathDict[1] = 'data/D. melanogaster/'
pathDict[2] = 'data/H. sapiens/'
# prepara o dicionário de path dos arquivos acessados
fileDict = {}
fileDict[0] = 'C. elegans n-'
fileDict[1] = 'D.m n-'
fileDict[2] = 'human n-'

# avalia bpe de tamanho de vocabulário 4^1 a 4^7
for k in range(1,8,1):
    vocab_size = 4 ** k
    print('Iniciando processamento para vocab_size = ' + str(vocab_size) + '.')
    
    for index in range(0,3):
        print('   Iniciando processamento para dataset de index = ' + str(index) + '.')
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]"]
        trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=1, special_tokens=special_tokens)
        tokenizer.normalizer = normalizers.Replace(">","")
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.model = BPE(unk_token="[UNK]")
        tokenizer.train([pathDict[index]+fileDict[index]+'f.txt', pathDict[index]+fileDict[index]+'i.txt'], trainer=trainer)
        
        cls_token_id = tokenizer.token_to_id("[CLS]")
        sep_token_id = tokenizer.token_to_id("[SEP]")
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"[CLS]:0 $A:0 [SEP]:0",
            special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
        )
        tokenizer.decoder = decoders.BPEDecoder(suffix="[SEP]")
        tokenizer.save(pathDict[index]+'model-'+str(vocab_size)+'-bpe')

        print('      Imprimindo frequências de bpe.')
        # obtém o vocabulário do modelo e o ordena
        dict_bpe = tokenizer.get_vocab()
        list_tokens = list(dict_bpe.keys())
        list_tokens.sort()
        list_tokens_filtered = list(set(list_tokens) - set(special_tokens))
        list_tokens_filtered.sort()
        # gera o dicionário de contagem de tokens
        dict_token_count = dict((element,0) for element in list_tokens_filtered)
        # calcula as frequências absolutas de tokens para cada dataset
        # abre o arquivo de sequências que formam os nucleosomos
        file = open(pathDict[index]+fileDict[index]+'f.txt','r')
        freqList = []
        # lê as linhas do arquivo
        for lines in file:
            # trata a formatação para pegar a sequência
            ls = lines.strip('\n')
            ls = ls[1:]
            # zera dicionário de frequências de tokens
            dict_token_freq = dict((element,0) for element in list_tokens_filtered)
            # obtém os tokens da sequência em ordem
            encoded = tokenizer.encode(ls)
            filtered_encoded = [item for item in encoded.tokens if item not in special_tokens]
            for token in filtered_encoded:
                dict_token_freq[token] += 1.0
                dict_token_count[token] += 1.0
            # guarda valores de frequências de tokens em vetor temporário com label 1 no final
            freqListTemp = []
            for key,value in dict_token_freq.items():
                freqListTemp.append(value)
            freqListTemp = freqListTemp / (np.sum(freqListTemp))
            freqListTemp = np.append(freqListTemp,1)
            # guarda vetor temporário de frequências de k-mers e label 1 em matriz de frequências de k-mers com labels
            freqList.append(freqListTemp)
        file.close()
        
        # abre o arquivo de sequências que inibem os nucleosomos
        file = open(pathDict[index]+fileDict[index]+'i.txt','r')
        # lê as linhas do arquivo
        for lines in file:
            # trata a formatação para pegar a sequência
            ls = lines.strip('\n')
            ls = ls[1:]
            # zera dicionário de frequências de tokens
            dict_token_freq = dict((element,0) for element in list_tokens_filtered)
            # obtém os tokens da sequência em ordem
            encoded = tokenizer.encode(ls)
            filtered_encoded = [item for item in encoded.tokens if item not in special_tokens]
            for token in filtered_encoded:
                dict_token_freq[token] += 1.0
                dict_token_count[token] += 1.0
            # guarda valores de frequências de tokens em vetor temporário
            freqListTemp = []
            for key,value in dict_token_freq.items():
                freqListTemp.append(value)
            freqListTemp = freqListTemp / (np.sum(freqListTemp))
            freqListTemp = np.append(freqListTemp,0)
            # guarda vetor temporário de frequências de k-mers e label 0 em matriz de frequências de k-mers com labels
            freqList.append(freqListTemp)
        file.close()

        # printar frequências de bpe pra csv com os labels 1 e 0
        freqFile = open(pathDict[index]+'freq-'+str(vocab_size)+'-bpe.csv','w')
        csv_writer = csv.writer(freqFile)
        for i in range(len(freqList)):
            csv_writer.writerow(freqList[i].tolist())
        freqFile.close()

        countList = list(dict_token_count.values())
        countList.sort(reverse=True)
        # printar contagens de bpe pra csv
        countFile = open(pathDict[index]+'count-'+str(vocab_size)+'-bpe.csv','w')
        csv_writer = csv.writer(countFile)
        csv_writer.writerow(countList)
        countFile.close()

        print('   Processamento para dataset de index = ' + str(index) + ' finalizado.')
        
    print('Processamento para vocab_size = ' + str(vocab_size) + ' finalizado.\n##############################')
# após o termino da execução, rodar o outro arquivo de processamento para gerar os vetores word2vec para as sequências em si