import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors 
import csv
from tokenizers import Tokenizer

#-- Entrada: apenas os dados como no artigo de referência
#-- Saída: csv com labels 1 e 0 para representação oneHot das sequências iniciais

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

# prepara o dicionário de tamanho máximo de sequências de tokens de cada dataset
maior = {}
maior[0] = 0
maior[1] = 0
maior[2] = 0
for index in range(0,3):
    k = 4
    vocab_size = 4 ** k
    tokenizer = Tokenizer.from_file(pathDict[index]+'model-'+str(vocab_size)+'-bpe')
    # abre o arquivo de sequências que formam os nucleosomos
    file = open(pathDict[index]+fileDict[index]+'f.txt','r')
    # lê as linhas do arquivo
    for lines in file:
        # trata a formatação para pegar a sequência
        ls = lines.strip('\n')
        ls = ls[1:]
        # obtém os tokens da sequência em ordem
        encoded = tokenizer.encode(ls)
        value = len(encoded.tokens)
        maior[index] = value if (value > maior[index]) else maior[index]
    file.close()
    # abre o arquivo de sequências que inibem os nucleosomos
    file = open(pathDict[index]+fileDict[index]+'i.txt','r')
    # lê as linhas do arquivo
    for lines in file:
        # trata a formatação para pegar a sequência
        ls = lines.strip('\n')
        ls = ls[1:]
        # obtém os tokens da sequência em ordem
        encoded = tokenizer.encode(ls)
        value = len(encoded.tokens)
        maior[index] = value if (value > maior[index]) else maior[index]
    file.close()

# avalia vocab_size de tamanho 4^4
for k in range(4,5,1):
    vocab_size = 4 ** k
    print('Iniciando processamento para vocab_size = ' + str(vocab_size) + '.')
    
    for index in range(0,3):
        print('   Iniciando processamento para dataset de index = ' + str(index) + '.')

        # obter vocabulário de todos os bpe possíveis para este valor de k
        # vocab será usado para inicializar dicionário das frequências de bpe com valores 0
        tokenizer = Tokenizer.from_file(pathDict[index]+'model-'+str(vocab_size)+'-bpe')
        dict_bpe = tokenizer.get_vocab()
        list_tokens = list(dict_bpe.keys())
        list_tokens.sort()

        # abre o arquivo no qual escreve as sequências de vetores
        oneHotFile = open(pathDict[index]+'one-'+str(vocab_size)+'-bpe.csv','w')
        csv_writer = csv.writer(oneHotFile)
        
        # abre o arquivo de sequências que formam os nucleosomos
        file = open(pathDict[index]+fileDict[index]+'f.txt','r')
        # lê as linhas do arquivo
        for lines in file:
            # trata a formatação para pegar a sequência
            ls = lines.strip('\n')
            ls = ls[1:]
            # zera lista de oneHot
            oneHotList = []
            # obtém os tokens da sequência em ordem
            encoded = tokenizer.encode(ls)
            sequence = encoded.tokens
            for padding in range(len(sequence),maior[index],1):
                sequence.append("[PAD]")
            for token in sequence:
                # zera a representação oneHot
                oneHot = np.zeros(len(list_tokens))
                oneHot[list_tokens.index(token)] = 1.0
                oneHotList.extend(oneHot)
            # guarda vetor oneHot em matriz de oneHot com label 1 no final
            oneHotList.append(1)
            csv_writer.writerow(oneHotList)
        file.close()
        
        # abre o arquivo de sequências que inibem os nucleosomos
        file = open(pathDict[index]+fileDict[index]+'i.txt','r')
        # lê as linhas do arquivo
        for lines in file:
            # trata a formatação para pegar a sequência
            ls = lines.strip('\n')
            ls = ls[1:]
            # zera lista de oneHot
            oneHotList = []
            # obtém os tokens da sequência em ordem
            encoded = tokenizer.encode(ls)
            sequence = encoded.tokens
            for padding in range(len(sequence),maior[index],1):
                sequence.append("[PAD]")
            for token in sequence:
                # zera a representação oneHot
                oneHot = np.zeros(len(list_tokens))
                oneHot[list_tokens.index(token)] = 1.0
                oneHotList.extend(oneHot)
            # guarda vetor oneHot em matriz de oneHot com label 0 no final
            oneHotList.append(0)
            csv_writer.writerow(oneHotList)
        file.close()

        oneHotFile.close()

        print('   Processamento para dataset de index = ' + str(index) + ' finalizado.')
        
    print('Processamento para vocab_size = ' + str(vocab_size) + ' finalizado.\n##############################')
# após o termino da execução, rodar o outro arquivo de processamento para gerar os vetores word2vec para as sequências em si