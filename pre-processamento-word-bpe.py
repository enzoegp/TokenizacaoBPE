import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors 
import csv
from tokenizers import Tokenizer

#-- Entrada: apenas os dados como no artigo de referência
#-- Saída: txt de label 1 para sequências de bpe das sequências iniciais
	  # txt de label 0 para sequências de bpe das sequências iniciais
      # modelos do word2vec para serem usados na obtenção dos vetores junto com os arquivos txt

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
    k = 5
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

# avalia bpe de tamanho de vocabulário 4^5
for k in range(5,6,1):
    vocab_size = 4 ** k
    print('Iniciando processamento para vocab_size = ' + str(vocab_size) + '.')
    
    # corpus é usado para treinar o word2vec
    corpus = []
    for index in range(0,3):
        print('   Iniciando processamento para dataset de index = ' + str(index) + '.')
        tokenizer = Tokenizer.from_file(pathDict[index]+'model-'+str(vocab_size)+'-bpe')
        dict_bpe = tokenizer.get_vocab()
        list_tokens = list(dict_bpe.keys())
        list_tokens.sort()
        
        # abre o arquivo de sequências que formam os nucleosomos
        file = open(pathDict[index]+fileDict[index]+'f.txt','r')
        lineList = []
        corpusTemp = []
        # lê as linhas do arquivo
        for lines in file:
            # trata a formatação para pegar a sequência
            ls = lines.strip('\n')
            ls = ls[1:]
            # obtém os tokens da sequência em ordem
            encoded = tokenizer.encode(ls)
            sequence = encoded.tokens
            for padding in range(len(sequence),maior[index],1):
                sequence.append("[PAD]")
            for token in sequence:
                lineList.append(token)
            corpus.append(lineList)
            corpusTemp.append(lineList)
            lineList = []
        file.close()
        # printar sequências de bpe pra txt de label 1
        seqFile = open(pathDict[index]+'seq-'+str(vocab_size)+'-bpe-forming.txt','w')
        for i in range(len(corpusTemp)):
            print(*corpusTemp[i], file=seqFile)
        seqFile.close()
        
        # abre o arquivo de sequências que inibem os nucleosomos
        file = open(pathDict[index]+fileDict[index]+'i.txt','r')
        lineList = []
        corpusTemp = []
        # lê as linhas do arquivo
        for lines in file:
            # trata a formatação para pegar a sequência
            ls = lines.strip('\n')
            ls = ls[1:]
            # obtém os tokens da sequência em ordem
            encoded = tokenizer.encode(ls)
            sequence = encoded.tokens
            for padding in range(len(sequence),maior[index],1):
                sequence.append("[PAD]")
            for token in sequence:
                lineList.append(token)
            corpus.append(lineList)
            corpusTemp.append(lineList)
            lineList = []
        file.close()
        # printar sequências de bpe pra txt de label 0
        seqFile = open(pathDict[index]+'seq-'+str(vocab_size)+'-bpe-inhibiting.txt','w')
        for i in range(len(corpusTemp)):
            print(*corpusTemp[i], file=seqFile)
        seqFile.close()

        # usar corpus pra treinar word2vec
        # itera por tamanhos de janela 3, 5 e 7
        for windowSize in range(5,6,2):
            print('      Iniciando processamento para janela de tamanho = ' + str(windowSize) + '.')
            # itera por tamanhos de embeddings de 50, 100 e 150
            for vectorSize in range(150,151,50):
                print('         Iniciando processamento para vetor de tamanho = ' + str(vectorSize) + '.')
                # usar corpus para treinar word2vec e salva o modelo
                model = Word2Vec(corpus,min_count=1,vector_size=vectorSize,sg=1,hs=1,window=windowSize)
                model.save(pathDict[index]+"model-word-"+str(vocab_size)+"-bpe-"+str(windowSize)+"-win-"+str(vectorSize)+"-size.model")
                
                print('      Processamento para vetor de tamanho = ' + str(vectorSize) + ' finalizado.')
            print('      Processamento para janela de tamanho = ' + str(windowSize) + ' finalizado.')
        
        # zera o corpus para próxima iteração
        corpus = []

        print('   Processamento para dataset de index = ' + str(index) + ' finalizado.')
        
    print('Processamento para vocab_size = ' + str(vocab_size) + ' finalizado.\n##############################')
# após o termino da execução, rodar o outro arquivo de processamento para gerar os vetores word2vec para as sequências em si