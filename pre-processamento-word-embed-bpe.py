import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors 
import csv

#-- Entrada: txt de label 1 para sequências de bpe das sequências iniciais
	  # txt de label 0 para sequências de bpe das sequências iniciais
      # modelos do word2vec para serem usados na obtenção dos vetores junto com os arquivos txt
#-- Saída: csv de labels 0 e 1 com as sequências representadas por word embeddings e o label

# prepara o dicionário de path dos dados
pathDict = {}
pathDict[0] = 'data/C. elegans/'
pathDict[1] = 'data/D. melanogaster/'
pathDict[2] = 'data/H. sapiens/'

windowSize = 5
vectorSize = 150

# avalia bpe de tamanho de vocabulário 4^5
for k in range(5,6,1):
    vocab_size = 4 ** k
    print('Iniciando processamento para vocab_size = ' + str(vocab_size) + '.')
    
    for index in range(0,3):
        print('   Iniciando processamento para dataset de index = ' + str(index) + '.')

        # carrega o modelo de word2vec já treinado
        model = Word2Vec.load(pathDict[index]+"model-word-"+str(vocab_size)+"-bpe-"+str(windowSize)+"-win-"+str(vectorSize)+"-size.model")

        # abre o arquivo no qual escreve as sequências de vetores
        vectorFile = open(pathDict[index]+"word-"+str(vocab_size)+"-bpe-"+str(windowSize)+"-win-"+str(vectorSize)+"-size.csv",'w')
        csv_writer = csv.writer(vectorFile)
        
        # abre o arquivo de sequências que formam os nucleosomos
        file = open(pathDict[index]+'seq-'+str(vocab_size)+'-bpe-forming.txt','r')
        corpus1 = []
        # lê as linhas do arquivo
        for lines in file:
            # trata a formatação para pegar a sequência
            ls = lines.strip('\n')
            # obtém os tokens da sequência em ordem
            corpus1 = ls.split(" ")
            listValues = []
            for subseq in corpus1:
                vector = model.wv.get_vector(subseq, norm=True)
                listValues.extend(vector.tolist())
            listValues.append(1)
            # printar linha com label 1
            csv_writer.writerow(listValues)
            corpus1 = []
        file.close()
        
        # abre o arquivo de sequências que inibem os nucleosomos
        file = open(pathDict[index]+'seq-'+str(vocab_size)+'-bpe-inhibiting.txt','r')
        corpus2 = []
        # lê as linhas do arquivo
        for lines in file:
            # trata a formatação para pegar a sequência
            ls = lines.strip('\n')
            # obtém os tokens da sequência em ordem
            corpus2 = ls.split(" ")
            listValues = []
            for subseq in corpus2:
                vector = model.wv.get_vector(subseq, norm=True)
                listValues.extend(vector.tolist())
            listValues.append(0)
            # printar linha com label 1
            csv_writer.writerow(listValues)
            corpus2 = []
        file.close()

        vectorFile.close()

        print('   Processamento para dataset de index = ' + str(index) + ' finalizado.')
        
    print('Processamento para vocab_size = ' + str(vocab_size) + ' finalizado.\n##############################')