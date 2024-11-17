# Trabalho de redes neurais, utilizando BPE como tokenização para sequências de DNA

Este repositório contém os arquivos, datasets e resultados dos experimentos realizados na disciplina de redes neurais profundas, mais especificamente relacionados ao uso da técnica de tokenização Byte Pair Encoding (BPE) para a classificação de sequências de DNA (o BPE é muito utilizado no processamento de linguagem natural e em LLM). Todos os arquivos secundários foram deletados, porque eram muitos e ocupavam muito espaço.

Para realizar os experimentos das frequências de tokens (excluindo tokens especiais):
pre-processamento-freq-bpe.py -> classificador-svm-freq-bpe.py

Para realizar os experimentos da representação oneHot:
pre-processamento-one-bpe.py -> classificador-svm-one-bpe.py

Para realizar os experimentos dos word embeddings do Word2Vec:
pre-processamento-word-bpe.py -> pre-processamento-word-embed-bpe.py -> classificador-svm-word-bpe.py

Para realizar os experimentos dos word embeddings do FastText:
pre-processamento-fast-bpe.py -> pre-processamento-fast-embed-bpe.py -> classificador-svm-fast-bpe.py

Observações:
- Não deletar arquivos secundários (que são vários) durante um dos fluxos de execução, apenas depois
- Ajustar manualmente os tamanhos de vocabulário, de janela e de vetor nos arquivos para realizar os experimentos.
