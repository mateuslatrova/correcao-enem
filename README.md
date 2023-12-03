# Correção automática de redações do ENEM usando aprendizado de máquina

Este projeto foi desenvolvido como o meu Trabalho de Conclusão de Curso no Bacharelado em Ciência da Computação no IME-USP.

Website com todas as demais informações do trabalho: https://linux.ime.usp.br/~mateuslatrov/mac0499/

A estrutura de pastas do repositório foi organizada da seguinte forma:

- correction: implementação dos modelos de correção
    - cluster_of_words: implementação do modelo de detecção de aglomerado de palavras
    - config_reader: implementação de uma classe para lidar com configurações dos modelos
    - data: dados crus e processados de ambos os experimentos
    - topic_deviation: implementação do modelo de detecção de fuga ao

- huggingface: notebooks de estudo da biblioteca huggingface

Na pasta raiz, também estão os arquivos das dependências de implementação do projeto, além de os arquivos contendo o poster elaborado para apresentação do trabalho, e também a monografia.

## Utilização

Para utilizar os modelos implementados, basta clonar o repositório e executar os notebooks `correction/topic_deviation/experiment.ipynb` e `correction/cluster_of_words/experiment.ipynb`. Fique à vontade para explorá-los e sugerir melhorias.