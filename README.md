# Trabalho Parcial – Aprendizado de Máquina
**Classificação de Identidades Faciais com CelebA**

## Descrição do problema

Este trabalho aborda um problema de **classificação supervisionada de imagens**, utilizando um subconjunto do dataset **CelebA (CelebFaces Attributes Dataset)**. O objetivo é identificar corretamente a **identidade** associada a uma imagem facial, a partir de características visuais extraídas automaticamente.

O dataset CelebA contém mais de 200 mil imagens faciais de mais de 10 mil identidades distintas, além de anotações adicionais como atributos binários, landmarks faciais e partições de treino e teste. Devido ao grande volume de dados, foi utilizado apenas um **subconjunto balanceado** de identidades para viabilizar os experimentos dentro das restrições computacionais.

Cada classe no problema corresponde a uma **identidade distinta**, e o modelo deve aprender a associar uma imagem facial à identidade correta.

Neste trabalho, o foco está especificamente na tarefa de **identificação facial**, na qual cada imagem deve ser associada a uma identidade conhecida presente no conjunto de treinamento. A tarefa de autenticação facial é discutida conceitualmente, mas não foi explorada experimentalmente nesta implementação.


---

## Dataset

- **Dataset:** CelebA
- **Número total de imagens:** 202.599
- **Número total de identidades:** 10.177
- **Anotações utilizadas:**
  - `identity_CelebA.txt` (mapeamento imagem → identidade)
  - Arquivos auxiliares de atributos e landmarks (disponíveis, mas não explorados diretamente neste trabalho)

Para evitar a seleção manual de imagens, foi utilizado o arquivo `identity_CelebA.txt` para **agrupar automaticamente as imagens por identidade** e construir um subconjunto contendo apenas algumas classes, com número controlado de imagens por classe.

---

## Metodologia

O pipeline de aprendizado de máquina adotado segue as seguintes etapas:

1. **Seleção automática do subconjunto**
   - Leitura do arquivo `identity_CelebA.txt`
   - Agrupamento das imagens por identidade
   - Seleção de um número fixo de identidades e imagens por identidade

2. **Pré-processamento**
   - Redimensionamento das imagens
   - Conversão para tons de cinza
   - Normalização dos dados

3. **Extração de características**
   - Utilização do descritor **HOG (Histogram of Oriented Gradients)** para transformar cada imagem em um vetor de características numéricas

4. **Divisão dos dados**
   - Separação em conjunto de treino e teste
   - Split estratificado para manter a proporção de classes

5. **Classificação**
   - Utilização de um classificador **SVM linear (LinearSVC)** treinado sobre as características HOG

6. **Avaliação**
   - Avaliação do desempenho utilizando **acurácia** no conjunto de teste

---

## Modelos Implementados

O projeto utiliza modelos supervisionados clássicos, incluindo:

- **Modelos Lineares (LinearSVC / SVM C-SVC)**
- Pipeline completo:
  - Extração de descritores
  - Normalização (StandardScaler)
  - Treinamento
  - Avaliação

A seleção de parâmetros é realizada de forma controlada, respeitando as
recomendações de balanceamento e validação cruzada.

---

## Estratégia de Treinamento e Avaliação

- Divisão treino / teste estratificada
- A estrutura do código permite a extensão para **k-fold cross-validation (k=5)**, conforme sugerido no enunciado
- Salvamento automático de:
  - Configuração (`config.txt`)
  - Evolução do erro (`error.txt`)
  - Modelo treinado (`model.dat`)

São gerados múltiplos cenários experimentais, incluindo:
- Modelo de melhor desempenho
- Modelo de pior desempenho
- Diferentes descritores

---

## Reprodutibilidade

Todas as execuções:

- Utilizam **seed fixa**
- Salvam a configuração completa do experimento
- Permitem reaplicação do modelo treinado em novas imagens

---

## Dependências

Principais bibliotecas utilizadas:

- Python ≥ 3.10
- NumPy
- scikit-learn
- scikit-image
- OpenCV (opcional)

---

## Como executar

1. Criar o ambiente virtual e instalar as dependências
2. Executar o script de criação do subconjunto:
```bash
   python make_subset.py
```
