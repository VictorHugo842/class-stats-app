# Analisador Estatístico com Classes

Bem-vindo ao **Analisador Estatístico com Classes**, uma ferramenta interativa desenvolvida em Python usando a biblioteca Streamlit. Este projeto foi criado para o trabalho da disciplina de Estatística do curso de Gestão da Tecnologia da Informação da FATEC Jundiaí. Ele permite calcular e visualizar medidas estatísticas de dados agrupados em classes de forma simples e intuitiva.

**Acesse o aplicativo online:** [https://estatisticadescritiva.streamlit.app/](https://estatisticadescritiva.streamlit.app/)

## Recursos

O aplicativo oferece funcionalidades para analisar dados agrupados, incluindo:

- **Medidas de Tendência Central:** média, mediana e moda (bruta e pela fórmula de Czuber)
- **Medidas de Dispersão:** variância, desvio padrão e coeficiente de variação
- **Classificação da Distribuição:** identificação do tipo de moda (Unimodal, Bimodal ou Multimodal)
- **Tabela de Classes Interativa:** limites das classes, frequências, pontos médios e cálculos auxiliares
- **Visualizações Dinâmicas:** histograma e polígono de frequência interativos com Plotly

## Funcionalidades Estatísticas

### Medidas de Tendência Central

- **Média Aritmética:** soma dos produtos das frequências pelos pontos médios das classes
- **Mediana:** valor que divide o conjunto de dados em duas partes iguais
- **Moda Bruta:** ponto médio da classe com maior frequência
- **Moda de Czuber:** estimativa mais precisa da moda usando interpolação

### Medidas de Dispersão

- **Variância:** grau de dispersão dos dados em relação à média
- **Desvio Padrão:** afastamento médio dos valores em relação à média
- **Coeficiente de Variação:** compara a dispersão relativa entre diferentes conjuntos de dados

**Classificação da Dispersão:**

- CV < 15%: baixa dispersão (dados homogêneos)  
- 15% ≤ CV < 30%: média dispersão  
- CV ≥ 30%: alta dispersão (dados heterogêneos)

## Como Usar Localmente

1. Atualize as dependências com:  
   `pip install -r requirements.txt`

2. Execute o aplicativo com:  
   `streamlit run seu_arquivo.py`

3. Abra o navegador, configure limite mínimo (L0), amplitude da classe (h) e número de classes (k), depois insira as frequências e clique em "Calcular Estatísticas".

## Integrantes do Grupo

Anderson Martinez, Isaac Pereira, Lucas Moraes, Fabiano Matheus e Victor Hugo

**Curso:** Gestão da Tecnologia da Informação  
**Disciplina:** Estatística  
**Professor:** João Carlos dos Santos
