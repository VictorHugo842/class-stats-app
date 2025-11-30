# Analisador Estatístico com Classes
Analisador Estatístico com Classes, uma aplicação interativa desenvolvida em Python utilizando Streamlit. Criado como trabalho da disciplina de Estatística do curso de Gestão da Tecnologia da Informação da FATEC Jundiaí, o sistema permite calcular e visualizar diversos conceitos estatísticos, incluindo dados agrupados em classes, distribuições de probabilidade, regressão e inferência estatística.

Acesse o aplicativo online:
https://estatisticadescritiva.streamlit.app/

🔧 Recursos da Aplicação

O aplicativo reúne vários módulos estatísticos, incluindo:

📘 Estatística Descritiva com Classes

Média, mediana e moda (bruta e pela fórmula de Czuber)

Variância, desvio padrão e coeficiente de variação

Classificação da distribuição (unimodal, bimodal ou multimodal)

Classificação da dispersão pelo coeficiente de variação

Tabela de classes com limites, pontos médios, frequências e cálculos auxiliares

Visualizações: histograma e polígono de frequência com Plotly

📊 Funcionalidades Estatísticas Adicionais
🔹 Distribuições de Probabilidade

Normal e Normal Padronizada (Z): probabilidades, áreas, valores críticos e curva interativa

Binomial: probabilidades individuais e acumuladas

Poisson: cálculo de ocorrências raras

Exponencial: função densidade, P(X ≤ x), P(X ≥ x) e gráfico

📈 Regressão Linear e Correlação

Cálculo dos coeficientes da reta

Gráfico de dispersão com linha de tendência

Correlação de Pearson

Valor de R²

📑 Inferência Estatística

Intervalos de confiança para médias (Z e t) e proporções

Testes de hipóteses para médias e proporções (unilateral e bilateral)

🧮 Modos de Inserção de Dados

Lista de valores

Tabela Xi e Fi

Dados agrupados em classes

▶️ Como Usar Localmente

Instale as dependências:

pip install -r requirements.txt


Execute o aplicativo:

streamlit run seu_arquivo.py


Para rodar em ambiente virtual (PowerShell):

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1


No navegador, configure L0 (limite mínimo), h (amplitude), k (número de classes), insira as frequências e clique em Calcular Estatísticas.

👥 Integrantes do Grupo

Anderson Martinez
Isaac Pereira
Lucas Moraes
Fabiano Matheus
Victor Hugo

Curso: Gestão da Tecnologia da Informação
Disciplina: Estatística
Professor: João Carlos dos Santos
