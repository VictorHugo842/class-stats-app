# 📊 Analisador Estatístico com Classes

O **Analisador Estatístico com Classes** é uma aplicação interativa desenvolvida em Python utilizando **Streamlit**. Criado como trabalho da disciplina de **Estatística** da **FATEC Jundiaí**, o sistema calcula e visualiza diversos conceitos estatísticos, incluindo dados agrupados, distribuições de probabilidade, regressão linear e módulos de inferência estatística.

🌐 **Acesse o aplicativo online:**  
https://estatisticadescritiva.streamlit.app/

---

## 🔧 Recursos da Aplicação

A ferramenta é dividida em módulos que facilitam o estudo e o cálculo estatístico.

---

## 📘 Estatística Descritiva com Dados Agrupados

- Média, mediana e moda (bruta e pela fórmula de Czuber)  
- Variância, desvio padrão e coeficiente de variação  
- Classificação da distribuição:
  - Unimodal  
  - Bimodal  
  - Multimodal  
- Classificação da dispersão (coeficiente de variação)  
- Tabela dinâmica com:
  - Limites  
  - Amplitude  
  - Pontos médios  
  - Frequências (fi)  
  - Cálculos auxiliares  
- Visualizações:
  - 📊 Histograma  
  - 📈 Polígono de frequência (Plotly)

---

## 📊 Distribuições de Probabilidade

### 🔹 Normal e Normal Padronizada (Z)
- Cálculo de probabilidades  
- Áreas e valores críticos  
- Curva interativa

### 🔹 Binomial
- Probabilidades individuais  
- Probabilidades acumuladas  

### 🔹 Poisson
- Cálculo de ocorrências raras  

### 🔹 Exponencial
- Função densidade  
- Probabilidades:
  - P(X ≤ x)  
  - P(X ≥ x)  
- Gráfico interativo  

---

## 📈 Regressão Linear e Correlação

- Coeficientes da reta (a e b)  
- Gráfico de dispersão com linha de tendência  
- Correlação de Pearson (r)  
- Coeficiente de determinação (R²)  

---

## 📑 Inferência Estatística

- Intervalos de confiança:
  - Para médias (Z e t)  
  - Para proporções  
- Testes de hipótese:
  - Unilaterais  
  - Bilaterais  
  - Para média (Z e t)  
  - Para proporções  

---

## 🧮 Modos de Inserção de Dados

- Lista de valores  
- Tabela Xi e Fi  
- Dados agrupados  

---

## ▶️ Como Usar Localmente

### 📦 Instale as dependências
```bash
pip install -r requirements.txt
streamlit run app.py
