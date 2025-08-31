# Autores: Anderson Martinez, Isaac Pereira, Lucas Moraes, Fabiano Matheus e Victor Hugo
# Trabalho de Estatística – Curso de Sistemas Embarcados – Fatec Jundiaí
# Analisador Estatístico com Classes – Estatísticas Descritivas

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Configuração da página ---
st.set_page_config(page_title="Analisador Estatístico com Classes", layout="wide")
st.title("📊 Analisador Estatístico com Classes (Nova Versão)")

st.markdown("""
**Trabalho de Estatística – Curso de Sistemas Embarcados – Fatec Jundiaí**  
**Tema:** Agrupamento em Classes (média, mediana, moda bruta e de Czuber, variância, desvio padrão e coeficiente de variação)  

**Integrantes:**  
- Anderson Martinez  
- Isaac Pereira  
- Lucas Moraes  
- Fabiano Matheus  
- Victor Hugo  
""")

# --- Entrada de dados ---
st.subheader("📥 Insira seus dados numéricos (um por linha)")
entrada = st.text_area("Cole os dados aqui:", height=200)
opcoes_classes = ['auto', 3, 5, 7, 9]  # número de classes
k_selecionado = st.selectbox("Número de classes:", opcoes_classes)

# --- Função para determinar tipo de moda ---
def tipo_moda(dados):
    from collections import Counter
    c = Counter(dados)
    freqs = list(c.values())
    max_f = max(freqs)
    qtd = freqs.count(max_f)
    if max_f == 1: return "Amodal"
    elif qtd == 1: return "Unimodal"
    elif qtd == 2: return "Bimodal"
    else: return "Multimodal"

# --- Função de análise agrupada ---
def analisar(dados, k=None):
    dados = sorted(dados)
    n = len(dados)
    minimo, maximo = min(dados), max(dados)
    amplitude_total = maximo - minimo

    if not k or k == 'auto':
        k = int(1 + 3.322 * np.log10(n))  # Regra de Sturges
    if k % 2 == 0: k += 1

    h = np.ceil(amplitude_total / k)
    limites = [(minimo + i*h, minimo + (i+1)*h) for i in range(k)]

    fi = [len([x for x in dados if lim[0] <= x < lim[1]]) for lim in limites]
    fi[-1] += dados.count(maximo)

    xi = [(lim[0]+lim[1])/2 for lim in limites]

    # Média
    media = sum(f*x for f,x in zip(fi, xi))/n

    # Mediana
    fac = np.cumsum(fi)
    n2 = n/2
    for i, f_ac in enumerate(fac):
        if f_ac >= n2:
            li = limites[i][0]
            fi_m = fi[i]
            fac_ant = fac[i-1] if i>0 else 0
            mediana = li + ((n2 - fac_ant)/fi_m)*h
            break

    # Moda
    i_moda = np.argmax(fi)
    moda_bruta = xi[i_moda]
    try:
        d1 = fi[i_moda] - fi[i_moda-1] if i_moda>0 else fi[i_moda]
        d2 = fi[i_moda] - fi[i_moda+1] if i_moda < len(fi)-1 else fi[i_moda]
        moda_czuber = limites[i_moda][0] + (d1/(d1+d2))*h
    except:
        moda_czuber = "Não aplicável"

    # Variância e desvio padrão
    variancia = sum(f*(x-media)**2 for f,x in zip(fi, xi))/(n-1)
    desvio = np.sqrt(variancia)
    cv = (desvio/media)*100

    tipo = tipo_moda(dados)

    return {
        "limites": limites,
        "fi": fi,
        "xi": xi,
        "media": media,
        "mediana": mediana,
        "moda_bruta": moda_bruta,
        "moda_czuber": moda_czuber,
        "variancia": variancia,
        "desvio": desvio,
        "cv": cv,
        "tipo_moda": tipo
    }

# --- Botão de análise ---
if st.button("✅ Analisar"):
    try:
        dados = [float(x.strip()) for x in entrada.splitlines() if x.strip()]
        if len(dados)<5:
            st.warning("Insira ao menos 5 dados para análise significativa.")
        else:
            k = None if k_selecionado=='auto' else k_selecionado
            res = analisar(dados, k)

            # --- Tabela ---
            st.subheader("📋 Tabela de Classes")
            n = len(dados)
            fac = np.cumsum(res["fi"])
            fri = [f/n for f in res["fi"]]
            frac = np.cumsum(fri)
            tabela = pd.DataFrame({
                "Limite Inferior":[f"{lim[0]:.2f}" for lim in res["limites"]],
                "Limite Superior":[f"{lim[1]:.2f}" for lim in res["limites"]],
                "Frequência (fi)": res["fi"],
                "Frequência Acumulada": fac,
                "Ponto Médio (xi)": [f"{x:.2f}" for x in res["xi"]],
                "Frequência Relativa (%)":[f"{f*100:.2f}" for f in fri],
                "Frequência Relativa Acumulada (%)":[f"{f*100:.2f}" for f in frac]
            })
            st.dataframe(tabela)

            # --- Resultados ---
            st.subheader("📈 Resultados Estatísticos")
            st.markdown(f"""
            - **Média Agrupada:** {res['media']:.2f}  
            - **Mediana Agrupada:** {res['mediana']:.2f}  
            - **Moda Bruta:** {res['moda_bruta']:.2f}  
            - **Moda de Czuber:** {res['moda_czuber'] if isinstance(res['moda_czuber'], str) else f"{res['moda_czuber']:.2f}"}  
            - **Variância:** {res['variancia']:.2f}  
            - **Desvio Padrão:** {res['desvio']:.2f}  
            - **Coeficiente de Variação:** {res['cv']:.2f}%  
            - **Tipo de Moda:** {res['tipo_moda']}
            """)

            # --- Gráficos ---
            st.subheader("📊 Gráficos")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=res["xi"], y=res["fi"], name="Frequência"))
            fig.update_layout(title="Histograma de Frequências", xaxis_title="Ponto Médio", yaxis_title="Frequência")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erro ao processar os dados: {e}")
