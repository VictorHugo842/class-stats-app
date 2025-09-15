import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Analisador Estatístico", layout="wide", initial_sidebar_state="collapsed")

# ---------------- HEADER ----------------
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem; text-align:center;">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">📊 Analisador Estatístico</h1>
    <p style="color: #e0e0e0; margin-top: 0.5rem; font-size: 1.1rem;">App interativo de medidas estatísticas – Fatec Jundiaí</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #667eea; margin-bottom: 2rem;">
    <h4 style="color: #2c3e50; margin-top: 0; font-weight: 600;">Trabalho de Estatística – Curso de Gestão da Tecnologia da Informação – Fatec Jundiaí</h4>
    <p style="margin-bottom: 0.5rem; color: #2c3e50;"><strong>Integrantes:</strong> Anderson Martinez, Isaac Pereira, Lucas Moraes, Fabiano Matheus, Victor Hugo</p>
    <p style="margin-bottom: 0; color: #2c3e50;"><strong>Professor:</strong> João Carlos dos Santos</p>
</div>
""", unsafe_allow_html=True)

# ---------------- UNIDADE DOS DADOS ----------------
st.markdown("### Unidade dos Dados")
unidade = st.selectbox(
    "Selecione a unidade:", 
    ["Nenhuma", "Temperatura (°C)", "Valor Monetário (R$)", "Volume de Chuva (mm)", "Tempo (h)"], 
    index=0
)

# Ícones de unidade
unidade_icones = {
    "Nenhuma": "",
    "Temperatura (°C)": "°C",
    "Valor Monetário (R$)": "R$",
    "Volume de Chuva (mm)": "mm",
    "Tempo (h)": "h"
}

def format_valor_simbolo(valor, medida, unidade):
    icone = unidade_icones.get(unidade, "")
    if medida == "Coef. de Variação":
        return f"{valor:.2f}%"
    elif medida == "Variância" and icone:
        return f"{valor:.2f} {icone}²"
    elif unidade == "Valor Monetário (R$)":
        return f"{icone} {valor:.2f}"
    elif icone:
        return f"{valor:.2f}{icone}"
    else:
        return f"{valor:.2f}"

# ---------------- MODO DE AGRUPAMENTO ----------------
st.markdown("### Escolha o Tipo de Agrupamento")
modo = st.radio("Selecione o tipo de agrupamento:", ["Discreto (Xi)", "Classes"], index=1)

# ---------------- CONFIGURAÇÕES INICIAIS ----------------
if modo == "Classes":
    st.markdown("### Configurações Iniciais de Classes")
    col1, col2, col3 = st.columns(3)
    with col1:
        LI = st.number_input("LI (Limite Inferior):", value=10)
    with col2:
        H = st.number_input("H (Amplitude da Classe):", value=5)
    with col3:
        k = st.selectbox("Número de Classes:", [3,5,7], index=0)
    
    st.divider()
    st.markdown("### Frequências das Classes")
    df_classes = pd.DataFrame({
        "Limite Inferior": [LI + i*H for i in range(k)],
        "Limite Superior": [LI + (i+1)*H for i in range(k)],
        "Frequência (fi)": [3,5,2] + [0]*(k-3)
    })
    df_classes = st.data_editor(df_classes, num_rows="dynamic", key="editor_classes")
    df_classes = df_classes.fillna(0)
    frequencias = df_classes["Frequência (fi)"].tolist()
    limites_inferiores = df_classes["Limite Inferior"].tolist()
    limites_superiores = df_classes["Limite Superior"].tolist()
    pontos_medios = [(li+ls)/2 for li, ls in zip(limites_inferiores, limites_superiores)]
    h = H

else:
    st.markdown("### Valores Discretos (Xi)")
    entrada_discreta = st.radio("Como deseja informar os dados Xi?", ["Inserir individualmente", "Usar tabela"], index=0)
    
    if entrada_discreta == "Inserir individualmente":
        n_valores = st.number_input("Número de valores:", min_value=2, value=5, step=1)
        valores_padrao = [10,12,15,17,20]
        valores = []
        cols = st.columns(n_valores)
        for i in range(n_valores):
            with cols[i]:
                val = st.number_input(f"{i+1}º valor:", value=(valores_padrao[i] if i<len(valores_padrao) else 0), key=f"xi_{i}")
                valores.append(val)
        st.markdown("### Frequências (fi)")
        frequencias = []
        cols = st.columns(n_valores)
        for i in range(n_valores):
            with cols[i]:
                fi = st.number_input(f"{i+1}º fi:", min_value=0, value=1, key=f"fi_xi_{i}")
                frequencias.append(fi)
    else:
        df_discreto = pd.DataFrame({
            "Valor": [10,12,15,17,20],
            "Frequência (fi)": [3, 5, 5, 5, 2]
        })
        df_discreto = st.data_editor(df_discreto, num_rows="dynamic", key="editor_discreto_v3")
        df_discreto = df_discreto.fillna(0)
        valores = df_discreto["Valor"].tolist()
        frequencias = df_discreto["Frequência (fi)"].tolist()

    limites_inferiores = valores
    limites_superiores = valores
    pontos_medios = valores
    k = len(valores)
    h = 0

# ---------------- SELEÇÃO DE MEDIDAS ----------------
st.divider()
st.markdown("### Medidas Estatísticas Desejadas")
if modo == "Discreto (Xi)":
    medidas_opcoes = ["Média","Mediana","Moda","Variância","Desvio Padrão","Coef. de Variação"]
else:
    medidas_opcoes = ["Média","Mediana","Moda Bruta","Moda Czuber","Variância","Desvio Padrão","Coef. de Variação"]

medidas_selecionadas = st.multiselect("Selecione as medidas que deseja calcular:", medidas_opcoes, default=medidas_opcoes)

# ---------------- BOTÃO CALCULAR ----------------
calcular = st.button("Calcular Estatísticas")

if calcular:
    df = pd.DataFrame({
        "Limite Inferior": limites_inferiores,
        "Limite Superior": limites_superiores,
        "Frequência (fi)": frequencias,
        "Ponto Médio (xi)": pontos_medios,
        "fi*xi": [f*x for f,x in zip(frequencias,pontos_medios)]
    })

    n = sum(frequencias)
    media = df["fi*xi"].sum()/n if n>0 else 0

    # Mediana
    fac = np.cumsum(frequencias)
    N2 = n/2
    mediana = 0
    for i in range(k):
        if fac[i]>=N2:
            Li = limites_inferiores[i]
            fi_class = frequencias[i]
            F_ant = fac[i-1] if i>0 else 0
            mediana = Li + ((N2-F_ant)/fi_class)*h if fi_class>0 else Li
            break

    # ---------------- MODA ----------------
    max_fi = max(frequencias)
    indices_modas = [i for i,f in enumerate(frequencias)]

    # Verifica amodalidade
    if max_fi == 0 or all(f==frequencias[0] for f in frequencias):
        modas = ["∄"]
        modas_brutas = ["∄"]
        modas_czuber = ["∄"]
    else:
        # Moda genérica / Bruta
        indices_modas = [i for i,f in enumerate(frequencias) if f==max_fi]
        modas = [pontos_medios[i] for i in indices_modas][:3]
        modas_brutas = modas.copy()

        # Moda Czuber
        modas_czuber = []
        if modo == "Classes":
            for i in indices_modas[:3]:
                f1 = frequencias[i]
                f0 = frequencias[i-1] if i>0 else 0
                f2 = frequencias[i+1] if i<k-1 else 0
                if (f1-f0)+(f1-f2) != 0:
                    moda_cz = limites_inferiores[i] + ((f1-f0)/((f1-f0)+(f1-f2)))*h
                else:
                    moda_cz = "∄"
                modas_czuber.append(moda_cz)
        else:
            modas_czuber = modas.copy()

    # Variância e desvio padrão
    df["(xi-media)^2"] = (df["Ponto Médio (xi)"]-media)**2
    df["fi*(xi-media)^2"] = df["Frequência (fi)"]*df["(xi-media)^2"]
    variancia = df["fi*(xi-media)^2"].sum()/(n-1) if n>1 else 0
    desvio_padrao = np.sqrt(variancia)
    coef_var = (desvio_padrao/media)*100 if media!=0 else 0

    # ---------------- EXIBIÇÃO ----------------
    st.divider()
    st.markdown("### Resultados Estatísticos Selecionados")
    for medida in medidas_selecionadas:
        if medida == "Moda":
            valores = modas
        elif medida == "Moda Bruta":
            valores = modas_brutas
        elif medida == "Moda Czuber":
            valores = modas_czuber
        elif medida == "Média":
            valores = [media]
        elif medida == "Mediana":
            valores = [mediana]
        elif medida == "Variância":
            valores = [variancia]
        elif medida == "Desvio Padrão":
            valores = [desvio_padrao]
        elif medida == "Coef. de Variação":
            valores = [coef_var]

        cols = st.columns(len(valores))
        for idx, val in enumerate(valores):
            display_val = val if isinstance(val, str) else format_valor_simbolo(val, medida, unidade)
            cols[idx].metric(f"{medida}" + (f" #{idx+1}" if len(valores) > 1 else ""), display_val)

    # ---------------- VISUALIZAÇÕES ----------------
    st.divider()
    st.markdown("### Visualizações Interativas")
    tab1, tab2 = st.tabs(["Histograma", "Polígono de Frequência"])
    xaxis_title = "Ponto Médio" + (f" ({unidade_icones[unidade]})" if unidade!="Nenhuma" and unidade!="Valor Monetário (R$)" else "")

    with tab1:
        fig_hist = go.Figure()
        cores_barras = ["#28a745" if i in indices_modas[:3] else "#667eea" for i in range(k)]
        fig_hist.add_trace(go.Bar(
            x=pontos_medios, y=frequencias, 
            width=[h*0.8]*k if modo=="Classes" else [0.8]*k,
            marker_color=cores_barras, marker_line_color="#4c63d2", marker_line_width=2,
            hovertemplate="<b>Ponto Médio:</b> %{x}<br><b>Frequência:</b> %{y}<extra></extra>"
        ))
        fig_hist.update_layout(title={'text': "Histograma de Frequências",'x':0.5,'xanchor':'center','font':{'size':20}},
                               xaxis_title=xaxis_title, yaxis_title="Frequência",
                               plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                               font=dict(size=12), showlegend=False,
                               xaxis=dict(showgrid=True,gridwidth=1,gridcolor='lightgray'),
                               yaxis=dict(showgrid=True,gridwidth=1,gridcolor='lightgray'))
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        fig_poly = go.Figure()
        fig_poly.add_trace(go.Scatter(
            x=pontos_medios, y=frequencias, mode='lines+markers',
            line=dict(color="#764ba2", width=3),
            marker=dict(size=8, color="#667eea", line=dict(width=2,color="#4c63d2")),
            hovertemplate="<b>Ponto Médio:</b> %{x}<br><b>Frequência:</b> %{y}<extra></extra>"
        ))
        fig_poly.update_layout(title={'text': "Polígono de Frequência",'x':0.5,'xanchor':'center','font':{'size':20}},
                               xaxis_title=xaxis_title, yaxis_title="Frequência",
                               plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                               font=dict(size=12), showlegend=False,
                               xaxis=dict(showgrid=True,gridwidth=1,gridcolor='lightgray'),
                               yaxis=dict(showgrid=True,gridwidth=1,gridcolor='lightgray'))
        st.plotly_chart(fig_poly, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <p>Analisador Estatístico | Fatec Jundiaí | 2025</p>
</div>
""", unsafe_allow_html=True)
