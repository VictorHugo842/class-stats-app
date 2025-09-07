import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Analisador Estatístico", layout="wide", initial_sidebar_state="collapsed")

# ---------------- HEADER ----------------
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
    <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem;">📊 Analisador Estatístico</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #667eea; margin-bottom: 2rem;">
    <h4 style="color: #2c3e50; margin-top: 0; font-weight: 600;">Trabalho de Estatística – Curso de Gestão da Tecnologia da Informação – Fatec Jundiaí</h4>
    <p style="margin-bottom: 0.5rem; color: #2c3e50;"><strong>Integrantes:</strong> Anderson Martinez, Isaac Pereira, Lucas Moraes, Fabiano Matheus, Victor Hugo</p>
    <p style="margin-bottom: 0; color: #2c3e50;"><strong>Professor:</strong> João Carlos dos Santos</p>
</div>
""", unsafe_allow_html=True)

# ---------------- MODO DE AGRUPAMENTO ----------------
st.markdown("### Escolha o Tipo de Agrupamento")
modo = st.radio("Selecione o tipo de agrupamento:", ["Discreto (Xi)", "Classes"], index=1)

# ---------------- CONFIGURAÇÕES INICIAIS ----------------
if modo == "Classes":
    st.markdown("### Configurações Iniciais de Classes")
    col1, col2, col3 = st.columns(3)
    with col1:
        L0 = st.number_input("Limite Mínimo (L0):", value=10, label_visibility="collapsed")
    with col2:
        h = st.number_input("Amplitude da Classe (h):", value=5, label_visibility="collapsed")
    with col3:
        k = st.selectbox("Número de Classes (k):", [3,5,7], index=0, label_visibility="collapsed")
    
    st.divider()
    st.markdown("### Frequências das Classes")
    df_classes = pd.DataFrame({
        "Limite Inferior": [L0 + i*h for i in range(k)],
        "Limite Superior": [L0 + (i+1)*h for i in range(k)],
        "Frequência (fi)": [3,5,2] + [0]*(k-3)
    })
    df_classes = st.data_editor(df_classes, num_rows="dynamic", key="editor_classes")
    df_classes = df_classes.fillna(0)  # resolve None
    frequencias = df_classes["Frequência (fi)"].tolist()
    limites_inferiores = df_classes["Limite Inferior"].tolist()
    limites_superiores = df_classes["Limite Superior"].tolist()
    pontos_medios = [(li+ls)/2 for li, ls in zip(limites_inferiores, limites_superiores)]

else:
    st.markdown("### Valores Discretos (Xi)")
    entrada_discreta = st.radio("Como deseja informar os dados Xi?", ["Inserir individualmente", "Usar tabela"], index=0)
    
    if entrada_discreta == "Inserir individualmente":
        n_valores = st.number_input("Número de valores Xi:", min_value=2, value=5, step=1)
        valores_padrao = [10,12,15,17,20]
        valores = []
        cols = st.columns(n_valores)
        for i in range(n_valores):
            with cols[i]:
                val = st.number_input(f"Xi {i+1}:", value=(valores_padrao[i] if i<len(valores_padrao) else 0), key=f"xi_{i}", label_visibility="collapsed")
                valores.append(val)
        st.markdown("### Frequências (fi) dos valores Xi")
        frequencias = []
        cols = st.columns(n_valores)
        for i in range(n_valores):
            with cols[i]:
                fi = st.number_input(f"fi {i+1}:", min_value=0, value=1, key=f"fi_xi_{i}", label_visibility="collapsed")
                frequencias.append(fi)
    else:
        df_discreto = pd.DataFrame({
            "Xi": [10,12,15,17,20],
            "Frequência (fi)": [1,1,1,1,1]
        })
        df_discreto = st.data_editor(df_discreto, num_rows="dynamic", key="editor_discreto")
        df_discreto = df_discreto.fillna(0)  # resolve None
        valores = df_discreto["Xi"].tolist()
        frequencias = df_discreto["Frequência (fi)"].tolist()

    limites_inferiores = valores
    limites_superiores = valores
    pontos_medios = valores
    k = len(valores)
    h = 0  # Para mediana e moda Czuber discreto

# ---------------- SELEÇÃO DE MEDIDAS ----------------
st.divider()
st.markdown("### Medidas Estatísticas Desejadas")
opcoes_medidas = ["Média","Mediana","Moda","Moda Czuber","Variância","Desvio Padrão","Coef. de Variação"]
medidas_selecionadas = st.multiselect("Selecione as medidas que deseja calcular:", opcoes_medidas, default=opcoes_medidas)

# ---------------- BOTÃO CALCULAR ----------------
calcular = st.button("Calcular Estatísticas")

# CSS para centralizar e estilizar
st.markdown("""
    <style>
    div.stButton > button:first-child {
        display: block;
        margin: 1rem auto;  /* centraliza horizontalmente */
        padding: 0.6rem 1.5rem;
        font-size: 1rem;
        background-color: #667eea;
        color: white;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #764ba2;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)


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

    # Moda
    i_moda = np.argmax(frequencias)
    f1 = frequencias[i_moda]
    f0 = frequencias[i_moda-1] if i_moda>0 else 0
    f2 = frequencias[i_moda+1] if i_moda<k-1 else 0
    moda_czuber = limites_inferiores[i_moda] + ((f1-f0)/((f1-f0)+(f1-f2)))*h if (f1-f0)+(f1-f2)!=0 and modo=="Classes" else limites_inferiores[i_moda]
    moda_bruta = pontos_medios[i_moda]

    # Variância e desvio padrão
    df["(xi-media)^2"] = (df["Ponto Médio (xi)"]-media)**2
    df["fi*(xi-media)^2"] = df["Frequência (fi)"]*df["(xi-media)^2"]
    variancia = df["fi*(xi-media)^2"].sum()/(n-1) if n>1 else 0
    desvio_padrao = np.sqrt(variancia)
    coef_var = (desvio_padrao/media)*100 if media!=0 else 0

    # Tipo de moda
    max_fi = max(frequencias)
    qtd_modas = frequencias.count(max_fi)
    tipo_moda = "Unimodal" if qtd_modas==1 else "Bimodal" if qtd_modas==2 else "Multimodal"

    # ---------------- EXIBIÇÃO ----------------
    st.divider()
    st.markdown("### Tabela de Classes / Valores")
    st.dataframe(df.round(4), use_container_width=True)

    st.divider()
    st.markdown("### Resultados Estatísticos Selecionados")
    col_count = len(medidas_selecionadas)
    cols = st.columns(col_count)
    for idx, medida in enumerate(medidas_selecionadas):
        valor = None
        if medida=="Média":
            valor = f"{media:.2f}"
        elif medida=="Mediana":
            valor = f"{mediana:.2f}"
        elif medida=="Moda":
            valor = f"{moda_bruta:.2f}"
        elif medida=="Moda Czuber":
            valor = f"{moda_czuber:.2f}"
        elif medida=="Variância":
            valor = f"{variancia:.2f}"
        elif medida=="Desvio Padrão":
            valor = f"{desvio_padrao:.2f}"
        elif medida=="Coef. de Variação":
            valor = f"{coef_var:.2f}%"
        if valor is not None:
            cols[idx].metric(medida, valor)

    # ---------------- CARD SEPARADO PARA CLASSIFICAÇÃO ----------------
    st.divider()
    st.markdown("### Classificação da Distribuição")
    cor_moda = "#28a745" if tipo_moda=="Unimodal" else "#ffc107" if tipo_moda=="Bimodal" else "#dc3545"
    st.markdown(f"""
    <div style="background-color: {cor_moda}20; padding: 1rem; border-radius: 8px; border-left: 4px solid {cor_moda}; text-align:center;">
        <h5 style="color: {cor_moda}; margin: 0;">Tipo de Distribuição: {tipo_moda}</h5>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- VISUALIZAÇÕES ----------------
    st.divider()
    st.markdown("### Visualizações Interativas")
    tab1, tab2 = st.tabs(["Histograma", "Polígono de Frequência"])
    with tab1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(
            x=pontos_medios, y=frequencias, 
            width=[h*0.8]*k if modo=="Classes" else [0.8]*k,
            marker_color="#667eea", marker_line_color="#4c63d2", marker_line_width=2,
            hovertemplate="<b>Ponto Médio:</b> %{x}<br><b>Frequência:</b> %{y}<extra></extra>"
        ))
        fig_hist.update_layout(title={'text': "Histograma de Frequências",'x':0.5,'xanchor':'center','font':{'size':20}},
                               xaxis_title="Ponto Médio", yaxis_title="Frequência",
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
                               xaxis_title="Ponto Médio", yaxis_title="Frequência",
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
