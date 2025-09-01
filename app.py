import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Analisador Estatístico", layout="wide", initial_sidebar_state="collapsed")

# Header
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
    <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem;">📊 Analisador Estatístico com Classes</h1>
</div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #667eea; margin-bottom: 2rem;">
        <h4 style="color: #2c3e50; margin-top: 0; font-weight: 600;">Trabalho de Estatística – Curso de Gestão da Tecnologia da Informação – Fatec Jundiaí</h4>
        <p style="margin-bottom: 0.5rem; color: #2c3e50;"><strong>Integrantes:</strong> Anderson Martinez, Isaac Pereira, Lucas Moraes, Fabiano Matheus, Victor Hugo</p>
        <p style="margin-bottom: 0; color: #2c3e50;"><strong>Professor:</strong> João Carlos dos Santos</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### Configurações Iniciais")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Limite Mínimo**")
    L0 = st.number_input("L0:", value=10, label_visibility="collapsed")

with col2:
    st.markdown("**Amplitude da Classe**")
    h = st.number_input("h:", value=5, label_visibility="collapsed")

with col3:
    st.markdown("**Número de Classes**")
    k = st.selectbox("k:", [3, 5, 7], index=0, label_visibility="collapsed")

st.divider()

st.markdown("### Frequências das Classes")

# Frequências já preenchidas para teste
frequencias_padrao = [3, 5, 2][:k]
frequencias = []

# Criar colunas para as frequências
cols = st.columns(k)
for i in range(k):
    with cols[i]:
        st.markdown(f"**Classe {i+1}**")
        fi = st.number_input(f"fi{i+1}:", min_value=0, value=frequencias_padrao[i], key=f"fi_{i}", label_visibility="collapsed")
        frequencias.append(fi)

st.divider()

# Botão calcular centralizado e estilizado
col_btn = st.columns([1, 2, 1])
with col_btn[1]:
    calcular = st.button("Calcular Estatísticas", type="primary", use_container_width=True)

if calcular:
    limites_inferiores = [L0 + i*h for i in range(k)]
    limites_superiores = [L0 + (i+1)*h for i in range(k)]
    pontos_medios = [(li+ls)/2 for li, ls in zip(limites_inferiores, limites_superiores)]
    
    df = pd.DataFrame({
        "Limite Inferior": limites_inferiores,
        "Limite Superior": limites_superiores,
        "Frequência (fi)": frequencias,
        "Ponto Médio (xi)": pontos_medios,
        "fi*xi": [f*x for f,x in zip(frequencias,pontos_medios)]
    })
    
    n = sum(frequencias)
    media = df["fi*xi"].sum() / n
    
    # Mediana agrupada
    fac = np.cumsum(frequencias)
    N2 = n/2
    for i in range(k):
        if fac[i]>=N2:
            Li = limites_inferiores[i]
            fi_class = frequencias[i]
            F_ant = fac[i-1] if i>0 else 0
            mediana = Li + ((N2 - F_ant)/fi_class)*h
            break
    
    # Moda de Czuber
    i_moda = np.argmax(frequencias)
    f1 = frequencias[i_moda]
    f0 = frequencias[i_moda-1] if i_moda>0 else 0
    f2 = frequencias[i_moda+1] if i_moda<k-1 else 0
    moda_czuber = limites_inferiores[i_moda] + ((f1-f0)/((f1-f0)+(f1-f2)))*h if (f1-f0)+(f1-f2)!=0 else limites_inferiores[i_moda]
    moda_bruta = pontos_medios[i_moda]
    
    # Variância amostral
    df["(xi - media)^2"] = (df["Ponto Médio (xi)"] - media)**2
    df["fi*(xi-media)^2"] = df["Frequência (fi)"] * df["(xi - media)^2"]
    variancia = df["fi*(xi-media)^2"].sum() / (n-1)
    desvio_padrao = np.sqrt(variancia)
    coef_var = (desvio_padrao/media)*100
    
    # Tipo de moda
    max_fi = max(frequencias)
    qtd_modas = frequencias.count(max_fi)
    tipo_moda = "Unimodal" if qtd_modas==1 else "Bimodal" if qtd_modas==2 else "Multimodal"
    
    st.divider()
    
    st.markdown("### Tabela de Classes")
    
    # Mostrar tabela simples e clara
    st.dataframe(df.round(4), use_container_width=True)
    
    st.divider()
    
    st.markdown("### Resultados Estatísticos")
    
    # Medidas de tendência central
    st.markdown("#### Medidas de Tendência Central")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Média Agrupada",
            value=f"{media:.2f}",
            help="Média calculada usando os pontos médios das classes"
        )
    
    with col2:
        st.metric(
            label="Mediana Agrupada", 
            value=f"{mediana:.2f}",
            help="Valor que divide a distribuição ao meio"
        )
    
    with col3:
        st.metric(
            label="Moda Bruta",
            value=f"{moda_bruta:.2f}",
            help="Ponto médio da classe com maior frequência"
        )
    
    # Medidas de dispersão
    st.markdown("#### Medidas de Dispersão")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Moda de Czuber",
            value=f"{moda_czuber:.2f}",
            help="Moda calculada pela fórmula de Czuber"
        )
    
    with col2:
        st.metric(
            label="Variância",
            value=f"{variancia:.2f}",
            help="Medida de dispersão dos dados"
        )
    
    with col3:
        st.metric(
            label="Desvio Padrão",
            value=f"{desvio_padrao:.2f}",
            help="Raiz quadrada da variância"
        )
    
    with col4:
        st.metric(
            label="Coef. Variação",
            value=f"{coef_var:.2f}%",
            help="Medida relativa de dispersão"
        )
    
    # Informação adicional sobre o tipo de moda
    st.markdown("#### Classificação da Distribuição")
    
    # Determinar cor baseada no tipo de moda
    cor_moda = "#28a745" if tipo_moda == "Unimodal" else "#ffc107" if tipo_moda == "Bimodal" else "#dc3545"
    
    st.markdown(f"""
    <div style="background-color: {cor_moda}20; padding: 1rem; border-radius: 8px; border-left: 4px solid {cor_moda};">
        <h5 style="color: {cor_moda}; margin: 0;">Tipo de Distribuição: {tipo_moda}</h5>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### Visualizações Interativas")
    
    # Criar abas para os gráficos
    tab1, tab2 = st.tabs(["Histograma", "Polígono de Frequência"])
    
    with tab1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(
            x=pontos_medios, 
            y=frequencias, 
            width=[h*0.8]*k, 
            marker_color="#667eea",
            marker_line_color="#4c63d2",
            marker_line_width=2,
            hovertemplate="<b>Ponto Médio:</b> %{x}<br><b>Frequência:</b> %{y}<extra></extra>"
        ))
        fig_hist.update_layout(
            title={
                'text': "Histograma de Frequências",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="Ponto Médio das Classes",
            yaxis_title="Frequência",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            showlegend=False,
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab2:
        fig_poly = go.Figure()
        fig_poly.add_trace(go.Scatter(
            x=pontos_medios, 
            y=frequencias, 
            mode='lines+markers', 
            line=dict(color="#764ba2", width=3),
            marker=dict(size=8, color="#667eea", line=dict(width=2, color="#4c63d2")),
            hovertemplate="<b>Ponto Médio:</b> %{x}<br><b>Frequência:</b> %{y}<extra></extra>"
        ))
        fig_poly.update_layout(
            title={
                'text': "Polígono de Frequência",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="Ponto Médio das Classes",
            yaxis_title="Frequência",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            showlegend=False,
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
        )
        st.plotly_chart(fig_poly, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <p>Analisador Estatístico | Fatec Jundiaí | 2025</p>
</div>
""", unsafe_allow_html=True)
