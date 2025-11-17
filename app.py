import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from scipy.optimize import curve_fit

st.set_page_config(page_title="Aplicativo Estatístico", layout="wide", initial_sidebar_state="collapsed")

# Inicializar session_state
if 'resultados_calculados' not in st.session_state:
    st.session_state.resultados_calculados = {}

# ---------------- HEADER ----------------
st.markdown("""
<div style="background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem; text-align:center;">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">📊 Aplicativo Estatístico</h1>
    <p style="color: #f1f1f1; margin-top: 0.5rem; font-size: 1.1rem;">App interativo de estatística e probabilidade – Fatec Jundiaí</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background-color: #f9fff6; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #11998e; margin-bottom: 2rem;">
    <h4 style="color: #2c3e50; margin-top: 0; font-weight: 600;">Trabalho de Estatística – Curso de Gestão da Tecnologia da Informação – Fatec Jundiaí</h4>
    <p style="margin-bottom: 0.5rem; color: #2c3e50;"><strong>Integrantes:</strong> Anderson Martinez, Fabiano Matheus, Lucas Moraes, Victor Hugo</p>
    <p style="margin-bottom: 0; color: #2c3e50;"><strong>Professor:</strong> João Carlos dos Santos</p>
</div>
""", unsafe_allow_html=True)

# ---------------- MENU PRINCIPAL ----------------
st.markdown("### Escolha o Módulo")
modulo = st.radio("Selecione o módulo:",
                  ["Estatística Descritiva", "Distribuições de Probabilidade", "Regressão Linear"],
                  horizontal=True)

# ===============================================
# MÓDULO 1: ESTATÍSTICA DESCRITIVA
# ===============================================
if modulo == "Estatística Descritiva":
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
            LI = st.number_input("LI (Limite Inferior):", value=10.0, format="%.2f")
        with col2:
            H = st.number_input("H (Amplitude da Classe):", value=5.0, format="%.2f")
        with col3:
            k = st.selectbox("Número de Classes:", [3, 5, 7], index=0)

        st.divider()
        st.markdown("### Frequências das Classes")
        df_classes = pd.DataFrame({
            "Limite Inferior": [LI + i * H for i in range(k)],
            "Limite Superior": [LI + (i + 1) * H for i in range(k)],
            "Frequência (fi)": [f for f in [3, 5, 2] + [0] * (k - 3)]
        })
        df_classes = st.data_editor(df_classes, num_rows="dynamic", key="editor_classes")
        df_classes = df_classes.fillna(0.0)
        frequencias = df_classes["Frequência (fi)"].astype(float).tolist()
        limites_inferiores = df_classes["Limite Inferior"].astype(float).tolist()
        limites_superiores = df_classes["Limite Superior"].astype(float).tolist()
        pontos_medios = [(li + ls) / 2 for li, ls in zip(limites_inferiores, limites_superiores)]
        h = H

    else:
        st.markdown("### Valores Discretos (Xi)")
        
        metodo_entrada_discreto = st.radio("Método de entrada:", 
                                           ["Lista de Valores", "Tabela Xi e Fi"], 
                                           horizontal=True,
                                           key="metodo_entrada_discreto")
        
        if metodo_entrada_discreto == "Lista de Valores":
            st.markdown("**Insira os valores separados por vírgula ou espaço:**")
            st.caption("Exemplo: 0,0,1,2,1,3,4,5 ou 0 0 1 2 1 3 4 5")
            
            valores_input = st.text_area("Valores:", value="0,0,1,2,1,3,4,5", height=100, key="valores_discretos_input")
            
            processar_valores = st.button("Processar Valores", key="processar_discreto")
            
            if processar_valores:
                try:
                    # Converter string em lista de valores
                    valores_lista = [float(x.strip()) for x in valores_input.replace(',', ' ').split() if x.strip()]
                    
                    if len(valores_lista) == 0:
                        st.error("Insira pelo menos um valor!")
                    else:
                        # Calcular frequências
                        valores_unicos, contagens = np.unique(valores_lista, return_counts=True)
                        
                        # Criar DataFrame
                        df_discreto = pd.DataFrame({
                            "Xi (Valor)": valores_unicos,
                            "Fi (Frequência)": contagens
                        })
                        
                        st.session_state.df_discreto_processado = df_discreto
                        st.success(f"✅ {len(valores_lista)} valores processados com sucesso!")
                        
                except ValueError:
                    st.error("Erro ao processar valores! Certifique-se de inserir apenas números.")
            
            if 'df_discreto_processado' in st.session_state:
                st.markdown("**Tabela de Frequências Gerada:**")
                df_discreto = st.data_editor(st.session_state.df_discreto_processado, 
                                            num_rows="dynamic", 
                                            key="editor_discreto_auto")
                df_discreto = df_discreto.fillna(0.0)
                valores = df_discreto["Xi (Valor)"].astype(float).tolist()
                frequencias = df_discreto["Fi (Frequência)"].astype(float).tolist()

                # --- CORREÇÃO --- 
                # Previne erros quando tabela está vazia ou com frequências inválidas
                if len(valores) == 0:
                    st.warning("Insira pelo menos um valor na tabela.")
                    st.stop()

                if sum(frequencias) == 0:
                    st.warning("A soma das frequências deve ser maior que zero.")
                    st.stop()

                # Mantém compatibilidade com o restante do módulo
                limites_inferiores = valores[:]
                limites_superiores = valores[:]
                pontos_medios = valores[:]
                k = len(valores)
                h = 0  # Discreto não usa amplitude, mantém zero sem quebrar nada

            else:
                # Valores padrão vazios - aguardando processamento
                valores = []
                frequencias = []
        
        else:  # Tabela Xi e Fi
            st.markdown("**Informe os valores e frequências na tabela abaixo:**")
            df_discreto = pd.DataFrame({
                "Xi (Valor)": [10.0, 12.0, 15.0, 17.0, 20.0],
                "Fi (Frequência)": [3.0, 5.0, 5.0, 5.0, 2.0]
            })
            df_discreto = st.data_editor(df_discreto, num_rows="dynamic", key="editor_discreto_manual")
            df_discreto = df_discreto.fillna(0.0)
            valores = df_discreto["Xi (Valor)"].astype(float).tolist()
            frequencias = df_discreto["Fi (Frequência)"].astype(float).tolist()

                        # --- CORREÇÃO --- 
            # Previne erros quando tabela está vazia ou com frequências inválidas
            if len(valores) == 0:
                st.warning("Insira pelo menos um valor na tabela.")
                st.stop()

            if sum(frequencias) == 0:
                st.warning("A soma das frequências deve ser maior que zero.")
                st.stop()

            # Mantém compatibilidade com o restante do módulo
            limites_inferiores = valores[:]
            limites_superiores = valores[:]
            pontos_medios = valores[:]
            k = len(valores)
            h = 0  # Discreto não usa amplitude, mantém zero sem quebrar nada

        limites_inferiores = valores
        limites_superiores = valores
        pontos_medios = valores
        k = len(valores)
        h = 0

    # ---------------- SELEÇÃO DE MEDIDAS ----------------
    st.divider()
    st.markdown("### Medidas Estatísticas Desejadas")
    if modo == "Discreto (Xi)":
        medidas_opcoes = ["Média", "Mediana", "Moda", "Variância", "Desvio Padrão", "Coef. de Variação"]
    else:
        medidas_opcoes = ["Média", "Mediana", "Moda Bruta", "Moda Czuber", "Variância", "Desvio Padrão",
                          "Coef. de Variação"]

    medidas_selecionadas = st.multiselect("Selecione as medidas que deseja calcular:", medidas_opcoes,
                                          default=medidas_opcoes)

    # ---------------- BOTÃO CALCULAR ----------------
    calcular = st.button("Calcular Estatísticas")

    if calcular:
        df = pd.DataFrame({
            "Limite Inferior": limites_inferiores,
            "Limite Superior": limites_superiores,
            "Frequência (fi)": frequencias,
            "Ponto Médio (xi)": pontos_medios,
            "fi*xi": [f * x for f, x in zip(frequencias, pontos_medios)]
        })

        n = sum(frequencias)
        media = df["fi*xi"].sum() / n if n > 0 else 0

        # Mediana
        fac = np.cumsum(frequencias)
        N2 = n / 2
        mediana = 0
        for i in range(k):
            if fac[i] >= N2:
                Li = limites_inferiores[i]
                fi_class = frequencias[i]
                F_ant = fac[i - 1] if i > 0 else 0
                mediana = Li + ((N2 - F_ant) / fi_class) * h if fi_class > 0 else Li
                break

        # ---------------- MODA ----------------
        freq_nao_zero = [f for f in frequencias if f > 0]

        # inicializa para garantir que sempre existam as variáveis
        indices_modas = []
        modas = []
        modas_brutas = []
        modas_czuber = []

        if len(freq_nao_zero) == 0:
            # sem dados válidos: mantém listas vazias (não quebra o restante)
            indices_modas = []
            modas = []
            modas_brutas = []
            modas_czuber = []
        else:            
            max_fi = max(frequencias)
            indices_modas = [i for i, f in enumerate(frequencias) if f == max_fi]
            modas = [pontos_medios[i] for i in indices_modas][:3]
            modas_brutas = modas.copy()
            modas_czuber = []

            if modo == "Classes":
                for i in indices_modas[:3]:
                    f1 = frequencias[i]
                    f0 = frequencias[i - 1] if i > 0 else 0
                    f2 = frequencias[i + 1] if i < k - 1 else 0
                    if (f1 - f0) + (f1 - f2) != 0:
                        moda_cz = limites_inferiores[i] + ((f1 - f0) / ((f1 - f0) + (f1 - f2))) * h
                    else:
                        moda_cz = pontos_medios[i]
                    modas_czuber.append(moda_cz)
            else:
                modas_czuber = modas.copy()

        # Variância e desvio padrão
        df["(xi-media)^2"] = (df["Ponto Médio (xi)"] - media) ** 2
        df["fi*(xi-media)^2"] = df["Frequência (fi)"] * df["(xi-media)^2"]
        variancia = df["fi*(xi-media)^2"].sum() / (n - 1) if n > 1 else 0
        desvio_padrao = np.sqrt(variancia)
        coef_var = (desvio_padrao / media) * 100 if media != 0 else 0

        # ---------------- EXIBIÇÃO ----------------
        st.divider()
        st.markdown("### Resultados Estatísticos Selecionados")
        for medida in medidas_selecionadas:
            if medida == "Moda":
                valores_exibir = modas
            elif medida == "Moda Bruta":
                valores_exibir = modas_brutas
            elif medida == "Moda Czuber":
                valores_exibir = modas_czuber
            elif medida == "Média":
                valores_exibir = [media]
            elif medida == "Mediana":
                valores_exibir = [mediana]
            elif medida == "Variância":
                valores_exibir = [variancia]
            elif medida == "Desvio Padrão":
                valores_exibir = [desvio_padrao]
            elif medida == "Coef. de Variação":
                valores_exibir = [coef_var]

            cols = st.columns(len(valores_exibir))
            for idx, val in enumerate(valores_exibir):
                display_val = val if isinstance(val, str) else format_valor_simbolo(val, medida, unidade)
                cols[idx].metric(f"{medida}" + (f" #{idx + 1}" if len(valores_exibir) > 1 else ""), display_val)

        # ---------------- VISUALIZAÇÕES ----------------
        st.divider()
        st.markdown("### Visualizações Interativas")
        tab1, tab2 = st.tabs(["Histograma", "Polígono de Frequência"])
        xaxis_title = "Ponto Médio" + (
            f" ({unidade_icones[unidade]})" if unidade != "Nenhuma" and unidade != "Valor Monetário (R$)" else "")

        with tab1:
            fig_hist = go.Figure()
            cores_barras = ["#28a745" if i in indices_modas[:3] else "#667eea" for i in range(k)]
            fig_hist.add_trace(go.Bar(
                x=pontos_medios, y=frequencias,
                width=[h * 0.8] * k if modo == "Classes" else [0.8] * k,
                marker_color=cores_barras, marker_line_color="#4c63d2", marker_line_width=2,
                hovertemplate="<b>Ponto Médio:</b> %{x}<br><b>Frequência:</b> %{y}<extra></extra>"
            ))
            fig_hist.update_layout(
                title={'text': "Histograma de Frequências", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
                xaxis_title=xaxis_title, yaxis_title="Frequência",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12), showlegend=False,
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'))
            st.plotly_chart(fig_hist, use_container_width=True)

        with tab2:
            fig_poly = go.Figure()
            fig_poly.add_trace(go.Scatter(
                x=pontos_medios, y=frequencias, mode='lines+markers',
                line=dict(color="#764ba2", width=3),
                marker=dict(size=8, color="#667eea", line=dict(width=2, color="#4c63d2")),
                hovertemplate="<b>Ponto Médio:</b> %{x}<br><b>Frequência:</b> %{y}<extra></extra>"
            ))
            fig_poly.update_layout(
                title={'text': "Polígono de Frequência", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
                xaxis_title=xaxis_title, yaxis_title="Frequência",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12), showlegend=False,
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'))
            st.plotly_chart(fig_poly, use_container_width=True)

# ===============================================
# MÓDULO 2: DISTRIBUIÇÕES DE PROBABILIDADE
# ===============================================
elif modulo == "Distribuições de Probabilidade":
    st.markdown("### Escolha o Tipo de Distribuição")
    tipo_var = st.radio("Tipo de Variável:", ["Discreta", "Contínua"], horizontal=True)

    if tipo_var == "Contínua":
        dist_continua = st.selectbox("Escolha a distribuição:",
                             ["Distribuição Uniforme", "Distribuição Exponencial", 
                              "Distribuição Normal (Padronizada Z)", "Distribuição Normal"])

        # ============ DISTRIBUIÇÃO UNIFORME ============
        if dist_continua == "Distribuição Uniforme":
            st.markdown("#### Distribuição Uniforme")
            st.markdown("Para uma variável aleatória X ~ U(a, b)")

            col1, col2 = st.columns(2)
            with col1:
                a_unif = st.number_input("Valor mínimo (a):", value=0.0, format="%.4f")
            with col2:
                b_unif = st.number_input("Valor máximo (b):", value=10.0, format="%.4f")

            calcular_unif = st.button("Calcular Uniforme", key="btn_unif")

            if calcular_unif:
                if b_unif <= a_unif:
                    st.error("O valor máximo (b) deve ser maior que o mínimo (a)!")
                else:
                    st.session_state.resultados_calculados['unif_calculado'] = {
                        'a': a_unif, 'b': b_unif
                    }

            if 'unif_calculado' in st.session_state.resultados_calculados:
                    dados = st.session_state.resultados_calculados['unif_calculado']
                    a_unif = dados['a']
                    b_unif = dados['b']

                    # Cálculos
                    media_unif = (a_unif + b_unif) / 2
                    variancia_unif = ((b_unif - a_unif) ** 2) / 12
                    desvio_unif = np.sqrt(variancia_unif)

                    st.markdown("### Resultados")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Média", f"{media_unif:.4f}")
                    col2.metric("Variância", f"{variancia_unif:.4f}")
                    col3.metric("Desvio Padrão", f"{desvio_unif:.4f}")

                    # Cálculo de probabilidade
                    st.markdown("#### Calcular Probabilidade P(X ≤ x)")
                    x_unif = st.number_input("Valor de x:", value=5.0, format="%.4f", key="x_unif")

                    if x_unif < a_unif:
                        prob = 0
                    elif x_unif > b_unif:
                        prob = 1
                    else:
                        prob = (x_unif - a_unif) / (b_unif - a_unif)

                    st.metric("P(X ≤ x)", f"{prob:.4f} = {prob * 100:.2f}%")

                    # Gráfico
                    x_range = np.linspace(a_unif - 2, b_unif + 2, 1000)
                    y_range = np.where((x_range >= a_unif) & (x_range <= b_unif),
                                       1 / (b_unif - a_unif), 0)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines',
                                             line=dict(color='#667eea', width=3),
                                             name='f(x)'))
                    fig.update_layout(title="Função Densidade de Probabilidade",
                                      xaxis_title="x", yaxis_title="f(x)",
                                      showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)

        # ============ DISTRIBUIÇÃO EXPONENCIAL ============
        elif dist_continua == "Distribuição Exponencial":
            st.markdown("#### Distribuição Exponencial")
            st.markdown("Para uma variável aleatória X ~ Exp(λ)")

            lambda_exp = st.number_input("Taxa λ (lambda):", value=1.0, min_value=0.01, format="%.4f")

            calcular_exp = st.button("Calcular Exponencial", key="btn_exp")

            if calcular_exp:
                st.session_state.resultados_calculados['exp_calculado'] = {'lambda': lambda_exp}

            if 'exp_calculado' in st.session_state.resultados_calculados:
                    lambda_exp = st.session_state.resultados_calculados['exp_calculado']['lambda']

                    # Cálculos
                    media_exp = 1 / lambda_exp
                    variancia_exp = 1 / (lambda_exp ** 2)
                    desvio_exp = np.sqrt(variancia_exp)

                    st.markdown("### Resultados")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Média", f"{media_exp:.4f}")
                    col2.metric("Variância", f"{variancia_exp:.4f}")
                    col3.metric("Desvio Padrão", f"{desvio_exp:.4f}")

                    # Cálculo de probabilidade
                    st.markdown("#### Calcular Probabilidade P(X ≤ x)")
                    x_exp = st.number_input("Valor de x:", value=2.0, min_value=0.0, format="%.4f", key="x_exp")

                    prob_exp = 1 - np.exp(-lambda_exp * x_exp)
                    st.metric("P(X ≤ x)", f"{prob_exp:.4f} = {prob_exp * 100:.2f}%")

                    # Gráfico
                    x_range = np.linspace(0, 5 / lambda_exp, 1000)
                    y_range = lambda_exp * np.exp(-lambda_exp * x_range)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines',
                                             line=dict(color='#764ba2', width=3),
                                             name='f(x)'))
                    fig.update_layout(title="Função Densidade de Probabilidade",
                                      xaxis_title="x", yaxis_title="f(x)",
                                      showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)

        # ============ DISTRIBUIÇÃO NORMAL PADRONIZADA ============
        elif dist_continua == "Distribuição Normal (Padronizada Z)":
            st.markdown("#### Distribuição Normal (Padronizada Z)")
            st.markdown("Para uma variável aleatória Z ~ N(0, 1)")

            tipo_calc = st.radio("Tipo de cálculo:",
                                 ["P(Z ≤ z)", "P(Z ≥ z)", "P(z1 ≤ Z ≤ z2)"],
                                 horizontal=True)

            if tipo_calc == "P(Z ≤ z)":
                z_val = st.number_input("Valor de z:", value=1.96, format="%.4f")
                if st.button("Calcular Normal"):
                    prob = stats.norm.cdf(z_val)
                    st.metric("P(Z ≤ z)", f"{prob:.4f} = {prob * 100:.2f}%")

            elif tipo_calc == "P(Z ≥ z)":
                z_val = st.number_input("Valor de z:", value=1.96, format="%.4f")
                if st.button("Calcular Normal"):
                    prob = 1 - stats.norm.cdf(z_val)
                    st.metric("P(Z ≥ z)", f"{prob:.4f} = {prob * 100:.2f}%")

            else:  # P(z1 ≤ Z ≤ z2)
                col1, col2 = st.columns(2)
                with col1:
                    z1 = st.number_input("Valor de z1:", value=-1.96, format="%.4f")
                with col2:
                    z2 = st.number_input("Valor de z2:", value=1.96, format="%.4f")

                if st.button("Calcular Normal"):
                    prob = stats.norm.cdf(z2) - stats.norm.cdf(z1)
                    st.metric("P(z1 ≤ Z ≤ z2)", f"{prob:.4f} = {prob * 100:.2f}%")

            # Gráfico da distribuição normal
            st.markdown("#### Visualização")
            x_range = np.linspace(-4, 4, 1000)
            y_range = stats.norm.pdf(x_range)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines',
                                     line=dict(color='#667eea', width=3),
                                     fill='tozeroy', name='N(0,1)'))
            fig.update_layout(title="Distribuição Normal Padronizada",
                              xaxis_title="z", yaxis_title="φ(z)",
                              showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        # ============ DISTRIBUIÇÃO NORMAL ============
        elif dist_continua == "Distribuição Normal":
            st.markdown("#### Distribuição Normal")
            st.markdown("Insira os dados para calcular média, desvio padrão e outras estatísticas")

            modo_normal = st.radio("Tipo de entrada:", 
                                   ["Dados Listados", "Dados Xi e Fi", "Dados em Classes"], 
                                   horizontal=True)

            if modo_normal == "Dados Listados":
                st.markdown("##### Insira os dados separados por vírgula ou espaço")
                dados_input = st.text_area("Dados:", value="1, 2, 2, 5, 5, 5", height=100)
                
                if st.button("Calcular Normal (Listados)"):
                    try:
                        dados = [float(x.strip()) for x in dados_input.replace(',', ' ').split() if x.strip()]
                        if len(dados) < 2:
                            st.error("Insira pelo menos 2 valores!")
                        else:
                            n = len(dados)
                            media = np.mean(dados)
                            desvio = np.std(dados, ddof=1)
                            mediana = np.median(dados)
                            
                            # Moda
                            valores_unicos, contagens = np.unique(dados, return_counts=True)
                            max_freq = contagens.max()
                            if len(set(contagens)) == 1:
                                moda = "∄ (Amodal)"
                            else:
                                modas = valores_unicos[contagens == max_freq]
                                moda = ", ".join([f"{m:.2f}" for m in modas[:3]])
                            
                            st.session_state.resultados_calculados['normal_calculado'] = {
                                'media': media, 'desvio': desvio, 'n': n, 
                                'mediana': mediana, 'moda': moda
                            }
                    except ValueError:
                        st.error("Dados inválidos! Use apenas números.")

            elif modo_normal == "Dados Xi e Fi":
                st.markdown("##### Insira valores (Xi) e frequências (Fi)")

                usar_lista_normal = st.checkbox(
                    "Inserir valores listados (ex: 1,1,2,5,5,5)",
                    key="checkbox_normal_listado"
                )

                # -------------------------------------------------------------------
                # MODO LISTADO → o usuário insere valores e o sistema gera Xi e Fi
                # -------------------------------------------------------------------
                if usar_lista_normal:

                    valores_input = st.text_area(
                        "Valores:",
                        value="1,1,2,5,5,5",
                        height=100,
                        key="valores_listados_normal_input"
                    )

                    if st.button("Gerar Tabela Xi e Fi", key="gerar_normal_listado"):
                        try:
                            lista = [
                                float(x.strip())
                                for x in valores_input.replace(",", " ").split()
                                if x.strip()
                            ]

                            if len(lista) == 0:
                                st.error("Insira pelo menos um valor!")
                            else:
                                valores_unicos, contagens = np.unique(lista, return_counts=True)

                                st.session_state.df_normal_xi = pd.DataFrame({
                                    "Xi (Valor)": valores_unicos,
                                    "Fi (Frequência)": contagens
                                })

                                st.success("Tabela gerada automaticamente!")

                        except:
                            st.error("Erro ao processar valores. Insira apenas números.")

                    # Exibir tabela gerada automaticamente
                    if "df_normal_xi" in st.session_state:
                        df_normal_xi = st.data_editor(
                            st.session_state.df_normal_xi,
                            num_rows="dynamic",
                            key="editor_normal_listado"
                        )

                    else:
                        df_normal_xi = pd.DataFrame(columns=["Xi (Valor)", "Fi (Frequência)"])

                # -------------------------------------------------------------------
                # MODO MANUAL → tabela de Xi e Fi preenchida pelo usuário
                # -------------------------------------------------------------------
                else:
                    df_normal_xi = pd.DataFrame({
                        "Xi (Valor)": [1.0, 2.0, 5.0],
                        "Fi (Frequência)": [1.0, 2.0, 3.0]
                    })

                    df_normal_xi = st.data_editor(
                        df_normal_xi,
                        num_rows="dynamic",
                        key="editor_normal_xi"
                    )

                # -------------------------------------------------------------------
                # CÁLCULOS
                # -------------------------------------------------------------------
                if st.button("Calcular Normal (Xi e Fi)", key="calcular_normal_final"):

                    df_normal_xi = df_normal_xi.dropna()

                    if len(df_normal_xi) < 2:
                        st.error("Insira pelo menos 2 valores!")
                    else:
                        xi = df_normal_xi["Xi (Valor)"].astype(float).values
                        fi = df_normal_xi["Fi (Frequência)"].astype(float).values

                        n = fi.sum()
                        media = np.sum(xi * fi) / n
                        variancia = np.sum(fi * (xi - media)**2) / (n - 1)
                        desvio = np.sqrt(variancia)

                        # Mediana
                        fac = np.cumsum(fi)
                        idx_mediana = np.where(fac >= n / 2)[0][0]
                        mediana = xi[idx_mediana]

                        # Moda (SEM AMODAL)
                        max_fi = fi.max()
                        modas = xi[fi == max_fi]
                        moda = ", ".join([f"{m:.2f}" for m in modas])

                        st.session_state.resultados_calculados['normal_calculado'] = {
                            'media': media,
                            'desvio': desvio,
                            'n': int(n),
                            'mediana': mediana,
                            'moda': moda
                        }

            else:  # Dados em Classes
                st.markdown("##### Configure as classes")
                col1, col2, col3 = st.columns(3)
                with col1:
                    LI_norm = st.number_input("LI (Limite Inferior):", value=10.0, format="%.2f", key="li_norm")
                with col2:
                    H_norm = st.number_input("H (Amplitude):", value=5.0, format="%.2f", key="h_norm")
                with col3:
                    k_norm = st.selectbox("Número de Classes:", [3, 5, 7], index=0, key="k_norm")
                
                df_classes_norm = pd.DataFrame({
                    "Limite Inferior": [LI_norm + i * H_norm for i in range(k_norm)],
                    "Limite Superior": [LI_norm + (i + 1) * H_norm for i in range(k_norm)],
                    "Fi (Frequência)": [3.0, 5.0, 2.0] + [0.0] * (k_norm - 3)
                })
                df_classes_norm = st.data_editor(df_classes_norm, num_rows="fixed", key="editor_classes_norm")
                
                if st.button("Calcular Normal (Classes)"):
                    df_classes_norm = df_classes_norm.fillna(0.0)
                    li = df_classes_norm["Limite Inferior"].astype(float).values
                    ls = df_classes_norm["Limite Superior"].astype(float).values
                    fi = df_classes_norm["Fi (Frequência)"].astype(float).values
                    xi = (li + ls) / 2
                    
                    n = fi.sum()
                    media = np.sum(xi * fi) / n
                    variancia = np.sum(fi * (xi - media)**2) / (n - 1)
                    desvio = np.sqrt(variancia)
                    
                    # Mediana
                    fac = np.cumsum(fi)
                    idx_med = np.where(fac >= n/2)[0][0]
                    F_ant = fac[idx_med - 1] if idx_med > 0 else 0
                    mediana = li[idx_med] + ((n/2 - F_ant) / fi[idx_med]) * H_norm if fi[idx_med] > 0 else xi[idx_med]
                    
                    # Moda Bruta e Czuber
                    max_fi = fi.max()
                    if len(set(fi[fi > 0])) == 1:
                        moda_bruta = "∄ (Amodal)"
                        moda_czuber = "∄ (Amodal)"
                    else:
                        idx_moda = np.where(fi == max_fi)[0][0]
                        moda_bruta = f"{xi[idx_moda]:.2f}"
                        
                        f0 = fi[idx_moda - 1] if idx_moda > 0 else 0
                        f2 = fi[idx_moda + 1] if idx_moda < len(fi) - 1 else 0
                        if (max_fi - f0) + (max_fi - f2) != 0:
                            moda_cz = li[idx_moda] + ((max_fi - f0) / ((max_fi - f0) + (max_fi - f2))) * H_norm
                            moda_czuber = f"{moda_cz:.2f}"
                        else:
                            moda_czuber = moda_bruta
                    
                    st.session_state.resultados_calculados['normal_calculado'] = {
                        'media': media, 'desvio': desvio, 'n': int(n),
                        'mediana': mediana, 'moda': moda_bruta, 'moda_czuber': moda_czuber
                    }

            # Exibir resultados
            if 'normal_calculado' in st.session_state.resultados_calculados:
                dados = st.session_state.resultados_calculados['normal_calculado']
                
                st.markdown("### Estatísticas Descritivas")
                col1, col2, col3 = st.columns(3)
                col1.metric("Média (μ)", f"{dados['media']:.4f}")
                col2.metric("Desvio Padrão (σ)", f"{dados['desvio']:.4f}")
                col3.metric("Tamanho da Amostra (n)", f"{dados['n']}")
                
                col1, col2 = st.columns(2)
                col1.metric("Mediana", f"{dados['mediana']:.4f}" if isinstance(dados['mediana'], float) else dados['mediana'])
                if modo_normal == "Dados em Classes" and 'moda_czuber' in dados:
                    col2.metric("Moda Bruta", dados['moda'])
                    st.metric("Moda Czuber", dados['moda_czuber'])
                else:
                    col2.metric("Moda", dados['moda'])
                
                # Cálculo de probabilidades
                st.markdown("### Calcular Probabilidades com Distribuição Normal")
                st.markdown(f"X ~ N({dados['media']:.2f}, {dados['desvio']:.2f}²)")
                
                tipo_prob_norm = st.radio("Tipo de cálculo:",
                                         ["P(X ≤ x)", "P(X ≥ x)", "P(x1 ≤ X ≤ x2)"],
                                         horizontal=True, key="tipo_prob_norm")
                
                if tipo_prob_norm in ["P(X ≤ x)", "P(X ≥ x)"]:
                    x_val = st.number_input("Valor de x:", value=float(dados['media']), format="%.4f", key="x_norm")
                    
                    z = (x_val - dados['media']) / dados['desvio']
                    
                    if tipo_prob_norm == "P(X ≤ x)":
                        prob = stats.norm.cdf(z)
                    else:
                        prob = 1 - stats.norm.cdf(z)
                    
                    st.metric(tipo_prob_norm, f"{prob:.4f} = {prob*100:.2f}%")
                    st.info(f"Valor Z padronizado: {z:.4f}")
                
                else:  # P(x1 ≤ X ≤ x2)
                    col1, col2 = st.columns(2)
                    with col1:
                        x1_val = st.number_input("Valor de x1:", value=float(dados['media'] - dados['desvio']), format="%.4f", key="x1_norm")
                    with col2:
                        x2_val = st.number_input("Valor de x2:", value=float(dados['media'] + dados['desvio']), format="%.4f", key="x2_norm")
                    
                    z1 = (x1_val - dados['media']) / dados['desvio']
                    z2 = (x2_val - dados['media']) / dados['desvio']
                    
                    prob = stats.norm.cdf(z2) - stats.norm.cdf(z1)
                    
                    st.metric("P(x1 ≤ X ≤ x2)", f"{prob:.4f} = {prob*100:.2f}%")
                    st.info(f"Valores Z: z1 = {z1:.4f}, z2 = {z2:.4f}")
                
                # Gráfico
                st.markdown("#### Visualização da Distribuição")
                x_range = np.linspace(dados['media'] - 4*dados['desvio'], 
                                     dados['media'] + 4*dados['desvio'], 1000)
                y_range = stats.norm.pdf(x_range, dados['media'], dados['desvio'])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines',
                                        line=dict(color='#667eea', width=3),
                                        fill='tozeroy', name=f'N({dados["media"]:.2f}, {dados["desvio"]:.2f}²)'))
                fig.update_layout(title="Distribuição Normal",
                                 xaxis_title="x", yaxis_title="f(x)",
                                 showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
    else:  # VARIÁVEIS DISCRETAS
        dist_discreta = st.selectbox("Escolha a distribuição:",
                                     ["Distribuição Binomial", "Distribuição Poisson"])

        # ============ DISTRIBUIÇÃO BINOMIAL ============
        if dist_discreta == "Distribuição Binomial":
            st.markdown("#### Distribuição Binomial")
            st.markdown("Para uma variável aleatória X ~ B(n, p)")

            col1, col2 = st.columns(2)
            with col1:
                n_binom = st.number_input("Número de tentativas (n):", value=10, min_value=1, step=1)
            with col2:
                p_binom = st.number_input("Probabilidade de sucesso (p):", value=0.5,
                                          min_value=0.0, max_value=1.0, format="%.4f")

            calcular_binom = st.button("Calcular Binomial", key="btn_binom")

            if calcular_binom:
                st.session_state.resultados_calculados['binom_calculado'] = {'n': n_binom, 'p': p_binom}

            if 'binom_calculado' in st.session_state.resultados_calculados:
                    dados = st.session_state.resultados_calculados['binom_calculado']
                    n_binom = dados['n']
                    p_binom = dados['p']

                    # Cálculos
                    media_binom = n_binom * p_binom
                    variancia_binom = n_binom * p_binom * (1 - p_binom)
                    desvio_binom = np.sqrt(variancia_binom)

                    st.markdown("### Resultados")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Média", f"{media_binom:.4f}")
                    col2.metric("Variância", f"{variancia_binom:.4f}")
                    col3.metric("Desvio Padrão", f"{desvio_binom:.4f}")

                    # Cálculo de probabilidade específica
                    st.markdown("#### Calcular Probabilidades")
                    tipo_prob = st.radio("Tipo:", ["P(X = k)", "P(X ≤ k)", "P(X ≥ k)"], horizontal=True)
                    k_binom = st.number_input("Valor de k:", value=5, min_value=0, max_value=n_binom, step=1)

                    if tipo_prob == "P(X = k)":
                        prob = stats.binom.pmf(k_binom, n_binom, p_binom)
                    elif tipo_prob == "P(X ≤ k)":
                        prob = stats.binom.cdf(k_binom, n_binom, p_binom)
                    else:  # P(X ≥ k)
                        prob = 1 - stats.binom.cdf(k_binom - 1, n_binom, p_binom)

                    st.metric(tipo_prob, f"{prob:.4f} = {prob * 100:.2f}%")

                    # Gráfico
                    x_range = np.arange(0, n_binom + 1)
                    y_range = stats.binom.pmf(x_range, n_binom, p_binom)

                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=x_range, y=y_range,
                                         marker_color='#667eea',
                                         name='P(X=k)'))
                    fig.update_layout(title="Função Massa de Probabilidade",
                                      xaxis_title="k", yaxis_title="P(X=k)",
                                      showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)

        # ============ DISTRIBUIÇÃO POISSON ============
        elif dist_discreta == "Distribuição Poisson":
            st.markdown("#### Distribuição Poisson")
            st.markdown("Para uma variável aleatória X ~ Poisson(λ)")

            lambda_pois = st.number_input("Taxa média (λ):", value=3.0, min_value=0.01, format="%.4f")

            calcular_pois = st.button("Calcular Poisson", key="btn_pois")

            if calcular_pois:
                st.session_state.resultados_calculados['pois_calculado'] = {'lambda': lambda_pois}

            if 'pois_calculado' in st.session_state.resultados_calculados:
                    lambda_pois = st.session_state.resultados_calculados['pois_calculado']['lambda']

                    # Cálculos
                    media_pois = lambda_pois
                    variancia_pois = lambda_pois
                    desvio_pois = np.sqrt(variancia_pois)

                    st.markdown("### Resultados")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Média", f"{media_pois:.4f}")
                    col2.metric("Variância", f"{variancia_pois:.4f}")
                    col3.metric("Desvio Padrão", f"{desvio_pois:.4f}")

                    # Cálculo de probabilidade específica
                    st.markdown("#### Calcular Probabilidades")
                    tipo_prob = st.radio("Tipo:", ["P(X = k)", "P(X ≤ k)", "P(X ≥ k)"], horizontal=True,
                                         key="tipo_pois")
                    k_pois = st.number_input("Valor de k:", value=3, min_value=0, step=1, key="k_pois")

                    if tipo_prob == "P(X = k)":
                        prob = stats.poisson.pmf(k_pois, lambda_pois)
                    elif tipo_prob == "P(X ≤ k)":
                        prob = stats.poisson.cdf(k_pois, lambda_pois)
                    else:  # P(X ≥ k)
                        prob = 1 - stats.poisson.cdf(k_pois - 1, lambda_pois)

                    st.metric(tipo_prob, f"{prob:.4f} = {prob * 100:.2f}%")

                    # Gráfico
                    x_range = np.arange(0, int(lambda_pois * 3) + 1)
                    y_range = stats.poisson.pmf(x_range, lambda_pois)

                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=x_range, y=y_range,
                                         marker_color='#764ba2',
                                         name='P(X=k)'))
                    fig.update_layout(title="Função Massa de Probabilidade",
                                      xaxis_title="k", yaxis_title="P(X=k)",
                                      showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)

# ===============================================
# MÓDULO 3: REGRESSÃO LINEAR
# ===============================================
elif modulo == "Regressão Linear":
    st.markdown("### Regressão Linear Simples")
    st.markdown("Modelo: Y = a + bX")

    st.markdown("#### Entrada de Dados")
    metodo_entrada = st.radio("Método de entrada:", ["Tabela Manual", "Valores Automáticos"], horizontal=True)

    if metodo_entrada == "Tabela Manual":
        df_regressao = pd.DataFrame({
            "X": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Y": [2.0, 4.0, 5.0, 4.0, 5.0]
        })
        df_regressao = st.data_editor(df_regressao, num_rows="dynamic", key="editor_regressao")
        df_regressao = df_regressao.dropna()

        X_data = df_regressao["X"].astype(float).values
        Y_data = df_regressao["Y"].astype(float).values
    else:
        col1, col2 = st.columns(2)
        with col1:
            n_pontos = st.number_input("Número de pontos:", value=10, min_value=3, max_value=100, step=1)
        with col2:
            ruido = st.slider("Nível de ruído:", 0.0, 5.0, 1.0, 0.1)

        # Gerar dados
        np.random.seed(42)
        X_data = np.linspace(0, 10, n_pontos)
        Y_data = 2 + 3 * X_data + np.random.normal(0, ruido, n_pontos)

    calcular_reg = st.button("Calcular Regressão Linear", key="btn_reg")

    if calcular_reg:
        if len(X_data) < 2:
            st.error("São necessários pelo menos 2 pontos para calcular a regressão!")
        else:
            st.session_state.resultados_calculados['reg_calculado'] = {
                'X_data': X_data.tolist(),
                'Y_data': Y_data.tolist()
            }

    if 'reg_calculado' in st.session_state.resultados_calculados:
            dados = st.session_state.resultados_calculados['reg_calculado']
            X_data = np.array(dados['X_data'])
            Y_data = np.array(dados['Y_data'])

            # Cálculos da regressão
            n = len(X_data)
            sum_x = np.sum(X_data)
            sum_y = np.sum(Y_data)
            sum_xy = np.sum(X_data * Y_data)
            sum_x2 = np.sum(X_data ** 2)
            sum_y2 = np.sum(Y_data ** 2)

            # Coeficientes
            b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            a = (sum_y - b * sum_x) / n

            # Coeficiente de correlação
            r = (n * sum_xy - sum_x * sum_y) / np.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))

            # Determinação
            r2_percent = r ** 2 * 100
            r2_percent = round(r2_percent, 2)

            # Valores preditos
            Y_pred = a + b * X_data

            # Resíduos
            residuos = Y_data - Y_pred

            # Erro padrão da estimativa
            sqe = np.sum(residuos ** 2)
            erro_padrao = np.sqrt(sqe / (n - 2)) if n > 2 else 0

            # Resultados
            st.markdown("### Resultados da Regressão")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Intercepto (a)", f"{a:.4f}")
            col2.metric("Inclinação (b)", f"{b:.4f}")
            col3.metric("Correlação (r)", f"{r:.4f}")
            col4.metric("R² (Determinação)", f"{r2_percent:.2f}%")

            col1, col2, col3 = st.columns(3)
            col1.metric("Erro Padrão", f"{erro_padrao:.4f}")
            col2.metric("Número de Pontos", f"{n}")
            
            # Calcular domínio
            x_min = X_data.min()
            x_max = X_data.max()
            col3.metric("Domínio (X)", f"[{x_min:.2f}, {x_max:.2f}]")

            # Equação
            st.markdown(f"#### Equação da Reta")
            sinal = "+" if a >= 0 else ""
            st.markdown(f"**Y = {b:.4f}X {sinal} {a:.4f}**")

            # Previsão
            st.markdown("#### Fazer Previsão")
            x_prev = st.number_input("Valor de X para prever Y:", value=float(np.mean(X_data)), format="%.4f")
            
            # Aviso de extrapolação
            if x_prev < x_min or x_prev > x_max:
                st.warning(f"⚠️ Atenção: O valor {x_prev:.4f} está fora do domínio observado [{x_min:.2f}, {x_max:.2f}]. A previsão pode não ser confiável (extrapolação).")
            
            y_prev = a + b * x_prev
            st.metric("Y previsto", f"{y_prev:.4f}")

            # Tabela de resíduos
            st.markdown("#### Análise de Resíduos")
            df_resultados = pd.DataFrame({
                "X": X_data,
                "Y Observado": Y_data,
                "Y Predito": Y_pred,
                "Resíduo": residuos
            })
            st.dataframe(df_resultados, use_container_width=True)

            # Gráficos
            st.markdown("### Visualizações")
            tab1, tab2, tab3 = st.tabs(["Dispersão e Reta", "Resíduos", "Q-Q Plot"])

            with tab1:
                fig1 = go.Figure()

                # Pontos observados
                fig1.add_trace(go.Scatter(
                    x=X_data, y=Y_data, mode='markers',
                    marker=dict(size=10, color='#667eea', line=dict(width=2, color='#4c63d2')),
                    name='Dados Observados'
                ))

                # Linha de regressão
                x_line = np.linspace(X_data.min(), X_data.max(), 100)
                y_line = a + b * x_line
                fig1.add_trace(go.Scatter(
                    x=x_line, y=y_line, mode='lines',
                    line=dict(color='#764ba2', width=3),
                    name=f'Y = {b:.2f}X + {a:.2f}'
                ))

                fig1.update_layout(
                    title="Diagrama de Dispersão com Reta de Regressão",
                    xaxis_title="X", yaxis_title="Y",
                    showlegend=True, hovermode='closest'
                )
                st.plotly_chart(fig1, use_container_width=True)

            with tab2:
                fig2 = go.Figure()

                # Resíduos vs valores preditos
                fig2.add_trace(go.Scatter(
                    x=Y_pred, y=residuos, mode='markers',
                    marker=dict(size=10, color='#667eea', line=dict(width=2, color='#4c63d2')),
                    name='Resíduos'
                ))

                # Linha zero
                fig2.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero")

                fig2.update_layout(
                    title="Gráfico de Resíduos",
                    xaxis_title="Valores Preditos", yaxis_title="Resíduos",
                    showlegend=True
                )
                st.plotly_chart(fig2, use_container_width=True)

            with tab3:
                # Q-Q Plot para normalidade dos resíduos
                residuos_padronizados = (residuos - np.mean(residuos)) / np.std(residuos)
                teoricos = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuos)))
                residuos_ordenados = np.sort(residuos_padronizados)

                fig3 = go.Figure()

                fig3.add_trace(go.Scatter(
                    x=teoricos, y=residuos_ordenados, mode='markers',
                    marker=dict(size=10, color='#667eea', line=dict(width=2, color='#4c63d2')),
                    name='Resíduos'
                ))

                # Linha de referência
                fig3.add_trace(go.Scatter(
                    x=teoricos, y=teoricos, mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Linha Teórica'
                ))

                fig3.update_layout(
                    title="Q-Q Plot (Normalidade dos Resíduos)",
                    xaxis_title="Quantis Teóricos", yaxis_title="Quantis Amostrais",
                    showlegend=True
                )
                st.plotly_chart(fig3, use_container_width=True)

            # Domínio e Contradomínio
            st.markdown("### Domínio e Contradomínio")
            
            # Calcular contradomínio (valores de Y preditos no intervalo de X)
            y_min_pred = a + b * x_min
            y_max_pred = a + b * x_max
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Domínio (X)")
                st.markdown(f"**D = [{x_min:.4f}, {x_max:.4f}]**")
                st.caption("Intervalo dos valores observados de X")
            
            with col2:
                st.markdown("#### Contradomínio (Y)")
                # O contradomínio depende da inclinação
                if b > 0:
                    st.markdown(f"**CD = [{y_min_pred:.4f}, {y_max_pred:.4f}]**")
                else:
                    st.markdown(f"**CD = [{y_max_pred:.4f}, {y_min_pred:.4f}]**")
                st.caption("Intervalo dos valores preditos de Y no domínio")

            # Interpretação
            st.markdown("### Interpretação")
            st.markdown(f"""
            - **Correlação (r = {r:.4f})**: {'Forte' if abs(r) > 0.7 else 'Moderada' if abs(r) > 0.4 else 'Fraca'} correlação {'positiva' if r > 0 else 'negativa'}
            - **R² = {r2_percent:.2f}%**: O modelo explica {r2_percent:.2f}% da variabilidade dos dados
            - **Interpretação da inclinação**: Para cada unidade de aumento em X, Y {'aumenta' if b > 0 else 'diminui'} em média {abs(b):.4f} unidades
            - **Domínio**: A regressão é válida para valores de X entre {x_min:.4f} e {x_max:.4f}
            - **Extrapolação**: Previsões fora do domínio observado podem não ser confiáveis
            """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <p>Aplicativo Estatístico | Fatec Jundiaí | 2025</p>
    <p style="font-size: 0.9em; margin-top: 0.5rem;">
        Estatística Descritiva • Distribuições de Probabilidade • Regressão Linear
    </p>
</div>
""", unsafe_allow_html=True)