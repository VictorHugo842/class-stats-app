import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Analisador Estatístico com Classes", layout="wide")
st.title("📊 Analisador Estatístico com Classes (Interativo)")

# --- Inicializar dataframe ---
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame({
        "Limite Inferior": [],
        "Limite Superior": [],
        "Frequência (fi)": []
    })

df = st.session_state.df.copy()

# --- Adicionar / remover classes ---
col_add, col_remove = st.columns(2)
with col_add:
    if st.button("➕ Adicionar Classe"):
        df = pd.concat([df, pd.DataFrame({"Limite Inferior":[0],"Limite Superior":[0],"Frequência (fi)":[0]})], ignore_index=True)
with col_remove:
    if st.button("➖ Remover Última Classe") and len(df) > 0:
        df = df[:-1]

# --- Entrada de dados por linha ---
st.subheader("📥 Insira suas classes e frequências")
for i in range(len(df)):
    st.markdown(f"**Classe {i+1}**")
    col1, col2, col3 = st.columns(3)
    li = col1.number_input("Limite Inferior", value=int(df.loc[i,"Limite Inferior"]), key=f"li_{i}")
    ls = col2.number_input("Limite Superior", value=int(df.loc[i,"Limite Superior"]), key=f"ls_{i}")
    fi = col3.number_input("Frequência (fi)", value=int(df.loc[i,"Frequência (fi)"]), key=f"fi_{i}")
    df.loc[i, "Limite Inferior"] = li
    df.loc[i, "Limite Superior"] = ls
    df.loc[i, "Frequência (fi)"] = fi

st.session_state.df = df

# --- Função de cálculo ---
def calcular_estatisticas(df):
    if len(df) == 0 or df["Frequência (fi)"].sum() == 0:
        return None

    n = df["Frequência (fi)"].sum()
    df["Ponto Médio"] = (df["Limite Inferior"] + df["Limite Superior"]) / 2
    df["fi*xi"] = df["Frequência (fi)"] * df["Ponto Médio"]

    # Média
    media = df["fi*xi"].sum() / n

    # Mediana
    N2 = n / 2
    F = 0
    for i, row in df.iterrows():
        if F + row["Frequência (fi)"] >= N2:
            med_class = row
            F_ant = F
            break
        F += row["Frequência (fi)"]
    h = med_class["Limite Superior"] - med_class["Limite Inferior"]
    mediana = med_class["Limite Inferior"] + ((N2 - F_ant) / med_class["Frequência (fi)"]) * h

    # Moda (Czuber)
    modal_idx = df["Frequência (fi)"].idxmax()
    modal = df.loc[modal_idx]
    f1 = modal["Frequência (fi)"]
    f0 = df["Frequência (fi)"].iloc[modal_idx - 1] if modal_idx > 0 else 0
    f2 = df["Frequência (fi)"].iloc[modal_idx + 1] if modal_idx < len(df) - 1 else 0
    d1 = f1 - f0
    d2 = f1 - f2
    moda = modal["Limite Inferior"] + (d1 / (d1 + d2)) * h if (d1 + d2) != 0 else modal["Limite Inferior"]

    # Tipo de moda
    max_fi = df["Frequência (fi)"].max()
    modal_classes = df[df["Frequência (fi)"] == max_fi]
    if len(modal_classes) == 1:
        tipo_moda = "Unimodal"
    elif len(modal_classes) == 2:
        tipo_moda = "Bimodal"
    else:
        tipo_moda = "Multimodal"

    # Variância e Desvio Padrão
    df["(xi - media)^2"] = (df["Ponto Médio"] - media) ** 2
    df["fi*(xi - media)^2"] = df["Frequência (fi)"] * df["(xi - media)^2"]
    variancia = df["fi*(xi - media)^2"].sum() / n
    desvio_padrao = np.sqrt(variancia)
    cv = (desvio_padrao / media) * 100

    return media, mediana, moda, tipo_moda, variancia, desvio_padrao, cv

# --- Botão calcular ---
if st.button("✅ Calcular Estatísticas"):
    stats = calcular_estatisticas(df)
    if stats:
        media, mediana, moda, tipo_moda, variancia, desvio_padrao, cv = stats

        st.subheader("📋 Resultados Estatísticos")
        st.write(f"Média: {media:.2f}")
        st.write(f"Mediana: {mediana:.2f}")
        st.write(f"Moda (Czuber): {moda:.2f}")
        st.write(f"Tipo de Moda: {tipo_moda}")
        st.write(f"Variância: {variancia:.2f}")
        st.write(f"Desvio Padrão: {desvio_padrao:.2f}")
        st.write(f"Coeficiente de Variação: {cv:.2f}%")

        st.subheader("📈 Gráficos Interativos")

        # Histograma interativo
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(
            x=df["Ponto Médio"],
            y=df["Frequência (fi)"],
            width=(df["Limite Superior"] - df["Limite Inferior"])*0.8,
            name="Frequência",
            marker_color="#1f77b4"
        ))
        fig_hist.update_layout(title="Histograma de Frequências",
                               xaxis_title="Ponto Médio", yaxis_title="Frequência")

        # Polígono de frequência interativo
        fig_poly = go.Figure()
        fig_poly.add_trace(go.Scatter(
            x=df["Ponto Médio"],
            y=df["Frequência (fi)"],
            mode='lines+markers',
            name="Frequência",
            line=dict(color="#ff7f0e", width=2)
        ))
        fig_poly.update_layout(title="Polígono de Frequência",
                               xaxis_title="Ponto Médio", yaxis_title="Frequência")

        st.plotly_chart(fig_hist, use_container_width=True)
        st.plotly_chart(fig_poly, use_container_width=True)
    else:
        st.warning("Insira pelo menos uma classe com frequência maior que 0.")
